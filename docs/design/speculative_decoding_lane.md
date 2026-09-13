# Speculative decoding lane

sglang-omni reuses SGLang's native speculative workers (`StandaloneWorkerV2`,
`EAGLEWorkerV2`, `DFlashWorkerV2`, `DSparkWorkerV2`); Omni never runs a draft/verify loop of its
own. What Omni owns is the wiring, and that wiring is split three ways so a
model-specific behavior can be separated from scheduler mechanics:

| Layer | Where | What lives there |
|---|---|---|
| Checkpoint-declared | the draft checkpoint | draft vocab / hidden size, aux hidden-state layer ids (`eagle_config`), block size, token maps. Omni does not re-declare these. |
| Config-time validation | `<family>/engine_builder.py` (`SpeculativeModelSupport.validate_speculative_draft`) | whether this draft can serve this target family (architecture, vocab, audio token ids, positional capacity, decoder budget). |
| Narrow runtime interface + bridge | `scheduling/speculative_model_support.py`, `scheduling/speculative_bridge.py`, `scheduling/speculative_support.py` | the few per-family runtime needs that no other system provides (draft view of a multimodal request, encoder-decoder position reserve / encoder sharing, request admission) and the model-agnostic scheduler bridge. |

## Modules

- `scheduling/speculative_support.py` (L1, algorithm rules): admission through
  `SpeculativeAlgorithm.is_standalone/is_eagle3/is_dflash/is_dspark`, the window rule per algorithm (chain: `N == steps + 1`,
  `topk == 1`; DFLASH: `N` is the block size; DSPARK: `N` is the
  proposal count plus one, with one draft step), whether an eager
  prefill leaves draft hidden state (`eager_prefill_has_hidden`; STANDALONE
  does not), the draft graph kind a stock `init_cuda_graphs()` captures, and
  the graph policy (`speculative_prefill_cuda_graph` opt-in, default off).
  `engine_factory.configure_speculative_overrides` is a composition of these;
  shared factory policy stays in this module; family builders still validate
  their supported combinations.
- `scheduling/speculative_bridge.py` (L0, Omni scheduler bridge): the pieces
  SGLang's own scheduler would do around a V2 worker — synchronous dispatch
  (`run_speculative_batch` → `draft_worker.forward_batch_generation`), result
  handoff (`next_draft_input`, `new_seq_lens`), terminal trimming
  (`finished_output_ids`, honouring `req.finished_len`), the STANDALONE
  hidden-state clear, `SpecInput.filter_batch` keyword compatibility, the
  future-map creation and the lookahead refusal. `cumulate_accepted_tokens`
  restores DSPARK history before drafting, including unequal accepted lengths
  and batch filter/merge transitions.
- `scheduling/bootstrap.py`: worker creation (`SpeculativeAlgorithm.create_worker`),
  shared pool/allocator/attention initialisation in SGLang's order, draft KV
  pricing by `layer_info.num_effective_layers`, and one draft graph helper
  (`init_speculative_draft_cuda_graphs`: signature dispatch first, then the
  algorithm's draft graph kind).
- `model_runner/model_worker.py`: `SpeculativeTargetWorkerAdapter` (target
  prefill/verify through Omni's runner, optional Omni prefill hooks) and the
  target registry. `register_speculative_target(arch, omni_model_cls=None,
  draft_archs=())` is called by each family's engine builder; model_worker
  holds no architecture list.
- `scheduling/speculative_model_support.py` (L2): `SpeculativeModelSupport`
  with defaults — `validate_speculative_draft`, `reject_speculative_request`,
  `draft_view_of_request` (identity), `position_reserve` (0), `normalize_draft_checkpoint`, `link_models`
  (assigns `spec_role`, alias `encoder_share_role`).

## Families on this branch

| Family | Algorithm | Support object | Registry |
|---|---|---|---|
| Whisper (encoder-decoder, needs the SGLang enc-dec patch) | STANDALONE | `WhisperSpeculativeSupport`: `position_reserve = 2 * N`, encoder-output sharing via `EncoderStateShare`, timestamp requests refused | Omni `WhisperForConditionalGeneration` class registered for the draft loader |
| Qwen3-ASR | STANDALONE | `Qwen3ASRSpeculativeSupport`: draft re-encodes from retained mel (`draft_view_mm_inputs`, salted mm hash) | Omni `Qwen3ASRForConditionalGeneration` class registered |
| Qwen3-Omni thinker | EAGLE3 / DFLASH | default support; the thinker implements SGLang's `get_embed_and_head` / `set_eagle3_layers_to_capture` / `set_dflash_layers_to_capture`; visual inputs refused | nothing registered (stock drafts) |

| CosyVoice3 | EAGLE3 / DSPARK | `CosyVoice3SpeculativeSupport`: checkpoint normalization; live sampling hooks on the target model | Speech-vocabulary EAGLE3 and DSPARK draft classes registered |

## How to add a family

1. Make the target model class implement the SGLang hooks the algorithm
   needs (`get_embed_and_head`, `set_eagle3_layers_to_capture`, ...). Do not
   rename them; SGLang calls them.
2. Implement a `SpeculativeModelSupport` subclass only for what the defaults
   do not cover (draft compatibility check, request refusal, a draft-owned
   encoder's input view, an encoder-decoder position reserve) and return it
   from the builder's `speculative_model_support()`.
3. Call `register_speculative_target(<arch>, omni_model_cls=...)` at the top
   of the family's engine builder (or before `create_sglang_infrastructure`
   in a bespoke bootstrap); pass `omni_model_cls` only when the draft must
   resolve an Omni model class through SGLang's registry.
4. Expose the factory knobs (`enable_speculative`, `speculative_algorithm`
   where more than one applies, `speculative_draft_model_path`,
   `speculative_num_steps`, `speculative_num_draft_tokens`,
   `speculative_cuda_graph`, `speculative_draft_cuda_graph`,
   `speculative_prefill_cuda_graph`) on the builder and typed factory args.
5. Add a request-admission call in the family's request builder if some
   request shapes cannot be verified natively (`reject_speculative_request`).

## Sampling and block layout

The synchronous bridge prepares the batch before calling the native worker:

1. Compose any Omni prefill embeddings.
2. For DSPARK, restore prepared penalizers from committed request histories.
   Rebuilding follows request rows after filter/merge and is idempotent; it
   also restores minimum-output-length state.
3. Refresh sampling penalties and publish the live batch through the optional
   model hook `set_speculative_sampling_info`.
4. Dispatch through SGLang. The target adapter calls the optional model hook
   `postprocess_speculative_logits` after eager forward or graph replay and
   before native verification.

CosyVoice3 owns the per-row verify repetition penalty and draft mirror. These
hooks are outside captured model forward. They do not establish exact native
stochastic verification for every penalty, nor refresh Python state captured
inside a folded draft graph. DFLASH sampling/history ownership remains outside
this migration's validated contract; its existing restricted path is preserved.

For the published CosyVoice3 DSPARK checkpoint, eight query rows comprise one
bonus anchor plus seven proposals. The target verifies eight positions.
`resolve_draft_window` uses SGLang's checkpoint parser, checks explicit overrides,
and `assert_native_reads` rejects a runtime that ignores the anchor declaration.
The native bonus-anchor patch is required; graph sizing and proposal slicing
remain SGLang responsibilities.

## Remaining integration work

This migration keeps `omni_scheduler.py` unchanged. It does not establish the
stronger claim that adding a family requires zero changes to every shared file:
CosyVoice3 still needs its speech-vocabulary model-config override in
`model_worker.py`. Its builder owns compatibility checks and invokes the support
normalizer. Migrating those construction details fully into family declarations
is a separate step. Native overlap integration is also separate.
