# Speculative decoding lane

sglang-omni reuses SGLang's native speculative workers (`StandaloneWorkerV2`,
`EAGLEWorkerV2`, `DFlashWorkerV2`); Omni never runs a draft/verify loop of its
own. What Omni owns is the wiring, and that wiring is split three ways so a
new model family only writes model-side code:

| Layer | Where | What lives there |
|---|---|---|
| Checkpoint-declared | the draft checkpoint | draft vocab / hidden size, aux hidden-state layer ids (`eagle_config`), block size, token maps. Omni does not re-declare these. |
| Config-time validation | `<family>/engine_builder.py` (`SpeculativeModelSupport.validate_speculative_draft`) | whether this draft can serve this target family (architecture, vocab, audio token ids, positional capacity, decoder budget). |
| Narrow runtime interface + bridge | `scheduling/speculative_model_support.py`, `scheduling/speculative_bridge.py`, `scheduling/speculative_support.py` | the few per-family runtime needs that no other system provides (draft view of a multimodal request, encoder-decoder position reserve / encoder sharing, request admission) and the model-agnostic scheduler bridge. |

## Modules

- `scheduling/speculative_support.py` (L1, algorithm rules): admission through
  `SpeculativeAlgorithm.is_standalone/is_eagle3/is_dflash` (a documented
  `is_dspark` slot), the window rule per algorithm (chain: `N == steps + 1`,
  `topk == 1`; block: `N` is the block size, no step count), whether an eager
  prefill leaves draft hidden state (`eager_prefill_has_hidden`; STANDALONE
  does not), the draft graph kind a stock `init_cuda_graphs()` captures, and
  the graph policy (`speculative_prefill_cuda_graph` opt-in, default off).
  `engine_factory.configure_speculative_overrides` is a composition of these;
  no `if algorithm == ...` chains exist outside this module.
- `scheduling/speculative_bridge.py` (L0, Omni scheduler bridge): the pieces
  SGLang's own scheduler would do around a V2 worker — synchronous dispatch
  (`run_speculative_batch` → `draft_worker.forward_batch_generation`), result
  handoff (`next_draft_input`, `new_seq_lens`), terminal trimming
  (`finished_output_ids`, honouring `req.finished_len`), the STANDALONE
  hidden-state clear, `SpecInput.filter_batch` keyword compatibility, the
  future-map creation and the lookahead refusal. `cumulate_accepted_tokens`
  is the hook point for a stochastic sampling contract (replaying accepted
  tokens into the penalizer orchestrator); it is a no-op for greedy lanes.
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
  `draft_view_of_request` (identity), `position_reserve` (0), `link_models`
  (assigns `spec_role`, alias `encoder_share_role`).

## Families on this branch

| Family | Algorithm | Support object | Registry |
|---|---|---|---|
| Whisper (encoder-decoder, needs the SGLang enc-dec patch) | STANDALONE | `WhisperSpeculativeSupport`: `position_reserve = 2 * N`, encoder-output sharing via `EncoderStateShare`, timestamp requests refused | Omni `WhisperForConditionalGeneration` class registered for the draft loader |
| Qwen3-ASR | STANDALONE | `Qwen3ASRSpeculativeSupport`: draft re-encodes from retained mel (`draft_view_mm_inputs`, salted mm hash) | Omni `Qwen3ASRForConditionalGeneration` class registered |
| Qwen3-Omni thinker | EAGLE3 / DFLASH | default support; the thinker implements SGLang's `get_embed_and_head` / `set_eagle3_layers_to_capture` / `set_dflash_layers_to_capture`; visual inputs refused | nothing registered (stock drafts) |

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

## Where the next pieces plug in

- **DSpark**: admit with `algorithm.is_dspark()` in
  `speculative_support.is_omni_supported_speculative_algorithm`, add the
  `BLOCK` window rule (with a checkpoint block-size resolver) and the
  `VERIFY` draft graph kind; its worker exposes no `speculative_algorithm`
  attribute, so `init_speculative_draft_cuda_graphs` keeps reading
  `server_args.speculative_algorithm`. Omni-owned draft classes go through
  `register_speculative_target(..., draft_archs=(...))`.
- **Stochastic sampling contract** (TTS): `speculative_bridge.cumulate_accepted_tokens`
  is the per-step replay point; exact per-verify-row penalties belong to
  SGLang upstream, draft-side penalty mirroring is an Omni experimental
  feature gated behind that hook.
