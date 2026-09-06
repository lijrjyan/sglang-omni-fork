# SPDX-License-Identifier: Apache-2.0

"""Fun-CosyVoice3 LLM backbone for sglang-omni."""

from __future__ import annotations

import copy
import os
from typing import Any, Iterable, Optional, Tuple

import torch
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.models.dspark import DSparkDraftModel, VanillaMarkov
from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from torch import nn

from sglang_omni.utils.spec_debug import SPEC_DEBUG, spec_trace

from . import divprobe as _divprobe

VOCAB_SIZE = 6561
TOTAL_VOCAB_SIZE = VOCAB_SIZE + 200

SOS_ID = VOCAB_SIZE + 0
EOS_ID = VOCAB_SIZE + 1
TASK_ID = VOCAB_SIZE + 2
FILL_ID = VOCAB_SIZE + 3

CONTROL_TOKEN_IDS: tuple[int, ...] = tuple(range(VOCAB_SIZE, TOTAL_VOCAB_SIZE))

# Draft architecture name published by the AngelSlim EAGLE3 checkpoint.
EAGLE3_DRAFT_ARCHITECTURE = "CosyVoice3Eagle3LlamaForCausalLM"

# Omni-owned DSpark draft (yuekai/cosyvoice3_llm_dspark remapped to the
# speech id space); registered under this architecture name.
DSPARK_DRAFT_ARCHITECTURE = "CosyVoice3DSparkDraftModel"
# The DSpark checkpoint was trained on the HF-format LLM whose unified vocab
# places the 6761 speech ids at this offset; its mask token is the row right
# after the speech range.
DSPARK_UNIFIED_SPEECH_OFFSET = 151924
# Draft-owned embedding rows: the speech vocab plus one trained mask row.
DSPARK_MASK_TOKEN_ID = TOTAL_VOCAB_SIZE
DSPARK_EMBED_ROWS = TOTAL_VOCAB_SIZE + 1


_VERIFYPROBE_DIR = os.environ.get("SGLANG_OMNI_VERIFYPROBE_DIR")
# Round-7 A3 diagnostic: per-verify-step draft block vs target argmax (env-gated).
_A3_TRACE = os.environ.get("SGLANG_OMNI_COSYVOICE3_A3_TRACE") == "1"
# Round-8 H1 sensitivity probe: zeros | noise | final (replace the aux features).
_AUX_PROBE = os.environ.get("SGLANG_OMNI_COSYVOICE3_AUX_PROBE", "")
# Round-8 H2: log draft id -> HF unified id -> served speech id for the first proposals.
_REMAP_TRACE = os.environ.get("SGLANG_OMNI_COSYVOICE3_REMAP_TRACE") == "1"


def _a3_trace(tag, input_ids, positions, forward_batch, logits):
    n = min(int(logits.shape[0]), 16)
    spec_trace(
        f"a3 verify {tag}: ids={input_ids[:n].tolist()} argmax={logits[:n].argmax(dim=-1).tolist()} "
        f"positions={positions[:n].tolist()} seq_len={int(forward_batch.seq_lens[0])}"
    )


def _verifyprobe_dump(pre, logits_output, input_ids, positions) -> None:
    """Diagnostic only: per verify row, top-5 logits before/after the block
    repetition penalty (env ``SGLANG_OMNI_VERIFYPROBE_DIR``)."""
    import json as _json

    post = logits_output.next_token_logits.float().topk(5, dim=-1)
    os.makedirs(_VERIFYPROBE_DIR, exist_ok=True)
    with open(os.path.join(_VERIFYPROBE_DIR, "verifyprobe.jsonl"), "a") as f:
        for r in range(input_ids.shape[0]):
            f.write(
                _json.dumps(
                    {
                        "pos": int(positions[r]),
                        "row": r,
                        "input_id": int(input_ids[r]),
                        "pre": [
                            [int(i), round(float(v), 4)]
                            for v, i in zip(
                                pre.values[r].tolist(), pre.indices[r].tolist()
                            )
                        ],
                        "post": [
                            [int(i), round(float(v), 4)]
                            for v, i in zip(
                                post.values[r].tolist(), post.indices[r].tolist()
                            )
                        ],
                    }
                )
                + "\n"
            )


class FunCosyVoice3SGLangModel(Qwen2ForCausalLM):

    def __init__(
        self,
        config: Any,
        quant_config: Any = None,
        prefix: str = "",
    ):
        super().__init__(config, quant_config, prefix)
        self.text_embed_tokens = self.model.embed_tokens
        self.speech_embedding = nn.Embedding(TOTAL_VOCAB_SIZE, config.hidden_size)
        self.llm_decoder = nn.Linear(config.hidden_size, TOTAL_VOCAB_SIZE, bias=False)
        # note (Junnan Li): SGLang's LogitsProcessor owns per-forward-mode row
        # selection (prefill last rows, decode, TARGET_VERIFY all rows) and the
        # EAGLE3 aux hidden-state packing, so the speech head reuses it with a
        # speech-vocab view of the config instead of hand-slicing rows.
        speech_config = copy.deepcopy(config)
        speech_config.vocab_size = TOTAL_VOCAB_SIZE
        self.logits_processor = LogitsProcessor(speech_config)
        self._cached_params_dict = dict(self.named_parameters())

    @property
    def vocab_size(self) -> int:
        return TOTAL_VOCAB_SIZE

    @vocab_size.setter
    def vocab_size(self, value: int) -> None:
        pass

    def get_input_embeddings(self):
        return self.speech_embedding

    def get_embed_and_head(self):
        """Speech-token embedding and head shared with native speculative drafts."""
        return self.speech_embedding.weight, self.llm_decoder.weight

    def set_speculative_repetition_penalties(self, values: Optional[torch.Tensor]):
        """Scheduler hook: per-request repetition penalty (bs, 1) for this step.

        The forward-time sampling info is a copy without the penalizer
        orchestrator, so the scheduler publishes the values explicitly for the
        verify-block penalty and the draft mirror.
        """
        _DRAFT_SAMPLING_INFO["rep_values"] = values

    def set_dspark_layers_to_capture(self, layer_ids: Optional[list[int]] = None):
        """DSpark aux capture: HF-style "after layer k" ids, like EAGLE3."""
        if layer_ids is None:
            raise ValueError("DSpark requires explicit target layer ids to capture")
        layer_ids = list(layer_ids)
        # Bring-up probe: the checkpoint's aux ids are read as "output of
        # layer k" (SGLang's DFlash/EAGLE3 convention, captured before layer
        # k+1); set 0 to capture the input of layer k instead.
        if os.environ.get("SGLANG_OMNI_COSYVOICE3_DSPARK_CAPTURE_AFTER", "1") == "0":
            layer_ids = [max(i - 1, 0) for i in layer_ids]
        self.set_eagle3_layers_to_capture(layer_ids)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):
        """Stock capture points (EAGLE3 default / explicit ids + 1), plus a
        round-8 bring-up probe: ``SGLANG_OMNI_COSYVOICE3_CAPTURE_SHIFT=<int>``
        shifts every capture point (capture before layer i + shift)."""
        super().set_eagle3_layers_to_capture(layer_ids)
        shift = int(os.environ.get("SGLANG_OMNI_COSYVOICE3_CAPTURE_SHIFT", "0") or 0)
        if shift and getattr(self, "capture_aux_hidden_states", False):
            num_layers = int(self.config.num_hidden_layers)
            self.model.layers_to_capture = [
                min(max(i + shift, 0), num_layers - 1)
                for i in self.model.layers_to_capture
            ]
            spec_trace(
                f"capture shift {shift:+d}: layers_to_capture={self.model.layers_to_capture}"
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[Any] = None,
    ) -> Any:
        fwd_mode = forward_batch.forward_mode
        # The draft-side sampling mirror reads the live batch sampling state
        # (penalties are refreshed in place before every speculative step).
        _DRAFT_SAMPLING_INFO["info"] = getattr(forward_batch, "sampling_info", None)
        if input_embeds is None:
            # Decode, TARGET_VERIFY and any other id-driven forward: every
            # input id is a speech token.
            input_embeds = self.speech_embedding(input_ids)
        elif fwd_mode.is_extend():
            # Prefill runs on projected prompt embeddings. Publish them so a
            # native speculative draft can prefill on the same rows (the draft
            # cannot rebuild text/speaker embeddings from speech ids).
            forward_batch.mm_input_embeds = input_embeds

        probe_kind = (
            _divprobe.match(forward_batch, positions) if _divprobe.ENABLED else None
        )
        probe_hooks = _divprobe.Hooks(self.model) if probe_kind else None
        try:
            hidden_states = self.model(
                input_ids,
                positions,
                forward_batch,
                input_embeds,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        finally:
            if probe_hooks is not None:
                probe_hooks.remove()
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
            if _AUX_PROBE and aux_hidden_states is not None:
                # Round-8 bring-up probe: destroy the target features handed to
                # the draft to test whether the draft uses them at all.
                if _AUX_PROBE == "zeros":
                    aux_hidden_states = [torch.zeros_like(a) for a in aux_hidden_states]
                elif _AUX_PROBE == "noise":
                    aux_hidden_states = [
                        torch.randn_like(a) * a.float().std().to(a.dtype)
                        for a in aux_hidden_states
                    ]
                elif _AUX_PROBE == "final":
                    aux_hidden_states = [hidden_states.clone() for _ in aux_hidden_states]
        if SPEC_DEBUG and not getattr(self, "_traced_capture", False):
            self._traced_capture = True
            spec_trace(
                f"target forward mode={fwd_mode} capture_aux={self.capture_aux_hidden_states} "
                f"layers_to_capture={getattr(self.model, 'layers_to_capture', None)} "
                f"hidden={tuple(hidden_states.shape)} "
                f"aux={[tuple(a.shape) for a in aux_hidden_states] if aux_hidden_states is not None else None} "
                f"capture_mode={getattr(forward_batch, 'capture_hidden_mode', None)}"
            )

        logits_output = self.logits_processor(
            input_ids,
            hidden_states,
            self.llm_decoder,
            forward_batch,
            aux_hidden_states,
        )
        if probe_kind:
            try:
                _divprobe.save(
                    probe_kind,
                    self,
                    probe_hooks,
                    input_ids,
                    positions,
                    forward_batch,
                    input_embeds,
                    hidden_states,
                    logits_output,
                )
            except Exception as exc:  # diagnostic must not kill the server
                import traceback

                print(f"[divprobe] capture failed: {exc!r}", flush=True)
                traceback.print_exc()
        is_verify = getattr(fwd_mode, "is_target_verify", lambda: False)()
        if is_verify and SPEC_DEBUG and getattr(self, "_verify_trace", 0) < 4:
            self._trace_verify_rows(
                input_ids, positions, hidden_states, forward_batch, logits_output
            )
        _vp_pre = (
            logits_output.next_token_logits.float().topk(5, dim=-1)
            if _VERIFYPROBE_DIR and is_verify
            else None
        )
        if is_verify and _is_dspark_forward(forward_batch):
            if False:
                self._verify_trace = getattr(self, "_verify_trace", 0) + 1
                logits = logits_output.next_token_logits
                spec_trace(
                    f"verify rows: ids={input_ids[:9].tolist()} "
                    f"argmax_pre_penalty={logits[:9].argmax(dim=-1).tolist()} "
                    f"logits_shape={tuple(logits.shape)} hidden_rows={hidden_states.shape[0]} "
                    f"positions={positions[:9].tolist()} seq_lens={forward_batch.seq_lens[:2].tolist()} "
                    f"out_cache_loc={forward_batch.out_cache_loc[:9].tolist()} "
                    f"capture={forward_batch.capture_hidden_mode} "
                    f"hs_store={None if logits_output.hidden_states is None else tuple(logits_output.hidden_states.shape)}"
                )
            apply_block_repetition_penalty(
                logits_output.next_token_logits, input_ids, forward_batch.sampling_info
            )
            if _A3_TRACE:
                _a3_trace("dspark-post-penalty", input_ids, positions, forward_batch, logits_output.next_token_logits)
            if SPEC_DEBUG and getattr(self, "_verify_trace", 0) <= 4:
                lg = logits_output.next_token_logits[0].float()
                top = lg.topk(4)
                spec_trace(
                    f"verify row0 post-penalty: argmax={int(lg.argmax())} top_ids={top.indices.tolist()} "
                    f"top_vals={[round(v, 3) for v in top.values.tolist()]} "
                    f"logit3={lg[3].item():.3f} logit0={lg[0].item():.3f} contiguous={logits_output.next_token_logits.is_contiguous()}"
                )
        elif SPEC_DEBUG and is_verify:
            spec_trace(
                f"verify penalty skipped: spec_algorithm={getattr(forward_batch, 'spec_algorithm', None)}"
            )
            if _A3_TRACE:
                _a3_trace("eagle3-pre-penalty", input_ids, positions, forward_batch, logits_output.next_token_logits)
        if _vp_pre is not None:  # diagnostic only, env-gated
            _verifyprobe_dump(_vp_pre, logits_output, input_ids, positions)
        return logits_output

    def _trace_verify_rows(
        self, input_ids, positions, hidden_states, forward_batch, logits_output
    ):
        self._verify_trace = getattr(self, "_verify_trace", 0) + 1
        logits = logits_output.next_token_logits
        top = logits[0].float().topk(4)
        spec_trace(
            f"verify row0 pre-penalty top_ids={top.indices.tolist()} top_vals={[round(v, 3) for v in top.values.tolist()]}"
        )
        spec_trace(
            f"verify rows: spec={getattr(forward_batch, 'spec_algorithm', None)} ids={input_ids[:9].tolist()} "
            f"argmax_pre_penalty={logits[:9].argmax(dim=-1).tolist()} "
            f"logits_shape={tuple(logits.shape)} hidden_rows={hidden_states.shape[0]} "
            f"positions={positions[:9].tolist()} seq_lens={forward_batch.seq_lens[:2].tolist()} "
            f"out_cache_loc={forward_batch.out_cache_loc[:9].tolist()} "
            f"input_embeds={None if forward_batch.input_embeds is None else tuple(forward_batch.input_embeds.shape)} "
            f"capture={forward_batch.capture_hidden_mode} "
            f"hs_store={None if logits_output.hidden_states is None else tuple(logits_output.hidden_states.shape)}"
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        backbone_weights = []

        for name, loaded_weight in weights:
            # note: Qwen2 backbone inside CosyVoice3LM uses llm.model.model.* prefix
            backbone_prefix = "llm.model.model."
            if name.startswith(backbone_prefix):
                backbone_weights.append(
                    ("model." + name[len(backbone_prefix) :], loaded_weight)
                )
                continue

            # note: skip Qwen2 text lm_head — CosyVoice3 uses llm_decoder instead
            if name == "llm.model.lm_head.weight":
                continue

            if name in (
                "speech_embedding.weight",
                "llm_decoder.weight",
                "llm_decoder.bias",
            ):
                param = self._cached_params_dict.get(name)
                if param is not None:
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    loader(param, loaded_weight)
                continue

            # note (PoTaTo): pass through HF safetensors keys (model.*) to Qwen2 backbone
            backbone_weights.append((name, loaded_weight))

        if backbone_weights:
            super().load_weights(backbone_weights)


def _is_dspark_forward(forward_batch: Any) -> bool:
    algorithm = getattr(forward_batch, "spec_algorithm", None)
    is_dspark = getattr(algorithm, "is_dspark", None)
    return bool(is_dspark is not None and is_dspark())


def repetition_penalty_values(sampling_info: Any) -> Optional[torch.Tensor]:
    """Per-request repetition penalty (bs, 1) from the live penalizer, or None."""
    provided = getattr(sampling_info, "_omni_repetition_penalties", None)
    if provided is None:
        provided = _DRAFT_SAMPLING_INFO.get("rep_values")
    if provided is not None:
        return provided
    orchestrator = getattr(sampling_info, "penalizer_orchestrator", None)
    if orchestrator is None or not orchestrator.is_required:
        # The forward-time sampling info is a copy without the orchestrator.
        return None
    for penalizer in orchestrator.penalizers.values():
        values = getattr(penalizer, "repetition_penalties", None)
        if values is not None and getattr(penalizer, "_is_prepared", True):
            return values
    reqs = list(orchestrator.reqs())
    penalties = [float(req.sampling_params.repetition_penalty) for req in reqs]
    if not penalties or all(value == 1.0 for value in penalties):
        return None
    if SPEC_DEBUG and not getattr(orchestrator, "_omni_pen_traced", False):
        orchestrator._omni_pen_traced = True
        spec_trace(
            "repetition penalty from sampling_params: "
            f"penalizers={[(type(p).__name__, hasattr(p, 'repetition_penalties'), getattr(p, '_is_prepared', None)) for p in orchestrator.penalizers.values()]}"
        )
    return torch.tensor(
        penalties, dtype=torch.float32, device=orchestrator.device
    ).view(-1, 1)


@torch.no_grad()
def apply_block_repetition_penalty(
    logits: torch.Tensor, block_ids: torch.Tensor, sampling_info: Any
) -> None:
    """Omni's per-token repetition penalty on a DSpark verify block, in place.

    SGLang's DFlash/DSpark verify only broadcasts additive penalties over the
    block; the multiplicative repetition penalty is dropped, and CosyVoice3
    degenerates without it. Row ``j`` of each request's ``T`` verify rows
    penalises the tokens already generated (``acc_scaling_penalties``, refreshed
    before the step) plus the block tokens ``0..j`` it has consumed, which is
    exactly what token-by-token decoding would have applied.
    """
    if logits is None or sampling_info is None:
        return
    scaling = getattr(sampling_info, "acc_scaling_penalties", None)
    if scaling is None:
        return
    bs = scaling.shape[0]
    rows = logits.shape[0]
    if bs == 0 or rows % bs != 0:
        return
    block = rows // bs
    width = min(scaling.shape[1], logits.shape[1])
    penalties = scaling[:, None, :width].expand(bs, block, width).clone()
    values = repetition_penalty_values(sampling_info)
    if SPEC_DEBUG and not getattr(sampling_info, "_omni_penalty_traced", False):
        sampling_info._omni_penalty_traced = True
        spec_trace(
            f"verify block penalty: bs={bs} block={block} width={width} "
            f"scaling_max={float(scaling.max()):.3f} values={None if values is None else values.flatten()[:4].tolist()} "
            f"ids={block_ids.flatten()[:9].tolist()}"
        )
    if values is not None:
        ids = block_ids.view(bs, block).clamp(max=width - 1)
        for j in range(block):
            penalties[:, j].scatter_(
                1, ids[:, : j + 1], values.to(penalties.dtype).expand(bs, j + 1)
            )
    view = logits.view(bs, block, -1)[..., :width]
    pen = penalties.to(view.dtype)
    view.copy_(torch.where(view < 0, view * pen, view / pen))


def _shift_prompt_embeds_default() -> bool:
    # EAGLE convention: the draft input at prompt position i is the embedding
    # of token i+1 (the target's input rows rolled left by one, last row =
    # the sampled first speech token). SGLang's multimodal EAGLE3 path feeds
    # unshifted rows instead; expose the alternative for acceptance probes.
    return os.environ.get("SGLANG_OMNI_COSYVOICE3_EAGLE3_UNSHIFTED_PROMPT", "0") != "1"


_DRAFT_SAMPLING_INFO: dict = {"info": None, "rep_values": None}


def _mirror_draft_logits(logits: torch.Tensor, sampling_info: Any) -> torch.Tensor:
    """Apply the target's (relaxed) repetition penalty to the draft logits."""
    if (
        sampling_info is None
        or os.environ.get("SGLANG_OMNI_COSYVOICE3_EAGLE3_DRAFT_REP") != "1"
    ):
        return logits
    penalties = getattr(sampling_info, "acc_scaling_penalties", None)
    if penalties is None or penalties.shape[0] != logits.shape[0]:
        return logits
    width = min(penalties.shape[1], logits.shape[1])
    pen = penalties[:, :width].to(logits.dtype)
    head = logits[:, :width]
    head = torch.where(head > 0, head / pen, head * pen)
    if width == logits.shape[1]:
        return head
    return torch.cat([head, logits[:, width:]], dim=1)


def _mirror_draft_probs(probs: torch.Tensor, sampling_info: Any) -> torch.Tensor:
    """Renormalise the draft proposal with the target's top-k / top-p."""
    if (
        sampling_info is None
        or os.environ.get("SGLANG_OMNI_COSYVOICE3_EAGLE3_DRAFT_TOPKP") != "1"
    ):
        return probs
    top_ks = getattr(sampling_info, "top_ks", None)
    if top_ks is None or top_ks.shape[0] != probs.shape[0]:
        return probs
    from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

    if sampling_info.need_top_k_sampling:
        probs = top_k_renorm_prob(probs, top_ks)
    if sampling_info.need_top_p_sampling:
        probs = top_p_renorm_prob(probs, sampling_info.top_ps)
    return probs


def _install_draft_sampling_mirror() -> None:
    """Env-gated experiment: make the draft proposal q track the target's
    top-k / top-p / repetition-penalty filtering (any q is unbiased; a closer
    q raises acceptance). Patches the names bound in eagle_worker_v2."""
    if os.environ.get("SGLANG_OMNI_COSYVOICE3_EAGLE3_DRAFT_TOPKP") != "1" and (
        os.environ.get("SGLANG_OMNI_COSYVOICE3_EAGLE3_DRAFT_REP") != "1"
    ):
        return
    import sglang.srt.speculative.eagle_worker_v2 as eagle_worker_v2
    from sglang.srt.speculative.spec_utils import fast_sample

    if getattr(eagle_worker_v2, "_omni_cosyvoice3_mirror", False):
        return

    def sample_draft_proposal(next_token_logits, temperatures):
        info = _DRAFT_SAMPLING_INFO["info"]
        logits = _mirror_draft_logits(next_token_logits, info)
        probs = torch.softmax(logits / temperatures, dim=-1)
        probs = _mirror_draft_probs(probs, info)
        topk_p, topk_index = fast_sample(probs, num_samples=1)
        return probs, topk_p, topk_index

    def renorm_draft_probs(next_token_logits, sampling_info, use_rejection_sampling):
        if not use_rejection_sampling or not next_token_logits.size(0):
            return torch.softmax(next_token_logits, dim=-1)
        logits = _mirror_draft_logits(next_token_logits, sampling_info)
        probs = torch.softmax(logits / sampling_info.temperatures, dim=-1)
        return _mirror_draft_probs(probs, sampling_info)

    eagle_worker_v2.sample_draft_proposal = sample_draft_proposal
    eagle_worker_v2.renorm_draft_probs = renorm_draft_probs
    eagle_worker_v2._omni_cosyvoice3_mirror = True
    spec_trace(
        "draft sampling mirror installed: "
        f"topkp={os.environ.get('SGLANG_OMNI_COSYVOICE3_EAGLE3_DRAFT_TOPKP')} "
        f"rep={os.environ.get('SGLANG_OMNI_COSYVOICE3_EAGLE3_DRAFT_REP')}"
    )


class CosyVoice3Eagle3LlamaForCausalLM(LlamaForCausalLMEagle3):
    """AngelSlim EAGLE3 draft for the CosyVoice3 speech-token LLM.

    The checkpoint publishes ``tie_word_embeddings=True`` although it ships an
    independently trained ``lm_head`` (not equal to the target head); the
    stock loader would alias the head to the target embedding. It also
    publishes the target's ``num_hidden_layers`` (24) for a single-layer draft.
    Both are normalised here before the stock draft is constructed.
    """

    def __init__(
        self,
        config: Any,
        quant_config: Any = None,
        prefix: str = "",
    ) -> None:
        normalize_cosyvoice3_eagle3_config(config)
        super().__init__(config, quant_config, prefix)
        self.shift_prompt_embeds = _shift_prompt_embeds_default()
        _install_draft_sampling_mirror()
        self.model.forward = self._model_forward  # type: ignore[method-assign]
        self._stock_model_forward = type(self.model).forward

    def _model_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[Any] = None,
    ) -> torch.Tensor:
        mm_input_embeds = forward_batch.mm_input_embeds
        _DRAFT_SAMPLING_INFO["info"] = forward_batch.sampling_info
        if (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and (input_embeds is not None or mm_input_embeds is not None)
        ):
            # note (Junnan Li): the draft prefill reuses the target's
            # ScheduleBatch, whose ``input_embeds`` (Omni's projected prompt
            # rows) the runner forwards verbatim. Those rows are the TARGET's
            # inputs; the EAGLE draft needs them rolled by one with the sampled
            # first token in the last row, so always rebuild them here.
            source = input_embeds if input_embeds is not None else mm_input_embeds
            input_embeds = self.prompt_input_embeds(source, input_ids, forward_batch)
        trace_key = "_traced_draft_" + str(forward_batch.forward_mode)
        if (
            SPEC_DEBUG
            and getattr(self, "_trace_count", 0) < 6
            and not getattr(self, trace_key, False)
        ):
            setattr(self, trace_key, True)
            self._trace_count = getattr(self, "_trace_count", 0) + 1
            spec_info = forward_batch.spec_info
            hidden = getattr(spec_info, "hidden_states", None)
            import hashlib

            def _h(t):
                return hashlib.sha256(
                    t.detach()
                    .to("cpu", torch.bfloat16)
                    .contiguous()
                    .view(torch.int16)
                    .numpy()
                    .tobytes()
                ).hexdigest()[:16]

            spec_trace(
                f"draft forward mode={forward_batch.forward_mode} "
                f"mm_rows={None if mm_input_embeds is None else tuple(mm_input_embeds.shape)} "
                f"embeds={None if input_embeds is None else tuple(input_embeds.shape)} "
                f"spec_hidden={None if hidden is None else tuple(hidden.shape)} "
                f"shift={self.shift_prompt_embeds} ids={input_ids[:8].tolist()} "
                f"lm_head={tuple(self.lm_head.weight.shape)}:{_h(self.lm_head.weight[:TOTAL_VOCAB_SIZE])} "
                f"embed={tuple(self.model.embed_tokens.weight.shape)}:{_h(self.model.embed_tokens.weight)} "
                f"fc={tuple(self.model.fc.weight.shape)}:{_h(self.model.fc.weight)} "
                f"layers={len(self.model.layers)} hot={None if self.hot_token_id is None else int((self.hot_token_id != torch.arange(self.hot_token_id.shape[0], device=self.hot_token_id.device)).sum())}"
            )
        if (
            os.environ.get("SGLANG_OMNI_COSYVOICE3_EAGLE3_ATTN_PROBE") == "1"
            and getattr(self, "_attn_probe_count", 0) < 6
        ):
            # round-2 R1 bring-up probe: draft attention metadata + KV ablation
            self._attn_probe_count = getattr(self, "_attn_probe_count", 0) + 1
            out = self._attn_probe(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        else:
            out = self._stock_model_forward(
                self.model,
                input_ids,
                positions,
                forward_batch,
                input_embeds,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        if (
            SPEC_DEBUG
            and mm_input_embeds is not None
            and forward_batch.forward_mode.is_extend()
            and not getattr(self, "_traced_prefill_acc", False)
        ):
            self._traced_prefill_acc = True
            self._trace_prefill_accuracy(out, mm_input_embeds, input_ids, forward_batch)
        if SPEC_DEBUG and getattr(self, "_kv_trace_count", 0) < 4:
            self._kv_trace_count = getattr(self, "_kv_trace_count", 0) + 1
            self._trace_kv(forward_batch)
        return out

    @torch.no_grad()
    def _attn_probe(self, input_ids, positions, forward_batch, input_embeds, pp):
        """R1: print the draft attention metadata for this forward and, for
        decode steps, compare the layer-0 attention output with the history
        KV rows intact vs zeroed (rows restored afterwards)."""
        from sglang.srt.model_executor.forward_context import get_attn_backend

        def _t(x, n=8):
            if x is None:
                return None
            if torch.is_tensor(x):
                return x.flatten()[:n].tolist()
            return x

        def run():
            return self._stock_model_forward(
                self.model,
                input_ids,
                positions,
                forward_batch,
                input_embeds,
                pp_proxy_tensors=pp,
            )

        try:
            backend = get_attn_backend()
            md = getattr(backend, "forward_metadata", None)
            attn = self.model.layers[0].self_attn.attn
            pool = backend.token_to_kv_pool
            rtt = backend.req_to_token_pool.req_to_token
            runner_rtt = backend.req_to_token  # captured at backend init
            runner_pool = getattr(forward_batch, "token_to_kv_pool", None) or pool
            req = forward_batch.req_pool_indices[0].long()
            n_hist = int(forward_batch.seq_lens[0].item())
            spec_trace(
                f"attn-probe mode={forward_batch.forward_mode} bs={forward_batch.batch_size} "
                f"backend={type(backend).__name__} step_id={getattr(backend, 'speculative_step_id', None)} "
                f"topk={getattr(backend, 'topk', None)} seq_lens={_t(forward_batch.seq_lens)} "
                f"seq_lens_cpu={_t(forward_batch.seq_lens_cpu)} req_pool_indices={_t(forward_batch.req_pool_indices)} "
                f"positions={_t(positions)} out_cache_loc={_t(forward_batch.out_cache_loc)} "
                f"extend_seq_lens={_t(forward_batch.extend_seq_lens)} extend_prefix_lens={_t(forward_batch.extend_prefix_lens)} "
                f"input_ids={_t(input_ids)} spec_info={type(forward_batch.spec_info).__name__} "
                f"md.cache_seqlens={_t(getattr(md, 'cache_seqlens_int32', None))} "
                f"md.max_seq_len_k={getattr(md, 'max_seq_len_k', None)} md.max_seq_len_q={getattr(md, 'max_seq_len_q', None)} "
                f"md.cu_seqlens_q={_t(getattr(md, 'cu_seqlens_q', None))} md.cu_seqlens_k={_t(getattr(md, 'cu_seqlens_k', None))} "
                f"md.page_table_shape={None if getattr(md, 'page_table', None) is None else tuple(md.page_table.shape)} "
                f"md.page_table_row0={_t(getattr(md, 'page_table', None))} "
                f"rtt_row={_t(rtt[req])} rtt_row_tail={rtt[req, max(n_hist - 3, 0): n_hist + 3].tolist()} "
                f"rtt_live_same_as_init={rtt.data_ptr() == runner_rtt.data_ptr()} "
                f"pool_same_as_fb={pool is runner_pool} pool={type(pool).__name__} "
                f"k_shape={tuple(pool.get_key_buffer(attn.layer_id).shape)} layer_id={attn.layer_id} "
                f"attn_heads={attn.tp_q_head_num}/{attn.tp_k_head_num} head_dim={attn.head_dim} "
                f"max_context_len={getattr(backend, 'max_context_len', None)}"
            )
            if not forward_batch.forward_mode.is_decode():
                if (
                    input_embeds is None
                    or forward_batch.forward_mode.is_draft_extend_v2()
                ):
                    return run()
                # extend probe: do the prompt rows influence the written KV?
                k = pool.get_key_buffer(attn.layer_id)
                locs = forward_batch.out_cache_loc.long()
                real_embeds = input_embeds
                input_embeds = torch.randn_like(real_embeds)
                out_b = run()
                k_b = k[locs].float().clone()
                input_embeds = real_embeds
                out_a = run()
                k_a = k[locs].float().clone()
                spec_trace(
                    f"attn-probe extend-ablation mode={forward_batch.forward_mode} rows={int(locs.numel())} "
                    f"probe_env={os.environ.get('SGLANG_OMNI_COSYVOICE3_EAGLE3_PROMPT_PROBE')} "
                    f"embeds_row0_norm={real_embeds[0].float().norm().item():.3f} "
                    f"embeds_rowlast_norm={real_embeds[-1].float().norm().item():.3f} "
                    f"k_real_norm_mean={k_a.flatten(1).norm(dim=-1).mean().item():.3f} "
                    f"k_noise_norm_mean={k_b.flatten(1).norm(dim=-1).mean().item():.3f} "
                    f"k_maxabsdiff={(k_a - k_b).abs().max().item():.4f} "
                    f"hidden_maxabsdiff={(out_a[0].float() - out_b[0].float()).abs().max().item():.4f} "
                    f"hidden_last_maxabsdiff={(out_a[0][-1].float() - out_b[0][-1].float()).abs().max().item():.4f}"
                )
                return out_a
            k = pool.get_key_buffer(attn.layer_id)
            v = pool.get_value_buffer(attn.layer_id)
            locs = rtt[req, :n_hist].long()
            captured = {}
            hook = attn.register_forward_hook(
                lambda m, i, o: captured.__setitem__("o", o.detach().float().clone())
            )
            try:
                out_a = run()
                attn_a = captured["o"]
                hist_norm = k[locs].float().flatten(1).norm(dim=-1)
                k_save, v_save = k[locs].clone(), v[locs].clone()
                k[locs] = 0
                v[locs] = 0
                out_b = run()
                attn_b = captured["o"]
                k[locs] = k_save
                v[locs] = v_save
            finally:
                hook.remove()
            ha, hb = out_a[0].float(), out_b[0].float()
            spec_trace(
                f"attn-probe ablation mode={forward_batch.forward_mode} step_id={getattr(backend, 'speculative_step_id', None)} "
                f"n_hist={n_hist} locs_head={locs[:4].tolist()} hist_k_norm_mean={hist_norm.mean().item():.3f} "
                f"hist_k_zero_rows={int((hist_norm == 0).sum())} "
                f"attn_out_shape={tuple(attn_a.shape)} attn_out_norm={attn_a.norm().item():.4f} "
                f"attn_maxabsdiff={(attn_a - attn_b).abs().max().item():.6f} "
                f"hidden_maxabsdiff={(ha - hb).abs().max().item():.6f} "
                f"identical={bool(torch.equal(attn_a, attn_b))}"
            )
            return out_a
        except Exception as exc:  # pragma: no cover - probe only
            spec_trace(f"attn-probe failed: {exc!r}")
            import traceback

            spec_trace(traceback.format_exc()[-1500:])
            return run()

    @torch.no_grad()
    def _trace_kv(self, forward_batch):
        try:
            attn = self.model.layers[0].self_attn.attn
            runner = getattr(forward_batch.attn_backend, "model_runner", None)
            pool = runner.token_to_kv_pool
            k = pool.get_key_buffer(attn.layer_id)
            locs = forward_batch.out_cache_loc
            written = k[locs.long()].float().norm(dim=-1)
            seq = forward_batch.seq_lens
            req_to_token = runner.req_to_token_pool.req_to_token
            row = req_to_token[forward_batch.req_pool_indices[0].long()]
            n = int(seq[0].item()) if seq is not None and seq.numel() else 0
            hist = k[row[: max(n, 1)].long()].float().norm(dim=-1)
            spec_trace(
                f"draft kv mode={forward_batch.forward_mode} layer_id={attn.layer_id} "
                f"pool={type(pool).__name__} k_shape={tuple(k.shape)} "
                f"written_rows={int(locs.numel())} written_norm_mean={written.mean().item():.3f} "
                f"written_zero_rows={int((written == 0).sum())} seq_len0={n} "
                f"history_norm_mean={hist.mean().item():.3f} history_zero_rows={int((hist == 0).sum())} "
                f"locs_head={locs[:3].tolist()} hist_locs_head={row[:3].tolist()} "
                f"attn_backend={type(forward_batch.attn_backend).__name__}"
            )
        except Exception as exc:  # pragma: no cover - probe only
            spec_trace(f"draft kv trace failed: {exc!r}")

    @torch.no_grad()
    def _trace_prefill_accuracy(self, out, prompt_rows, input_ids, forward_batch):
        """Bring-up probe: draft top-1 vs the true next prompt speech token."""
        hidden_to_logits = out[0] if isinstance(out, tuple) else out
        num_rows = input_ids.shape[0]
        rows = prompt_rows[:num_rows].float()
        weight = self.model.embed_tokens.weight[:TOTAL_VOCAB_SIZE].float()
        # exact-match speech rows -> ids (text/speaker rows map to -1)
        dist = torch.cdist(rows, weight)
        best = dist.min(dim=1)
        ids = torch.where(
            best.values < 1e-3, best.indices, torch.full_like(best.indices, -1)
        )
        logits = torch.nn.functional.linear(
            hidden_to_logits.float(), self.lm_head.weight[:TOTAL_VOCAB_SIZE].float()
        )
        pred = logits.argmax(dim=-1)
        next_ids = torch.cat([ids[1:], torch.full_like(ids[:1], -1)])
        valid_next = next_ids >= 0
        valid_cur = ids >= 0
        acc_next = (
            (pred[valid_next] == next_ids[valid_next]).float().mean().item()
            if valid_next.any()
            else float("nan")
        )
        acc_cur = (
            (pred[valid_cur] == ids[valid_cur]).float().mean().item()
            if valid_cur.any()
            else float("nan")
        )
        spec_trace(
            f"draft prefill acc: rows={num_rows} speech_rows={int(valid_cur.sum())} "
            f"acc_next={acc_next:.3f} acc_cur={acc_cur:.3f} "
            f"first_speech_row={int(valid_cur.nonzero()[0]) if valid_cur.any() else -1} "
            f"pred_tail={pred[-12:].tolist()} true_tail={ids[-12:].tolist()}"
        )

    def prompt_input_embeds(
        self,
        prompt_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        forward_batch: Any,
    ) -> torch.Tensor:
        """Draft prefill rows from the target's projected prompt embeddings."""
        num_rows = input_ids.shape[0]
        embeds = prompt_embeds[:num_rows].to(self.model.embed_tokens.weight.dtype)
        last_indices = (
            forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
        ).long()
        if self.shift_prompt_embeds:
            embeds = torch.roll(embeds, shifts=-1, dims=0)
        else:
            embeds = embeds.clone()
        probe = os.environ.get("SGLANG_OMNI_COSYVOICE3_EAGLE3_PROMPT_PROBE")
        if probe == "noise":
            # bring-up probe: destroy the prompt rows to test their influence
            embeds = torch.randn_like(embeds)
        elif probe == "zeros":
            embeds = torch.zeros_like(embeds)
        last_rows = self.model.embed_tokens(input_ids[last_indices])
        embeds[last_indices] = last_rows
        if SPEC_DEBUG and getattr(self, "_last_row_trace", 0) < 3:
            self._last_row_trace = getattr(self, "_last_row_trace", 0) + 1
            emb = self.model.embed_tokens
            ids = input_ids[last_indices]
            spec_trace(
                f"draft prompt last rows: ids={ids.tolist()} "
                f"row_norms={last_rows.float().norm(dim=-1).tolist()} "
                f"weight_row_norms={emb.weight[ids.clamp(max=emb.weight.shape[0] - 1)].float().norm(dim=-1).tolist()} "
                f"weight_shape={tuple(emb.weight.shape)} num_embeddings={getattr(emb, 'num_embeddings', None)} "
                f"vocab_range={getattr(emb, 'shard_indices', None)} "
                f"prompt_row_norms_tail={embeds[-3:].float().norm(dim=-1).tolist()}"
            )
        return embeds

    def load_weights(self, weights):
        super().load_weights(weights)
        if os.environ.get("SGLANG_OMNI_COSYVOICE3_EAGLE3_ZERO_FC") == "1":
            # bring-up probe: sever the target-hidden path into the draft
            with torch.no_grad():
                self.model.fc.weight.zero_()


def normalize_cosyvoice3_eagle3_config(config: Any) -> Any:
    """Fix the published AngelSlim draft config for the stock EAGLE3 loader."""
    config.tie_word_embeddings = False
    config.num_hidden_layers = 1
    if getattr(config, "draft_vocab_size", None) is None:
        config.draft_vocab_size = TOTAL_VOCAB_SIZE
    return config


_DSPARK_MIRROR_DEFAULTS = {"REP": "1", "TOPKP": "0"}


def _dspark_mirror_enabled(name: str) -> bool:
    return (
        os.environ.get(
            f"SGLANG_OMNI_COSYVOICE3_DSPARK_DRAFT_{name}", _DSPARK_MIRROR_DEFAULTS[name]
        )
        == "1"
    )


class CosyVoice3MarkovHead(VanillaMarkov):
    """DSpark Markov head over a reduced draft vocab, evaluated in the speech space.

    The checkpoint's ``lm_head`` / ``markov_w2`` speak an 8192-id draft vocab
    mapped to target ids by ``d2t``; ``markov_w1`` is indexed by target ids.
    Every step's logits are scattered into the 6761 speech space (unmapped
    draft ids and text ids drop out as ``-inf``) so the sampled tokens, the
    proposal distribution ``q`` handed to rejection sampling and the verify
    candidates all live in the target id space.
    """

    def __init__(self, *, vocab_size: int, draft_vocab_size: int, markov_rank: int):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.draft_vocab_size = int(draft_vocab_size)
        self.markov_w2 = nn.Linear(self.markov_rank, self.draft_vocab_size, bias=False)
        self.register_buffer(
            "d2t_omni",
            torch.full((self.draft_vocab_size,), -1, dtype=torch.long),
            persistent=False,
        )
        self._valid_draft_ids: Optional[torch.Tensor] = None
        self._valid_target_ids: Optional[torch.Tensor] = None

    def set_d2t(self, d2t: torch.Tensor) -> None:
        """``d2t`` is an EAGLE3-style offset table: target id = d2t[j] + j."""
        d2t = d2t.to(torch.long)
        self._d2t_raw = d2t.clone()  # round-8 H2 remap trace only
        absolute = d2t + torch.arange(d2t.numel(), device=d2t.device)
        mapped = absolute - DSPARK_UNIFIED_SPEECH_OFFSET
        valid = (mapped >= 0) & (mapped < self.vocab_size)
        self.d2t_omni.copy_(torch.where(valid, mapped, torch.full_like(mapped, -1)))
        self._valid_draft_ids = None
        self._valid_target_ids = None

    def _trace_remap(self, next_tokens: torch.Tensor, step_logits: torch.Tensor) -> None:
        """Round-8 H2: per proposal, the draft id j, the HF unified id d2t[j]+j
        and the served 6761-space id; also the draft-space argmax for comparison."""
        d2t_raw = getattr(self, "_d2t_raw", None)
        if d2t_raw is None:
            return
        out = []
        for r in range(int(next_tokens.shape[0])):
            tid = int(next_tokens[r])
            js = (self.d2t_omni == tid).nonzero(as_tuple=False).flatten()
            j = int(js[0]) if js.numel() else -1
            hf = int(d2t_raw[j]) + j if j >= 0 else -1
            out.append(
                dict(
                    served_id=tid,
                    draft_id=j,
                    hf_id=hf,
                    hf_minus_offset=hf - DSPARK_UNIFIED_SPEECH_OFFSET,
                    ok=(hf - DSPARK_UNIFIED_SPEECH_OFFSET) == tid,
                    n_draft_ids_for_served=int(js.numel()),
                    served_argmax=int(step_logits[r].argmax()),
                )
            )
        spec_trace(f"h2 remap: {out}")

    def _valid_ids(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._valid_draft_ids is None or self._valid_draft_ids.device != device:
            valid = self.d2t_omni >= 0
            self._valid_draft_ids = valid.nonzero(as_tuple=False).flatten().to(device)
            self._valid_target_ids = self.d2t_omni[valid].to(device)
        return self._valid_draft_ids, self._valid_target_ids

    def to_target_space(self, draft_logits: torch.Tensor) -> torch.Tensor:
        """(n, draft_vocab) draft logits -> (n, speech vocab) with -inf fill."""
        draft_ids, target_ids = self._valid_ids(draft_logits.device)
        out = draft_logits.new_full(
            (draft_logits.shape[0], self.vocab_size), float("-inf")
        )
        out[:, target_ids] = draft_logits[:, draft_ids]
        return out

    def apply_step_logits(self, logits, *, token_ids, hidden_states):
        step = super().apply_step_logits(
            logits, token_ids=token_ids, hidden_states=hidden_states
        )
        return self.to_target_space(step)

    def apply_block_logits(self, base_logits, *, token_ids, hidden_states):
        block = super().apply_block_logits(
            base_logits, token_ids=token_ids, hidden_states=hidden_states
        )
        shape = block.shape
        return self.to_target_space(block.reshape(-1, shape[-1])).view(
            *shape[:-1], self.vocab_size
        )

    @torch.no_grad()
    def sample_block(self, base_logits, *, first_prev_tokens, hidden_states, sampler):
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty = torch.empty(
                batch_size, 0, dtype=torch.long, device=base_logits.device
            )
            return empty, base_logits.new_empty((batch_size, 0, self.vocab_size))
        sampling_info = _DRAFT_SAMPLING_INFO["info"]
        mirror_topkp = _dspark_mirror_enabled("TOPKP")
        if SPEC_DEBUG and not getattr(self, "_mirror_traced", False):
            self._mirror_traced = True
            scaling = getattr(sampling_info, "acc_scaling_penalties", None)
            spec_trace(
                f"draft mirror: rep={_dspark_mirror_enabled('REP')} topkp={mirror_topkp} "
                f"info={type(sampling_info).__name__} scaling={None if scaling is None else tuple(scaling.shape)} "
                f"values={repetition_penalty_values(sampling_info)}"
            )
        # Draft-side mirror of the target's per-token repetition penalty: the
        # accumulated history scaling plus every block token sampled so far.
        penalties = None
        rep_values = None
        if _dspark_mirror_enabled("REP"):
            scaling = getattr(sampling_info, "acc_scaling_penalties", None)
            if scaling is not None and scaling.shape[0] == batch_size:
                penalties = scaling[:, : self.vocab_size].clone()
                rep_values = repetition_penalty_values(sampling_info)
        sampled, corrected = [], []
        prev_tokens = first_prev_tokens.long()
        for step_idx in range(proposal_len):
            step_hidden = None if hidden_states is None else hidden_states[:, step_idx]
            step_logits = self.apply_step_logits(
                base_logits[:, step_idx, :],
                token_ids=prev_tokens,
                hidden_states=step_hidden,
            )
            if penalties is not None:
                if rep_values is not None:
                    penalties.scatter_(
                        1,
                        prev_tokens.view(-1, 1).clamp(max=penalties.shape[1] - 1),
                        rep_values.to(penalties.dtype).view(-1, 1),
                    )
                step_logits = _scale_draft_logits(step_logits, penalties)
            if mirror_topkp:
                step_logits = _mask_draft_logits_top_kp(step_logits, sampling_info)
            next_tokens = sampler(step_logits, step_idx)
            if _REMAP_TRACE and getattr(self, "_remap_traced", 0) < 64:
                self._remap_traced = getattr(self, "_remap_traced", 0) + 1
                self._trace_remap(next_tokens, step_logits)
            sampled.append(next_tokens)
            corrected.append(step_logits.unsqueeze(1))
            prev_tokens = next_tokens
        return torch.stack(sampled, dim=1), torch.cat(corrected, dim=1)


def _scale_draft_logits(logits: torch.Tensor, penalties: torch.Tensor) -> torch.Tensor:
    """Multiplicative (repetition) penalty on draft logits, target semantics."""
    width = min(penalties.shape[1], logits.shape[1])
    pen = penalties[:, :width].to(logits.dtype)
    head = logits[:, :width]
    head = torch.where(head < 0, head * pen, head / pen)
    if width == logits.shape[1]:
        return head
    return torch.cat([head, logits[:, width:]], dim=1)


def _mask_draft_logits_top_kp(logits: torch.Tensor, sampling_info: Any) -> torch.Tensor:
    """Restrict the draft proposal to its own top-k / top-p support (the target's
    filter settings), keeping q consistent between sampling and verify."""
    top_ks = getattr(sampling_info, "top_ks", None)
    if top_ks is None or top_ks.shape[0] != logits.shape[0]:
        return logits
    probs = torch.softmax(logits.float(), dim=-1)
    keep = torch.ones_like(probs, dtype=torch.bool)
    if sampling_info.need_top_k_sampling:
        k = top_ks.view(-1).clamp(min=1, max=probs.shape[1])
        sorted_probs, _ = probs.sort(dim=-1, descending=True)
        kth = sorted_probs.gather(1, (k - 1).view(-1, 1))
        keep &= probs >= kth
    if sampling_info.need_top_p_sampling:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        cum = sorted_probs.cumsum(dim=-1) - sorted_probs
        keep_sorted = cum < sampling_info.top_ps.view(-1, 1)
        keep &= torch.zeros_like(keep).scatter(1, order, keep_sorted)
    return logits.masked_fill(~keep, float("-inf"))


class CosyVoice3DSparkDraftModel(DSparkDraftModel):
    """yuekai/cosyvoice3_llm_dspark remapped to Omni's 6761 speech id space.

    The stock DSpark draft shares the target's embedding / lm_head; this
    checkpoint ships its own (unified-vocab embedding, 8192-id draft head with
    ``d2t``). The draft owns a speech-vocab embedding (+1 trained mask row),
    the reduced head, and a Markov head evaluated in the speech space.
    """

    uses_own_vocab_modules = True

    def __init__(self, config: Any, quant_config: Any = None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        hidden = int(config.hidden_size)
        draft_vocab = int(getattr(config, "draft_vocab_size", 0) or 0)
        if draft_vocab <= 0:
            raise ValueError("CosyVoice3 DSpark draft requires draft_vocab_size")
        self.embed_tokens = nn.Embedding(DSPARK_EMBED_ROWS, hidden)
        self.lm_head = nn.Linear(hidden, draft_vocab, bias=False)
        # Width of the proposal distribution handed to verify / graph buffers.
        self.lm_head.org_vocab_size = TOTAL_VOCAB_SIZE
        self.markov_head = CosyVoice3MarkovHead(
            vocab_size=TOTAL_VOCAB_SIZE,
            draft_vocab_size=draft_vocab,
            markov_rank=int(config.markov_rank),
        )
        # Inference-only weights: the draft proposal runs on inference-mode
        # activations, which autograd refuses to pair with grad-tracked params.
        for module in (self.embed_tokens, self.lm_head, self.markov_head):
            for param in module.parameters():
                param.requires_grad_(False)

    def attach_shared_modules(self, *, embed_tokens: nn.Module, lm_head: nn.Module):
        raise RuntimeError("CosyVoice3 DSpark draft owns its vocab modules")

    def forward_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    @torch.no_grad()
    def compute_base_logits(self, hidden: torch.Tensor):
        weight = self.lm_head.weight
        if hidden.dtype != weight.dtype:
            hidden = hidden.to(weight.dtype)
        return torch.matmul(hidden, weight.T), None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters())
        backbone, confidence = [], []
        offset = DSPARK_UNIFIED_SPEECH_OFFSET
        for name, loaded in weights:
            if name == "embed_tokens.weight":
                rows = loaded[offset : offset + DSPARK_EMBED_ROWS]
                default_weight_loader(params["embed_tokens.weight"], rows)
            elif name == "lm_head.weight":
                default_weight_loader(params["lm_head.weight"], loaded)
            elif name == "markov_head.markov_w1.weight":
                rows = loaded[offset : offset + TOTAL_VOCAB_SIZE]
                default_weight_loader(params[name], rows)
            elif name == "markov_head.markov_w2.weight":
                default_weight_loader(params[name], loaded)
            elif name == "d2t":
                self.markov_head.set_d2t(loaded.to(self.markov_head.d2t_omni.device))
            elif name == "t2d":
                continue
            elif name.startswith("confidence_head."):
                confidence.append((name, loaded))
            else:
                backbone.append((name, loaded))
        # Backbone / fc / norms go through the DFlash loader; the DSpark mixin
        # loader would drop the vocab modules this draft owns.
        DFlashDraftModel.load_weights(self, backbone)
        self._load_confidence_weights(confidence_weights=confidence, params_dict=params)
        if (self.markov_head.d2t_omni < 0).all():
            raise ValueError("CosyVoice3 DSpark draft checkpoint has no usable d2t")


def normalize_cosyvoice3_dspark_config(config: dict) -> dict:
    """Flatten the speculators-format DSpark config for SGLang's loader.

    Nested ``transformer_layer_config`` fields move to the top level, the
    architecture becomes the Omni draft class, ``aux_hidden_state_layer_ids``
    become DFlash ``target_layer_ids``, and the vocab / mask ids are expressed
    in the draft-owned speech embedding space.
    """
    normalized = dict(config)
    layer_config = normalized.pop("transformer_layer_config", None) or {}
    for key, value in layer_config.items():
        normalized.setdefault(key, value)
    normalized.pop("auto_map", None)
    normalized["architectures"] = [DSPARK_DRAFT_ARCHITECTURE]
    normalized.setdefault("model_type", layer_config.get("model_type", "qwen3"))
    aux_ids = normalized.get("aux_hidden_state_layer_ids")
    if aux_ids is not None:
        normalized["target_layer_ids"] = [int(i) for i in aux_ids]
    normalized["vocab_size"] = DSPARK_EMBED_ROWS
    normalized["mask_token_id"] = DSPARK_MASK_TOKEN_ID
    normalized["tie_word_embeddings"] = False
    return normalized


EntryClass = FunCosyVoice3SGLangModel
