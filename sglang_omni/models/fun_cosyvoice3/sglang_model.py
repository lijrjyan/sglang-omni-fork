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

    def set_speculative_sampling_info(self, sampling_info):
        """Publish live state before drafting, after scheduler penalty refresh.

        Target forward sees a copy and runs after drafting. Its previous copy
        cannot supply the next draft's history or filtered batch layout.
        This Python hook does not update operations captured in a CUDA graph.
        """
        _DRAFT_SAMPLING_INFO["info"] = sampling_info
        orchestrator = getattr(sampling_info, "penalizer_orchestrator", None)
        _DRAFT_SAMPLING_INFO["rep_values"] = None
        if orchestrator is not None and orchestrator.is_required:
            from sglang.srt.sampling.penaltylib import BatchedRepetitionPenalizer

            penalty = orchestrator.penalizers.get(BatchedRepetitionPenalizer)
            if penalty is not None and penalty.is_prepared():
                _DRAFT_SAMPLING_INFO["rep_values"] = penalty.repetition_penalties

    def set_dspark_layers_to_capture(self, layer_ids: Optional[list[int]] = None):
        """Capture normalized SGLang block-output ids from the draft config."""
        if layer_ids is None:
            raise ValueError("DSpark requires explicit target layer ids to capture")
        layer_ids = list(layer_ids)
        # Diagnostic override relative to the normalized capture contract.
        self.set_eagle3_layers_to_capture(layer_ids)

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

        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        logits_output = self.logits_processor(
            input_ids,
            hidden_states,
            self.llm_decoder,
            forward_batch,
            aux_hidden_states,
        )
        return logits_output

    def postprocess_speculative_logits(self, logits_output, forward_batch) -> None:
        """Apply live sampling state after eager forward or CUDA graph replay.

        These penalties must not be captured inside model.forward: graph
        warmup has no request penalizer, and replay bypasses Python branches.
        Native speculative verification consumes the corrected logits next.
        """
        sampling_info = forward_batch.sampling_info
        _DRAFT_SAMPLING_INFO["info"] = sampling_info
        if not (
            forward_batch.forward_mode.is_target_verify()
            and _is_dspark_forward(forward_batch)
        ):
            return
        logits = logits_output.next_token_logits
        apply_block_repetition_penalty(logits, forward_batch.input_ids, sampling_info)

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
    # unshifted rows instead.
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
        out = self._stock_model_forward(
            self.model,
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        return out

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
        last_rows = self.model.embed_tokens(input_ids[last_indices])
        embeds[last_indices] = last_rows
        return embeds


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
        absolute = d2t + torch.arange(d2t.numel(), device=d2t.device)
        mapped = absolute - DSPARK_UNIFIED_SPEECH_OFFSET
        valid = (mapped >= 0) & (mapped < self.vocab_size)
        self.d2t_omni.copy_(torch.where(valid, mapped, torch.full_like(mapped, -1)))
        self._valid_draft_ids = None
        self._valid_target_ids = None

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
    normalized["dspark_bonus_anchor"] = True
    normalized.setdefault("model_type", layer_config.get("model_type", "qwen3"))
    aux_ids = normalized.get("aux_hidden_state_layer_ids")
    if aux_ids is not None:
        # The checkpoint indexes HF hidden_states (input of block i).
        # SGLang's explicit capture ids select output of block i instead.
        normalized["target_layer_ids"] = [int(i) - 1 for i in aux_ids]
    normalized["vocab_size"] = DSPARK_EMBED_ROWS
    normalized["mask_token_id"] = DSPARK_MASK_TOKEN_ID
    normalized["tie_word_embeddings"] = False
    return normalized


EntryClass = FunCosyVoice3SGLangModel
