# SPDX-License-Identifier: Apache-2.0

"""Fun-CosyVoice3 LLM backbone for sglang-omni."""

from __future__ import annotations

import copy
import os
from typing import Any, Iterable, Optional, Tuple

import torch
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from torch import nn

from sglang_omni.utils.spec_debug import SPEC_DEBUG, spec_trace

VOCAB_SIZE = 6561
TOTAL_VOCAB_SIZE = VOCAB_SIZE + 200

SOS_ID = VOCAB_SIZE + 0
EOS_ID = VOCAB_SIZE + 1
TASK_ID = VOCAB_SIZE + 2
FILL_ID = VOCAB_SIZE + 3

CONTROL_TOKEN_IDS: tuple[int, ...] = tuple(range(VOCAB_SIZE, TOTAL_VOCAB_SIZE))

# Draft architecture name published by the AngelSlim EAGLE3 checkpoint.
EAGLE3_DRAFT_ARCHITECTURE = "CosyVoice3Eagle3LlamaForCausalLM"


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
        if SPEC_DEBUG and not getattr(self, "_traced_capture", False):
            self._traced_capture = True
            spec_trace(
                f"target forward mode={fwd_mode} capture_aux={self.capture_aux_hidden_states} "
                f"layers_to_capture={getattr(self.model, 'layers_to_capture', None)} "
                f"hidden={tuple(hidden_states.shape)} "
                f"aux={[tuple(a.shape) for a in aux_hidden_states] if aux_hidden_states is not None else None} "
                f"capture_mode={getattr(forward_batch, 'capture_hidden_mode', None)}"
            )

        return self.logits_processor(
            input_ids,
            hidden_states,
            self.llm_decoder,
            forward_batch,
            aux_hidden_states,
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


def _shift_prompt_embeds_default() -> bool:
    # EAGLE convention: the draft input at prompt position i is the embedding
    # of token i+1 (the target's input rows rolled left by one, last row =
    # the sampled first speech token). SGLang's multimodal EAGLE3 path feeds
    # unshifted rows instead; expose the alternative for acceptance probes.
    return os.environ.get("SGLANG_OMNI_COSYVOICE3_EAGLE3_UNSHIFTED_PROMPT", "0") != "1"


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
        if (
            input_embeds is None
            and mm_input_embeds is not None
            and forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
        ):
            input_embeds = self.prompt_input_embeds(
                mm_input_embeds, input_ids, forward_batch
            )
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
        embeds[last_indices] = self.model.embed_tokens(input_ids[last_indices])
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


EntryClass = FunCosyVoice3SGLangModel
