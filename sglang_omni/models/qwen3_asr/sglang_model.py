"""Qwen3-ASR model compatible with HuggingFace weights"""

import copy
import logging
from types import MethodType
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from sglang.srt.configs.qwen3_omni import Qwen3OmniMoeAudioEncoderConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeAudioEncoder
from sglang.srt.utils import add_prefix

from sglang_omni.platforms import current_platform

from .configuration_qwen3_asr import Qwen3ASRConfig
from .encoder_cuda_graph import (
    Qwen3ASREncoderLayerStackGraphRunner,
    build_buckets,
    eager_preamble,
    window_lens_from_token_counts,
)

logger = logging.getLogger(__name__)
fused_qk_norm_rope = current_platform.get_fused_qk_norm_rope()

_MROPE_ONLY_KEYS = frozenset({"interleaved", "mrope_interleaved", "mrope_section"})

# Keys written by request_builders when a speculative draft must re-encode.
DRAFT_AUDIO_FEATURE_KEY = "draft_audio_feature"
DRAFT_FEATURE_MASK_KEY = "draft_feature_attention_mask"
# Keeps the draft's embedding-cache entries apart from the target's, which
# share SGLang's process-wide cache keyed by item hash.
_DRAFT_MM_HASH_SALT = 0x5D2A_F7C3_9E1B_4A65


def draft_view_mm_inputs(mm_inputs: list[Any] | None) -> list[Any] | None:
    """Return per-request MultimodalInputs copies the draft can embed itself.

    The target's items carry the target tower's ``precomputed_embeddings`` and
    no mel; the draft tower has a different width, so each audio item is
    replaced by a copy holding the retained mel, no precomputed embeddings and
    a salted hash (``pad_value`` is kept so placeholder matching is unchanged).
    """
    if mm_inputs is None:
        return None
    views: list[Any] = []
    for mm_input in mm_inputs:
        if mm_input is None:
            views.append(None)
            continue
        items: list[Any] = []
        for item in mm_input.mm_items:
            if item is None or not item.is_modality(Modality.AUDIO):
                items.append(item)
                continue
            data = item.model_specific_data or {}
            mel = data.get(DRAFT_AUDIO_FEATURE_KEY)
            if mel is None:
                raise RuntimeError(
                    "Qwen3-ASR speculative draft needs the request's mel features; "
                    "the request builder must keep_draft_audio_features"
                )
            view = copy.copy(item)
            view.model_specific_data = {
                key: value
                for key, value in data.items()
                if key not in (DRAFT_AUDIO_FEATURE_KEY, DRAFT_FEATURE_MASK_KEY)
            }
            view.model_specific_data["feature_attention_mask"] = data.get(
                DRAFT_FEATURE_MASK_KEY
            )
            view.feature = mel
            view.precomputed_embeddings = None
            view.format = MultimodalInputFormat.NORMAL
            view.hash = int(item.hash) ^ _DRAFT_MM_HASH_SALT
            items.append(view)
        view_input = copy.copy(mm_input)
        view_input.mm_items = items
        views.append(view_input)
    return views


def _normalize_asr_text_rope(text_config: Any) -> None:
    # note (luojiaxuan): ASR has no spatial axes: all three MRoPE position
    # rows are identical, so ordinary text RoPE is numerically equivalent and
    # avoids the multimodal permutation/copy path on every decoder layer.
    for attr in ("rope_parameters", "rope_scaling"):
        parameters = getattr(text_config, attr, None)
        if not isinstance(parameters, dict) or "mrope_section" not in parameters:
            continue
        setattr(
            text_config,
            attr,
            {
                key: value
                for key, value in parameters.items()
                if key not in _MROPE_ONLY_KEYS
            },
        )


def _fused_asr_forward_prepare_native(
    attention: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hidden_states.dtype != torch.bfloat16 or fused_qk_norm_rope is None:
        return attention._asr_unfused_forward_prepare_native(
            positions,
            hidden_states,
        )
    if positions.dtype != torch.int32:
        # note(ratish): the prefill CUDA graph bypasses forward()'s int32
        # cast; the kernel rejects int64 positions.
        positions = positions.to(torch.int32)
    qkv, _ = attention.qkv_proj(hidden_states)
    fused_qk_norm_rope(
        qkv,
        attention.num_heads,
        attention.num_kv_heads,
        attention.num_kv_heads,
        attention.head_dim,
        attention.q_norm.variance_epsilon,
        attention.q_norm.weight,
        attention.k_norm.weight,
        attention.rope_theta,
        attention.rotary_emb.is_neox_style,
        positions,
        1.0,
        0,
        0,
        1.0,
    )
    return qkv.split(
        [attention.q_size, attention.kv_size, attention.kv_size],
        dim=-1,
    )


def _enable_fused_asr_qk_norm_rope(language_model: nn.Module) -> None:
    # note (luojiaxuan): the sgl-kernel op operates in place on packed QKV and
    # replaces split + two RMSNorm launches + RoPE for the equivalent text
    # positions. Keep the original bound method for non-bfloat16 fallbacks.
    if fused_qk_norm_rope is None:
        return

    for layer in language_model.model.layers:
        attention = layer.self_attn
        if attention.head_dim not in (64, 128, 256):
            continue
        attention._asr_unfused_forward_prepare_native = attention.forward_prepare_native
        attention.forward_prepare_native = MethodType(
            _fused_asr_forward_prepare_native,
            attention,
        )


class Qwen3ASRForConditionalGeneration(nn.Module):
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3ASRConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        thinker_config = config.thinker_config

        if getattr(thinker_config, "audio_config", None) is None:
            thinker_config.audio_config = Qwen3OmniMoeAudioEncoderConfig()

        _normalize_asr_text_rope(thinker_config.text_config)

        self.audio_tower = Qwen3OmniMoeAudioEncoder(thinker_config.audio_config)
        self.language_model = Qwen3ForCausalLM(
            thinker_config.text_config,
            quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        _enable_fused_asr_qk_norm_rope(self.language_model)
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        self._encoder_graph_runner: Qwen3ASREncoderLayerStackGraphRunner | None = None
        # "target" / "draft" under STANDALONE speculative decoding, else None.
        self.spec_role: str | None = None

    def init_encoder_graphs(
        self, *, max_batch_size: int, max_tokens_per_clip: int
    ) -> None:
        runner = Qwen3ASREncoderLayerStackGraphRunner(
            self.audio_tower,
            buckets=build_buckets(max_batch_size, max_tokens_per_clip),
            max_batch_size=max_batch_size,
        )
        runner.capture_all()
        self._encoder_graph_runner = runner

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        device = next(self.audio_tower.parameters()).device

        features = [item.feature for item in items]
        masks = [getattr(item, "feature_attention_mask", None) for item in items]

        # SGLang batches every cache miss in the forward batch into a single
        # data_embedding_func call (_batch_encode_per_image_misses), so
        # mixed-length mel features must agree on a frame count. Pad to the
        # batch max: every frame added here is masked out before the audio
        # tower runs, so it sees exactly the frames it saw one-at-a-time.
        frame_lens = [feature.shape[-1] for feature in features]
        max_frames = max(frame_lens)
        has_mask = any(mask is not None for mask in masks)
        if min(frame_lens) != max_frames or has_mask:
            normalized_masks = []
            for feature, mask, length in zip(features, masks, frame_lens):
                if mask is None:
                    mask = torch.ones(
                        (feature.shape[0], length),
                        dtype=torch.long,
                        device=device,
                    )
                elif mask.shape != (feature.shape[0], length):
                    raise ValueError(
                        "Qwen3-ASR feature_attention_mask shape "
                        f"{tuple(mask.shape)} does not match feature batch/frames "
                        f"{(feature.shape[0], length)}"
                    )
                else:
                    # Item masks can arrive on CPU while features are already
                    # on the model device; move for the cat below.
                    mask = mask.to(device=device)
                normalized_masks.append(
                    nn.functional.pad(mask, (0, max_frames - length))
                )
            masks = normalized_masks
            has_mask = True
            if min(frame_lens) != max_frames:
                features = [
                    nn.functional.pad(feature, (0, max_frames - feature.shape[-1]))
                    for feature in features
                ]

        input_features = torch.cat(features).type(self.audio_tower.dtype).to(device)

        if has_mask:
            feature_attention_mask = torch.cat(masks, dim=0).type(torch.long).to(device)
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[
                feature_attention_mask.bool()
            ].permute(1, 0)
        else:
            audio_feature_lengths = torch.tensor(
                [input_features.shape[-1]] * input_features.shape[0],
                dtype=torch.long,
                device=device,
            )
            input_features = input_features.permute(0, 2, 1).reshape(
                -1, input_features.shape[1]
            )

        runner = self._encoder_graph_runner
        if runner is not None:
            token_counts = [
                (item.model_specific_data or {}).get("num_audio_tokens")
                for item in items
            ]
            if all(count is not None for count in token_counts):
                window_lens = window_lens_from_token_counts(
                    token_counts, tokens_per_window=runner.tokens_per_window
                )
                hidden = eager_preamble(
                    self.audio_tower, input_features, audio_feature_lengths
                )
                graphed = runner.run(hidden, window_lens)
                if graphed is not None:
                    return graphed.unsqueeze(0)

        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=audio_feature_lengths,
        )
        return audio_outputs.last_hidden_state

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        spec_role = getattr(self, "spec_role", None)
        if forward_batch.mrope_positions is not None and spec_role != "draft":
            positions = forward_batch.mrope_positions[0]
        # note (Junnan Li): the speculative draft keeps the plain 1-D positions.
        # ASR MRoPE is degenerate (all rows == positions), and SGLang's draft
        # decode CUDA graph replays refresh ``positions`` but not
        # ``mrope_positions``; reading the stale row degrades draft proposals
        # (accept length ~2.6 vs ~3.9 of 4 measured on Qwen3-ASR-0.6B).
        if (
            spec_role == "draft"
            and forward_batch.mm_inputs is not None
            and not forward_batch.forward_mode.is_decode()
            and not forward_batch.forward_mode.is_target_verify()
        ):
            forward_batch.mm_inputs = draft_view_mm_inputs(forward_batch.mm_inputs)
        positions = positions.to(
            dtype=(torch.int32 if fused_qk_norm_rope is not None else torch.long),
            device=input_ids.device,
            non_blocking=True,
        ).contiguous()

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.AUDIO: self.get_audio_feature,
            },
            positions=positions,
        )
        return hidden_states

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    @property
    def lm_head(self) -> nn.Module:
        return self.language_model.lm_head

    def get_embed_and_head(self):
        return self.language_model.get_embed_and_head()

    def set_dflash_layers_to_capture(self, layer_ids: list[int]) -> None:
        # Qwen3ForCausalLM owns the HF-output to SGLang-input layer shift.
        self.capture_aux_hidden_states = True
        self.language_model.set_dflash_layers_to_capture(layer_ids)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        llm_stacked_params = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        # Audio tower has separate q/k/v in checkpoint → stack into qkv_proj
        audio_stacked_params = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            if (
                getattr(
                    self.config.thinker_config.text_config, "tie_word_embeddings", False
                )
                and "lm_head.weight" in name
            ):
                continue

            if "talker" in name or "code2wav" in name:
                continue

            if name.startswith("thinker.audio_tower."):
                name = name.replace("thinker.audio_tower.", "audio_tower.", 1)
            elif name.startswith("thinker.lm_head."):
                name = name.replace("thinker.lm_head.", "language_model.lm_head.", 1)
            elif name.startswith("thinker.model."):
                name = name.replace("thinker.model.", "language_model.model.", 1)

            is_audio = "audio_tower" in name

            # Audio tower: remap out_proj → proj for VisionAttention
            if is_audio and "out_proj" in name:
                name = name.replace("out_proj", "proj")

            stacked_params = audio_stacked_params if is_audio else llm_stacked_params

            for param_name, weight_name, shard_id in stacked_params:
                if weight_name not in name:
                    continue
                name_tmp = name.replace(weight_name, param_name)
                if name_tmp.endswith(".bias") and name_tmp not in params_dict:
                    continue
                if name_tmp not in params_dict:
                    continue
                param = params_dict[name_tmp]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = Qwen3ASRForConditionalGeneration
