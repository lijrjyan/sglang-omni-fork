# SPDX-License-Identifier: Apache-2.0
"""SGLang-native Whisper ASR model.

The Whisper encoder runs as the encoder side of an encoder-decoder SGLang
request. The decoder uses RadixAttention for both autoregressive self-attention
and cached cross-attention over encoder states, so normal SGLang KV cache,
CUDA Graph, and torch.compile paths apply to decode.
"""

from __future__ import annotations

from typing import Any, Iterable, Tuple

import torch
import torch.nn.functional as F
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from torch import nn
from transformers import WhisperConfig
from transformers.activations import ACT2FN

from sglang_omni.models.whisper_asr.encoder_compile import compile_encoder_layers
from sglang_omni.models.whisper_asr.encoder_cuda_graph import (
    WhisperEncoderCudaGraphRunner,
)
from sglang_omni.models.whisper_asr.encoder_share import EncoderStateShare
from sglang_omni.models.whisper_asr.quantization_scope import decoder_quant_config


class WhisperEncoderAttention(nn.Module):
    def __init__(
        self,
        config: WhisperConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.qkv_proj = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            self.num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.out_proj = RowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

    def _shape(self, states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = states.shape
        return states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)
        attn_output = F.scaled_dot_product_attention(
            self._shape(query),
            self._shape(key),
            self._shape(value),
            dropout_p=0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.embed_dim,
        )
        attn_output, _ = self.out_proj(attn_output)
        return attn_output


class WhisperEncoderLayer(nn.Module):
    def __init__(
        self,
        config: WhisperConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = WhisperEncoderAttention(
            config, quant_config=quant_config, prefix=f"{prefix}.self_attn"
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = ColumnParallelLinear(
            config.d_model,
            config.encoder_ffn_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            config.encoder_ffn_dim,
            config.d_model,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.activation_fn = ACT2FN[config.activation_function]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states, _ = self.fc2(self.activation_fn(hidden_states))
        return residual + hidden_states


class WhisperEncoder(nn.Module):
    def __init__(
        self,
        config: WhisperConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(
            config.num_mel_bins,
            config.d_model,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.embed_positions = nn.Embedding(config.max_source_positions, config.d_model)
        self.layers = nn.ModuleList(
            [
                WhisperEncoderLayer(
                    config, quant_config=quant_config, prefix=f"{prefix}.layers.{i}"
                )
                for i in range(config.encoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        # Note:(Chenchen Hong) move input_features to the conv weight's device
        # (not just dtype), else the CUDA conv1 raises a device-mismatch error.
        hidden_states = input_features.to(
            device=self.conv1.weight.device, dtype=self.conv1.weight.dtype
        )
        hidden_states = F.gelu(self.conv1(hidden_states))
        hidden_states = F.gelu(self.conv2(hidden_states))
        hidden_states = hidden_states.permute(0, 2, 1)

        embed_pos = self.embed_positions.weight[: hidden_states.shape[1]]
        hidden_states = hidden_states + embed_pos.to(hidden_states.device)

        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.layer_norm(hidden_states)


class WhisperSGLangSelfAttention(nn.Module):
    def __init__(
        self,
        config: WhisperConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.decoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv_proj = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            self.num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.out_proj = RowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_heads,
            layer_id=layer_id,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)
        attn_output = self.attn(
            query.reshape(-1, self.num_heads, self.head_dim),
            key.reshape(-1, self.num_heads, self.head_dim),
            value.reshape(-1, self.num_heads, self.head_dim),
            forward_batch,
        )
        attn_output, _ = self.out_proj(attn_output)
        return attn_output


class WhisperSGLangCrossAttention(nn.Module):
    def __init__(
        self,
        config: WhisperConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.decoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = ColumnParallelLinear(
            self.embed_dim,
            self.embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.kv_proj = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            total_num_heads=0,
            total_num_kv_heads=self.num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
        )
        self.out_proj = RowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_heads,
            layer_id=layer_id,
            is_cross_attention=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor | None,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        query, _ = self.q_proj(hidden_states)
        query = query.view(-1, self.num_heads, self.head_dim)
        if cross_attention_states is None:
            key = value = None
        else:
            kv, _ = self.kv_proj(cross_attention_states)
            key, value = kv.chunk(2, dim=-1)
            key = key.reshape(-1, self.num_heads, self.head_dim)
            value = value.reshape(-1, self.num_heads, self.head_dim)
        attn_output = self.attn(query, key, value, forward_batch)
        attn_output, _ = self.out_proj(attn_output)
        return attn_output


class WhisperDecoderLayer(nn.Module):
    def __init__(
        self,
        config: WhisperConfig,
        layer_idx: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        num_decoder_layers = int(config.decoder_layers)
        self.self_attn = WhisperSGLangSelfAttention(
            config,
            layer_id=layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.encoder_attn = WhisperSGLangCrossAttention(
            config,
            layer_id=num_decoder_layers + layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder_attn",
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = ColumnParallelLinear(
            config.d_model,
            config.decoder_ffn_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            config.decoder_ffn_dim,
            config.d_model,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.activation_fn = ACT2FN[config.activation_function]

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor | None,
        forward_batch: ForwardBatch,
        skip_cross_attention: bool,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, forward_batch)
        hidden_states = residual + hidden_states

        if not skip_cross_attention:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states = self.encoder_attn(
                hidden_states,
                cross_attention_states,
                forward_batch,
            )
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states, _ = self.fc2(self.activation_fn(hidden_states))
        return residual + hidden_states


class WhisperDecoder(nn.Module):
    def __init__(
        self,
        config: WhisperConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_positions = nn.Embedding(
            config.max_target_positions,
            config.d_model,
        )
        self.layers = nn.ModuleList(
            [
                WhisperDecoderLayer(
                    config,
                    layer_idx=i,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(config.decoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cross_attention_states: torch.Tensor | None,
        forward_batch: ForwardBatch,
        skip_cross_attention: bool,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states + self.embed_positions(positions).to(
            hidden_states.device
        )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cross_attention_states,
                forward_batch,
                skip_cross_attention,
            )
        return self.layer_norm(hidden_states)


class WhisperModel(nn.Module):
    def __init__(
        self,
        config: WhisperConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.encoder = WhisperEncoder(
            config, quant_config=quant_config, prefix=f"{prefix}.encoder"
        )
        self.decoder = WhisperDecoder(
            config,
            quant_config=decoder_quant_config(config, quant_config),
            prefix=f"{prefix}.decoder",
        )


class WhisperForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: WhisperConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.model = WhisperModel(config, quant_config=quant_config, prefix="model")
        self.proj_out = self.model.decoder.embed_tokens
        self.lm_head = self.proj_out
        self.logits_processor = LogitsProcessor(config)
        self.start_layer = 0
        self.end_layer = int(config.decoder_layers) * 2
        self._encoder_graph_runner: WhisperEncoderCudaGraphRunner | None = None
        self.encoder_compiled = False
        # note (Junnan Li): speculative draft/target pairs that ship the same
        # encoder weights share one encoder run per request via this object.
        self.encoder_share: EncoderStateShare | None = None
        self.encoder_share_role: str | None = None

    def compile_encoder(
        self,
        input_feature_len: int,
        *,
        mode: str | None = None,
    ) -> bool:
        """Fuse the encoder layers with torch.compile before graph capture."""
        self.encoder_compiled = compile_encoder_layers(
            self.model.encoder,
            num_mel_bins=self.config.num_mel_bins,
            input_feature_len=input_feature_len,
            mode=mode,
        )
        return self.encoder_compiled

    def init_encoder_graphs(
        self,
        batch_buckets: list[int] | tuple[int, ...],
        input_feature_len: int,
    ) -> None:
        """Capture fixed-shape Whisper encoder batches after model setup."""
        if not batch_buckets:
            return
        self._encoder_graph_runner = WhisperEncoderCudaGraphRunner(
            self.model.encoder,
            num_mel_bins=int(self.config.num_mel_bins),
            input_feature_len=int(input_feature_len),
        )
        self._encoder_graph_runner.capture(batch_buckets)

    def _batch_audio_inputs(
        self,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor | None, list[int] | None, list[int]]:
        if forward_batch.forward_mode.is_decode() or all(forward_batch.encoder_cached):
            return None, None, []

        features: list[torch.Tensor] = []
        encoder_lens: list[int] = []
        req_pool_indices: list[int] = []
        for index, mm_input in enumerate(forward_batch.mm_inputs):
            if forward_batch.encoder_cached[index] or mm_input is None:
                continue
            item_features = [
                item.feature for item in mm_input.mm_items if item.feature is not None
            ]
            if not item_features:
                continue
            features.append(torch.cat(item_features, dim=0))
            encoder_lens.append(int(forward_batch.encoder_lens[index].item()))
            req_pool_indices.append(int(forward_batch.req_pool_indices[index].item()))

        if not features:
            return None, None, []
        return torch.cat(features, dim=0), encoder_lens, req_pool_indices

    @staticmethod
    def _flat_encoder_result(
        encoder_states: torch.Tensor,
        encoder_lens: list[int],
    ) -> torch.Tensor:
        hidden_size = encoder_states.shape[-1]
        total_encoder_len = sum(encoder_lens)
        flat = torch.empty(
            total_encoder_len,
            hidden_size,
            device=encoder_states.device,
            dtype=encoder_states.dtype,
        )
        dst_start = 0
        for index, encoder_len in enumerate(encoder_lens):
            dst_end = dst_start + encoder_len
            flat[dst_start:dst_end] = encoder_states[index, :encoder_len]
            dst_start = dst_end
        return flat

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> Any:
        del kwargs
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        audio_features, encoder_lens, req_pool_indices = self._batch_audio_inputs(
            forward_batch
        )
        cross_attention_states = None

        if get_is_capture_mode():
            skip_cross_attention = False
        else:
            skip_cross_attention = forward_batch.encoder_lens.max() == 0

        if audio_features is not None and encoder_lens is not None:
            if self.encoder_share_role == "draft":
                cross_attention_states = self.encoder_share.take(req_pool_indices)
            if cross_attention_states is None:
                if self._encoder_graph_runner is not None:
                    encoder_states = self._encoder_graph_runner.run(audio_features)
                else:
                    encoder_states = self.model.encoder(audio_features)
                cross_attention_states = self._flat_encoder_result(
                    encoder_states,
                    encoder_lens,
                )
                if self.encoder_share_role == "target":
                    self.encoder_share.publish(
                        req_pool_indices, cross_attention_states, encoder_lens
                    )

        hidden_states = self.model.decoder(
            input_ids=input_ids,
            positions=positions,
            cross_attention_states=cross_attention_states,
            forward_batch=forward_batch,
            skip_cross_attention=skip_cross_attention,
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    _STACKED_PARAMS_MAPPING = (
        (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
        (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
        (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
        (".encoder_attn.kv_proj", ".encoder_attn.k_proj", "k"),
        (".encoder_attn.kv_proj", ".encoder_attn.v_proj", "v"),
    )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        weights_dict = dict(weights)
        if "proj_out.weight" in weights_dict:
            weights_dict.setdefault(
                "model.decoder.embed_tokens.weight", weights_dict.pop("proj_out.weight")
            )
        # note (Junnan Li): Whisper checkpoints have no k_proj bias, but the fused
        # qkv/kv projections carry one bias; feed the k shard explicit zeros.
        for name, weight in list(weights_dict.items()):
            if name.endswith("k_proj.weight"):
                weights_dict.setdefault(
                    name[: -len("weight")] + "bias", torch.zeros_like(weight[:, 0])
                )
        for name, loaded_weight in weights_dict.items():
            for param_name, weight_name, shard_id in self._STACKED_PARAMS_MAPPING:
                if weight_name not in name:
                    continue
                param = params_dict[name.replace(weight_name, param_name)]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = WhisperForConditionalGeneration
