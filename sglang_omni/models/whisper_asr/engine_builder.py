# SPDX-License-Identifier: Apache-2.0
"""Whisper ASR SGLang engine builder."""

from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import Any

from sglang_omni.model_runner.model_worker import register_speculative_target
from sglang_omni.models.whisper_asr.encoder_service import (
    WhisperPreLMEncoderService,
    build_cache_namespace,
)
from sglang_omni.scheduling.engine_factory import AsrEngineBuilder
from sglang_omni.scheduling.speculative_model_support import SpeculativeModelSupport

logger = logging.getLogger(__name__)

# STANDALONE drafts share the target architecture, so the draft loader must
# resolve Omni's Whisper class through SGLang's registry.
register_speculative_target(
    "WhisperForConditionalGeneration",
    omni_model_cls=(
        "sglang_omni.models.whisper_asr.sglang_model:WhisperForConditionalGeneration"
    ),
)

_DEFAULT_ENCODER_GRAPH_BATCH_BUCKETS = (1, 2, 4, 8, 12, 16)
_WHISPER_GENERATION_ARCHITECTURES = {"WhisperForConditionalGeneration"}


def _normalize_encoder_graph_buckets(buckets: list[int] | None) -> tuple[int, ...]:
    values = _DEFAULT_ENCODER_GRAPH_BATCH_BUCKETS if buckets is None else buckets
    normalized = {int(value) for value in values}
    return tuple(sorted(value for value in normalized if value >= 1))


def _resolve_encoder_graph_buckets(
    buckets: tuple[int, ...],
    *,
    enable_pre_lm_encoder: bool,
    pre_lm_max_batch_size: int,
    max_prefill_tokens: int,
    encoder_token_count: int,
    max_running_requests: int | None,
) -> tuple[int, ...]:
    # note (guozhihao-224): pre-LM encoding is no longer limited by atomic prefill
    # admission, so capture against pre_lm_max_batch_size. Keep
    # max_prefill_tokens // encoder_token_count only for encoder-in-prefill.
    if enable_pre_lm_encoder:
        capture_limit = pre_lm_max_batch_size
    else:
        if max_prefill_tokens < 1:
            raise ValueError(
                f"max_prefill_tokens must be >= 1, got {max_prefill_tokens}"
            )
        if encoder_token_count < 1:
            raise ValueError(
                f"encoder_token_count must be >= 1, got {encoder_token_count}"
            )
        capture_limit = max_prefill_tokens // encoder_token_count
    if max_running_requests is not None:
        if max_running_requests < 1:
            raise ValueError(
                "max_running_requests must be >= 1, " f"got {max_running_requests}"
            )
        capture_limit = min(capture_limit, max_running_requests)
    resolved = {bucket for bucket in buckets if bucket <= capture_limit}
    if max_running_requests is not None and capture_limit >= 1:
        resolved.add(capture_limit)
    return tuple(sorted(resolved))


def _validate_whisper_speculative_configs(
    target_config: Any,
    draft_config: Any,
    *,
    num_draft_tokens: int,
    decoder_context_len: int,
) -> None:
    draft_architectures = set(draft_config.architectures or [])
    if not draft_architectures.intersection(_WHISPER_GENERATION_ARCHITECTURES):
        raise ValueError(
            "Whisper speculative decoding requires a Whisper architecture for the "
            f"draft checkpoint; got {sorted(draft_architectures)!r}"
        )

    compatibility_values = (
        ("d_model", target_config.d_model, draft_config.d_model),
        ("vocab_size", target_config.vocab_size, draft_config.vocab_size),
        ("num_mel_bins", target_config.num_mel_bins, draft_config.num_mel_bins),
        (
            "max_source_positions",
            target_config.max_source_positions,
            draft_config.max_source_positions,
        ),
    )
    for field, target_value, draft_value in compatibility_values:
        if draft_value != target_value:
            raise ValueError(
                "Whisper target and draft configs must match for speculative "
                f"decoding: {field} target={target_value!r}, draft={draft_value!r}"
            )

    if draft_config.max_target_positions < decoder_context_len:
        raise ValueError(
            "Whisper draft max_target_positions is smaller than the target"
        )
    if 2 * num_draft_tokens + 4 >= decoder_context_len:
        raise ValueError(
            "Whisper speculative_num_draft_tokens exceeds the decoder budget: "
            f"two {num_draft_tokens}-token windows plus the task prefix "
            f"must fit {decoder_context_len} positions"
        )


class WhisperSpeculativeSupport(SpeculativeModelSupport):
    """Whisper: encoder-decoder target with a STANDALONE Whisper draft."""

    def __init__(self, *, share_encoder: bool, decoder_context_len: int) -> None:
        self.share_encoder = bool(share_encoder)
        self.decoder_context_len = int(decoder_context_len)

    def validate_speculative_draft(
        self,
        target_cfg: Any,
        draft_cfg: Any,
        algorithm: str,
        num_draft_tokens: int,
    ) -> None:
        del algorithm  # STANDALONE only; the lane admits nothing else for Whisper
        _validate_whisper_speculative_configs(
            target_cfg,
            draft_cfg,
            num_draft_tokens=num_draft_tokens,
            decoder_context_len=self.decoder_context_len,
        )

    def reject_speculative_request(self, request: Any) -> str | None:
        from sglang_omni.models.whisper_asr.request_builders import (
            reject_speculative_request,
        )

        return reject_speculative_request(request)

    def position_reserve(self, num_draft_tokens: int) -> int:
        # Reserve full verify and draft-repair windows: both can evaluate
        # positions beyond the request's final accepted output token.
        return 2 * int(num_draft_tokens)

    def link_models(self, target_model: Any, draft_model: Any) -> None:
        super().link_models(target_model, draft_model)
        if not self.share_encoder:
            return
        from sglang_omni.models.whisper_asr.encoder_share import EncoderStateShare

        # note (Junnan Li): distil-whisper drafts ship the target's encoder
        # weights verbatim, so the draft reads the target's states instead of
        # re-running a 1500-position encoder per request.
        share = EncoderStateShare()
        target_model.encoder_share = share
        draft_model.encoder_share = share
        logger.info("Whisper speculative draft shares the target encoder output")


class WhisperASREngineBuilder(AsrEngineBuilder):
    model_name = "Whisper ASR"
    model_arch_override = "WhisperForConditionalGeneration"

    def __init__(
        self,
        *,
        max_running_requests: int,
        max_new_tokens: int,
        mem_fraction_static: float,
        enable_encoder_cuda_graph: bool = False,
        encoder_graph_batch_buckets: list[int] | None = None,
        request_build_max_workers: int = 8,
        enable_async_decode: bool = True,
        async_decode_min_batch_size: int = 2,
        request_build_max_pending: int | None = 16,
        prefill_coalesce_requests: int = 2,
        prefill_coalesce_wait_ms: float = 6.0,
        prefill_coalesce_when_idle: bool = True,
        prefill_coalesce_requires_pending_builds: bool = True,
        prefill_coalesce_after_builds_during_decode: bool = False,
        enable_speculative: bool = False,
        speculative_draft_model_path: str | None = None,
        speculative_num_steps: int = 3,
        speculative_num_draft_tokens: int = 4,
        speculative_cuda_graph: bool = False,
        speculative_draft_cuda_graph: bool = False,
        speculative_share_encoder: bool = True,
        enable_pre_lm_encoder: bool = True,
        pre_lm_cache_max_entries: int = 1024,
        pre_lm_cache_size_bytes: int | None = None,
        pre_lm_max_batch_size: int = 8,
        pre_lm_max_batch_wait_ms: int = 0,
        pre_lm_cache_pin_host_memory: bool = True,
    ) -> None:
        if pre_lm_max_batch_size < 1:
            raise ValueError(
                f"pre_lm_max_batch_size must be >= 1, got {pre_lm_max_batch_size}"
            )
        if pre_lm_max_batch_wait_ms < 0:
            raise ValueError(
                f"pre_lm_max_batch_wait_ms must be >= 0, got {pre_lm_max_batch_wait_ms}"
            )
        self.max_running_requests = max_running_requests
        self.max_new_tokens = max_new_tokens
        self.mem_fraction_static = mem_fraction_static
        self.enable_encoder_cuda_graph = bool(enable_encoder_cuda_graph)
        self._using_default_encoder_graph_buckets = encoder_graph_batch_buckets is None
        self.encoder_graph_batch_buckets = _normalize_encoder_graph_buckets(
            encoder_graph_batch_buckets
        )
        if enable_speculative and enable_async_decode:
            # note (Junnan Li): the STANDALONE lane is synchronous by design, so
            # the async-decode default is turned off instead of failing startup.
            logger.info("Whisper speculative decoding disables enable_async_decode")
            enable_async_decode = False
        self.enable_async_decode = enable_async_decode
        self.async_decode_min_batch_size = async_decode_min_batch_size
        self.request_build_max_workers = request_build_max_workers
        self.request_build_max_pending = request_build_max_pending
        self.prefill_coalesce_requests = prefill_coalesce_requests
        self.prefill_coalesce_wait_ms = prefill_coalesce_wait_ms
        self.prefill_coalesce_when_idle = prefill_coalesce_when_idle
        self.prefill_coalesce_requires_pending_builds = (
            prefill_coalesce_requires_pending_builds
        )
        self.prefill_coalesce_after_builds_during_decode = (
            prefill_coalesce_after_builds_during_decode
        )
        self.enable_speculative = enable_speculative
        self.speculative_draft_model_path = speculative_draft_model_path
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        self.speculative_cuda_graph = bool(speculative_cuda_graph)
        self.speculative_draft_cuda_graph = bool(speculative_draft_cuda_graph)
        self.speculative_share_encoder = bool(speculative_share_encoder)
        self.enable_pre_lm_encoder = bool(enable_pre_lm_encoder)
        self.pre_lm_cache_max_entries = int(pre_lm_cache_max_entries)
        # None: derive the byte budget from the entry count (see encoder_service).
        self.pre_lm_cache_size_bytes = (
            None if pre_lm_cache_size_bytes is None else int(pre_lm_cache_size_bytes)
        )
        self.pre_lm_max_batch_size = int(pre_lm_max_batch_size)
        self.pre_lm_max_batch_wait_ms = int(pre_lm_max_batch_wait_ms)
        self.pre_lm_cache_pin_host_memory = bool(pre_lm_cache_pin_host_memory)
        self.processor: Any = None
        self.tokenizer: Any = None
        self.generation_config: Any = None
        self.encoder_token_count = 0
        self.context_length = 0
        self.decoder_context_len = 0
        self._target_hf_config: Any = None
        self.audio_encoder_service: Any | None = None

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        from transformers import AutoConfig, AutoProcessor, GenerationConfig

        from sglang_omni.models.whisper_asr.request_builders import (
            MAX_PREV_CONTEXT_TOKENS,
        )

        self._target_hf_config = AutoConfig.from_pretrained(checkpoint_dir)
        self.processor = AutoProcessor.from_pretrained(checkpoint_dir)
        self.tokenizer = self.processor.tokenizer
        self.generation_config = GenerationConfig.from_pretrained(checkpoint_dir)
        self.encoder_token_count = int(
            self.processor.feature_extractor.nb_max_frames // 2
        )
        self.context_length = (
            self.encoder_token_count + MAX_PREV_CONTEXT_TOKENS + self.max_new_tokens + 8
        )
        self.decoder_context_len = int(
            self._target_hf_config.max_target_positions or 448
        )

    def speculative_model_support(self) -> WhisperSpeculativeSupport:
        return WhisperSpeculativeSupport(
            share_encoder=self.speculative_share_encoder,
            decoder_context_len=self.decoder_context_len,
        )

    def setup_model_resources(
        self,
        model: Any,
        server_args: Any,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        if not self.enable_encoder_cuda_graph or not generation_cuda_graph_enabled:
            return

        max_prefill_tokens = int(server_args.max_prefill_tokens)
        max_running_requests = int(server_args.max_running_requests)
        resolved_buckets = _resolve_encoder_graph_buckets(
            self.encoder_graph_batch_buckets,
            enable_pre_lm_encoder=self.enable_pre_lm_encoder,
            pre_lm_max_batch_size=self.pre_lm_max_batch_size,
            max_prefill_tokens=max_prefill_tokens,
            encoder_token_count=self.encoder_token_count,
            max_running_requests=(
                max_running_requests
                if self._using_default_encoder_graph_buckets
                else None
            ),
        )
        logger.info(
            "Resolved Whisper encoder CUDA graph buckets configured=%s "
            "reachable=%s pre_lm=%s max_prefill_tokens=%d "
            "encoder_token_count=%d max_running_requests=%d",
            self.encoder_graph_batch_buckets,
            resolved_buckets,
            self.enable_pre_lm_encoder,
            max_prefill_tokens,
            self.encoder_token_count,
            max_running_requests,
        )
        model.init_encoder_graphs(
            resolved_buckets,
            int(self.processor.feature_extractor.nb_max_frames),
        )

    def setup_runtime_resources(self, model: Any, server_args: Any) -> None:
        del server_args
        if not self.enable_pre_lm_encoder:
            return

        self.audio_encoder_service = WhisperPreLMEncoderService(
            model,
            cache_namespace=build_cache_namespace(
                model,
                model_path=self.checkpoint_dir,
                feature_extractor=self.processor.feature_extractor,
            ),
            encoder_token_count=self.encoder_token_count,
            cache_max_entries=self.pre_lm_cache_max_entries,
            cache_max_bytes=self.pre_lm_cache_size_bytes,
            max_batch_size=self.pre_lm_max_batch_size,
            max_batch_wait_ms=self.pre_lm_max_batch_wait_ms,
            pin_host_memory=self.pre_lm_cache_pin_host_memory,
        )
        service = self.audio_encoder_service
        logger.info(
            "Whisper pre-LM encoder enabled (max_batch=%d, cache capacity=%d "
            "entries x %.2f MB = %.2f GB, configured entries=%d, byte cap=%s, "
            "pinned host memory=%s)",
            self.pre_lm_max_batch_size,
            service.cache_capacity_entries,
            service.entry_bytes / 1e6,
            service.cache_max_bytes / 1e9,
            self.pre_lm_cache_max_entries,
            self.pre_lm_cache_size_bytes,
            service.pin_host_memory,
        )

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        if (
            overrides.get("speculative_algorithm") is not None
            and find_spec("sglang.srt.speculative.encoder_decoder_utils") is None
        ):
            raise RuntimeError(
                "speculative decoding for encoder-decoder models requires "
                "the SGLang 0.5.18 encoder-decoder speculative patch"
            )
        if overrides.get("speculative_algorithm") is not None:
            self.enable_speculative = True
            self.enable_async_decode = False
            self.speculative_num_draft_tokens = overrides[
                "speculative_num_draft_tokens"
            ]
            from transformers import AutoConfig

            if self._target_hf_config is None:
                raise RuntimeError(
                    "Whisper target config must be loaded before speculative validation"
                )
            draft_config = AutoConfig.from_pretrained(
                overrides["speculative_draft_model_path"]
            )
            self.speculative_model_support().validate_speculative_draft(
                self._target_hf_config,
                draft_config,
                overrides["speculative_algorithm"],
                overrides["speculative_num_draft_tokens"],
            )
        if int(overrides.get("chunked_prefill_size") or 0) > 0:
            raise ValueError(
                "Whisper ASR requires chunked_prefill_size=0 because its encoder "
                "prefix must be admitted atomically"
            )
        overrides["chunked_prefill_size"] = 0
        # Note (Akazaakane): Timestamped Whisper requests install an internal
        # per-request processor; this flag permits SGLang to execute it.
        overrides["enable_custom_logit_processor"] = True

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        return {
            "max_running_requests": self.max_running_requests,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": True,
            "mem_fraction_static": self.mem_fraction_static,
            "max_prefill_tokens": 6144,
            "chunked_prefill_size": 0,
            "sampling_backend": "pytorch",
            "dtype": dtype,
        }

    def infra_kwargs(self) -> dict[str, Any]:
        return {"target_tokenizer": self.tokenizer}

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        from sglang_omni.models.whisper_asr.request_builders import (
            make_whisper_scheduler_adapters,
        )

        return make_whisper_scheduler_adapters(
            processor=self.processor,
            tokenizer=self.tokenizer,
            generation_config=self.generation_config,
            encoder_token_count=self.encoder_token_count,
            max_new_tokens=self.max_new_tokens,
            decoder_context_len=(
                self.decoder_context_len
                - self.speculative_model_support().position_reserve(
                    self.speculative_num_draft_tokens
                )
                if self.enable_speculative
                else self.decoder_context_len
            ),
            audio_encoder_service=self.audio_encoder_service,
            enable_speculative=self.enable_speculative,
        )

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {
            "enable_async_decode": self.enable_async_decode,
            "async_decode_min_batch_size": self.async_decode_min_batch_size,
            "request_build_max_workers": self.request_build_max_workers,
            "request_build_max_pending": self.request_build_max_pending,
            "prefill_coalesce_requests": self.prefill_coalesce_requests,
            "prefill_coalesce_wait_ms": self.prefill_coalesce_wait_ms,
            "prefill_coalesce_when_idle": self.prefill_coalesce_when_idle,
            "prefill_coalesce_requires_pending_builds": (
                self.prefill_coalesce_requires_pending_builds
            ),
            "prefill_coalesce_after_builds_during_decode": (
                self.prefill_coalesce_after_builds_during_decode
            ),
        }

    def extra_scheduler_callbacks(self) -> dict[str, Any]:
        if self.audio_encoder_service is None:
            return {}
        return {"shutdown_callback": self.audio_encoder_service.close}

    def cleanup_build_failure(self) -> None:
        if self.audio_encoder_service is not None:
            self.audio_encoder_service.close()
            self.audio_encoder_service = None
