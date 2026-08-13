# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Whisper ASR."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.whisper_asr"


class WhisperASRPipelineConfig(PipelineConfig):
    """Single-stage batched ASR pipeline for Whisper checkpoints."""

    architecture: ClassVar[str] = "WhisperForConditionalGeneration"

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"asr": "asr"}

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "asr"}

    def supports_audio_translation(self) -> bool:
        """Multilingual non-turbo Whisper checkpoints translate speech to
        English; `*.en` and turbo checkpoints do not, which is documented
        rather than gated, matching how sibling engines handle it."""
        return True

    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        StageConfig(
            name="asr",
            process="asr",
            factory=f"{_PKG}.stages.create_sglang_whisper_asr_executor",
            factory_args={
                "device": "cuda:0",
                "enable_encoder_cuda_graph": True,
                "request_build_max_workers": 2,
                "request_build_max_pending": 16,
                "prefill_coalesce_requests": 2,
                "prefill_coalesce_wait_ms": 6.0,
                "prefill_coalesce_when_idle": True,
                "prefill_coalesce_requires_pending_builds": True,
                "prefill_coalesce_after_builds_during_decode": False,
            },
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = WhisperASRPipelineConfig
