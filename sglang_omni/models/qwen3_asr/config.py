# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Qwen3-ASR."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

from .chunking import (
    QWEN3_ASR_AUTO_CHUNK_DEFAULT,
    QWEN3_ASR_CHUNK_OVERLAP_SECONDS,
    QWEN3_ASR_CHUNK_WINDOW_SECONDS,
)

_PKG = "sglang_omni.models.qwen3_asr"


class Qwen3ASRPipelineConfig(PipelineConfig):
    """Single-stage batched ASR pipeline for Qwen3-ASR checkpoints."""

    architecture: ClassVar[str] = "Qwen3ASRForConditionalGeneration"

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"asr": "asr"}

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "asr"}

    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        StageConfig(
            name="asr",
            process="asr",
            factory=f"{_PKG}.stages.create_sglang_qwen3_asr_executor",
            factory_args={
                "device": "cuda:0",
                "max_running_requests": 32,
                "max_new_tokens": 128,
                "request_build_max_workers": 2,
                "request_build_max_pending": 16,
                "asr_auto_chunk": QWEN3_ASR_AUTO_CHUNK_DEFAULT,
                "asr_chunk_max_seconds": QWEN3_ASR_CHUNK_WINDOW_SECONDS,
                "asr_chunk_overlap_seconds": QWEN3_ASR_CHUNK_OVERLAP_SECONDS,
            },
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = Qwen3ASRPipelineConfig
