# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Fun-CosyVoice3."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.platforms import current_platform

_PKG = "sglang_omni.models.fun_cosyvoice3"


class FunCosyVoice3FactoryArgs(FactoryArgs):
    """Fun-CosyVoice3 tts_engine constructor knobs, typed like the shared ones."""

    enable_speculative: bool | None = None
    speculative_algorithm: str | None = None
    speculative_draft_model_path: str | None = None
    speculative_num_steps: int | None = Field(default=None, ge=0)
    speculative_num_draft_tokens: int | None = Field(default=None, ge=1)
    speculative_cuda_graph: bool | None = None
    speculative_draft_cuda_graph: bool | None = None
    speculative_use_rejection_sampling: bool | None = None


class FunCosyVoice3EngineStageConfig(EngineStageConfig):
    factory: FunCosyVoice3FactoryArgs = Field(default_factory=FunCosyVoice3FactoryArgs)


class FunCosyVoice3PipelineConfig(PipelineConfig):
    """3-stage Fun-CosyVoice3 pipeline: preprocessing -> tts_engine -> vocoder."""

    architecture: ClassVar[str] = "FunCosyVoice3SGLangModel"
    # note (PoTaTo): This checkpoint has no built-in speaker presets, so public requests need
    # one reference clip for speaker conditioning.
    required_speech_reference_count: ClassVar[int | None] = 1
    speech_reference_text_excludes_instructions: ClassVar[bool] = True

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts_engine": FunCosyVoice3EngineStageConfig,
    }

    @classmethod
    def process_local_edges(cls) -> frozenset[tuple[str, str]]:
        return frozenset({("preprocessing", "tts_engine")})

    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            next="tts_engine",
        ),
        FunCosyVoice3EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory=FunCosyVoice3FactoryArgs(
                device=current_platform.device_type, dtype="bfloat16"
            ),
            gpu=0,
            next="vocoder",
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            factory=FactoryArgs(
                dtype="bfloat16",
                flow_batch_bucket_frames=50,
                flow_batch_admission_frames=2000,
                # Opt-in; off by default (one-time startup compile cost).
                enable_dit_torch_compile=False,
            ),
            gpu=0,
            terminal=True,
        ),
    ]


EntryClass = FunCosyVoice3PipelineConfig
