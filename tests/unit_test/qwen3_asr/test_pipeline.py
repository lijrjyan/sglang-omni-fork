# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import pytest
import typer

import sglang_omni.models.qwen3_asr.stages as qwen3_asr_stages
from sglang_omni.cli.serve import apply_asr_chunking_cli_overrides
from sglang_omni.models.qwen3_asr.chunking import (
    QWEN3_ASR_AUTO_CHUNK_DEFAULT,
    QWEN3_ASR_CHUNK_WINDOW_SECONDS,
)
from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig
from sglang_omni.models.qwen3_asr.stages import create_sglang_qwen3_asr_executor
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.serve.launcher import _transcription_chunking_kwargs
from tests.unit_test.fakes import FakeServerArgs


def test_qwen3_asr_config_uses_batched_stage_with_32_running_requests() -> None:
    config = Qwen3ASRPipelineConfig(model_path="Qwen/Qwen3-ASR-1.7B")

    assert config.entry_stage == "asr"
    assert [stage.name for stage in config.stages] == ["asr"]
    assert config.terminal_stages == ["asr"]
    assert config.gpu_placement == {"asr": 0}
    assert config.stages[0].factory.endswith("create_sglang_qwen3_asr_executor")
    assert config.stages[0].factory_args["device"] == "cuda:0"
    assert config.stages[0].factory_args["max_running_requests"] == 32
    assert config.stages[0].factory_args["request_build_max_workers"] == 2
    assert config.stages[0].factory_args["request_build_max_pending"] == 16
    assert config.stages[0].factory_args["asr_auto_chunk"] is True
    assert config.stages[0].factory_args["asr_chunk_max_seconds"] == 30.0
    assert "request_build_max_backlog" not in config.stages[0].factory_args
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("Qwen3ASRForConditionalGeneration")
        is Qwen3ASRPipelineConfig
    )


def test_qwen3_asr_stage_default_allows_32_running_requests() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["max_running_requests"].default == 32
    assert signature.parameters["request_build_max_workers"].default == 2
    assert signature.parameters["request_build_max_pending"].default == 16
    assert "request_build_max_backlog" not in signature.parameters


def test_qwen3_asr_stage_default_uses_auto_static_kv_budget() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["mem_fraction_static"].default is None


def test_qwen3_asr_stage_default_disables_multimodal_embedding_cache() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["mm_embedding_cache_size_bytes"].default == 0


def test_qwen3_asr_stage_default_disables_torch_compile() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["enable_torch_compile"].default is False


def test_qwen3_asr_stage_default_enables_async_decode() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["enable_async_decode"].default is True
    assert signature.parameters["async_decode_min_batch_size"].default == 2


def test_qwen3_asr_stage_owns_auto_chunk_defaults() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["asr_auto_chunk"].default is (
        QWEN3_ASR_AUTO_CHUNK_DEFAULT
    )
    assert signature.parameters["asr_chunk_max_seconds"].default == (
        QWEN3_ASR_CHUNK_WINDOW_SECONDS
    )


def test_qwen3_asr_cli_chunking_overrides_take_priority() -> None:
    config = Qwen3ASRPipelineConfig(model_path="Qwen/Qwen3-ASR-1.7B")

    result = apply_asr_chunking_cli_overrides(
        config,
        asr_auto_chunk=False,
        asr_chunk_max_seconds=45.0,
    )

    assert result is config
    assert config.stages[0].factory_args["asr_auto_chunk"] is False
    assert config.stages[0].factory_args["asr_chunk_max_seconds"] == 45.0


def test_qwen3_asr_threads_explicit_cuda_graph_bs(monkeypatch) -> None:
    build_kwargs: dict[str, object] = {}

    monkeypatch.setattr(
        qwen3_asr_stages.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        qwen3_asr_stages.AutoFeatureExtractor,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(nb_max_frames=3000),
    )
    monkeypatch.setattr(
        qwen3_asr_stages,
        "get_visible_gpu_sm_version",
        lambda gpu_id: None,
    )
    monkeypatch.setattr(qwen3_asr_stages, "init_mm_embedding_cache", lambda size: None)
    monkeypatch.setattr(
        qwen3_asr_stages,
        "make_qwen3_asr_scheduler_adapters",
        lambda **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        qwen3_asr_stages,
        "ModelRunner",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        qwen3_asr_stages,
        "SGLangOutputProcessor",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        qwen3_asr_stages,
        "OmniScheduler",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    def _fake_server_args_builder(model_path, context_length, **overrides):
        build_kwargs.update(overrides)
        server_args = FakeServerArgs(**overrides)
        server_args.cuda_graph_config = SimpleNamespace(
            decode=SimpleNamespace(
                max_bs=overrides["cuda_graph_max_bs"],
                bs=overrides["cuda_graph_bs"],
            )
        )
        return server_args

    def _fake_create_infrastructure(server_args, gpu_id, **kwargs):
        model_worker = SimpleNamespace(model_runner=SimpleNamespace(model=object()))
        return False, (
            model_worker,
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
        )

    monkeypatch.setattr(
        qwen3_asr_stages,
        "build_sglang_server_args",
        _fake_server_args_builder,
    )
    monkeypatch.setattr(
        qwen3_asr_stages,
        "create_sglang_infrastructure_defer_cuda_graph",
        _fake_create_infrastructure,
    )

    scheduler = qwen3_asr_stages.create_sglang_qwen3_asr_executor(
        "dummy",
        enable_async_decode=False,
        async_decode_min_batch_size=4,
    )

    assert build_kwargs["cuda_graph_max_bs"] == 32
    assert build_kwargs["cuda_graph_bs"] == [1, 2, 4, 8, 12, 16, 24, 32]
    assert scheduler.enable_async_decode is False
    assert scheduler.async_decode_min_batch_size == 4


def test_qwen3_asr_launcher_chunking_respects_runtime_overrides() -> None:
    """The endpoint must see the same resolved policy as the worker factory."""
    config = Qwen3ASRPipelineConfig(model_path="Qwen/Qwen3-ASR-1.7B")
    config.stages[0].factory_args.update(
        {
            "asr_auto_chunk": False,
            "asr_chunk_max_seconds": 45.0,
        }
    )
    config.runtime_overrides = {
        "asr": {
            "asr_auto_chunk": True,
            "asr_chunk_max_seconds": 30.0,
        }
    }

    assert _transcription_chunking_kwargs(config) == {
        "asr_auto_chunk": True,
        "asr_chunk_max_seconds": 30.0,
    }


@pytest.mark.parametrize("invalid_window", [-1.0, math.nan, math.inf])
def test_qwen3_asr_cli_chunking_validation_uses_model_validator(
    invalid_window: float,
) -> None:
    config = Qwen3ASRPipelineConfig(model_path="Qwen/Qwen3-ASR-1.7B")

    with pytest.raises(typer.BadParameter, match="finite positive"):
        apply_asr_chunking_cli_overrides(
            config,
            asr_auto_chunk=None,
            asr_chunk_max_seconds=invalid_window,
        )
