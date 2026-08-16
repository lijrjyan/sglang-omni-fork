# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import sys
from types import SimpleNamespace

import pytest

import sglang_omni.model_runner.base as model_runner_base
import sglang_omni.models.whisper_asr.stages as whisper_asr_stages
import sglang_omni.scheduling.bootstrap as bootstrap
import sglang_omni.scheduling.omni_scheduler as omni_scheduler
import sglang_omni.scheduling.sglang_backend as sglang_backend
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.models.whisper_asr import request_builders as whisper_request_builders
from sglang_omni.models.whisper_asr.config import WhisperASRPipelineConfig
from tests.unit_test.fakes import FakeServerArgs


def test_whisper_encoder_cuda_graph_is_opt_in() -> None:
    signature = inspect.signature(whisper_asr_stages.create_sglang_whisper_asr_executor)

    assert signature.parameters["enable_encoder_cuda_graph"].default is False
    assert signature.parameters["encoder_graph_batch_buckets"].default is None
    assert signature.parameters["enable_encoder_torch_compile"].default is False
    assert signature.parameters["encoder_torch_compile_mode"].default is None
    assert signature.parameters["quantization_scope"].default == "all"
    assert signature.parameters["request_build_max_workers"].default == 2
    assert signature.parameters["request_build_max_pending"].default == 16
    assert signature.parameters["prefill_coalesce_requests"].default == 2
    assert signature.parameters["prefill_coalesce_wait_ms"].default == 6.0
    assert signature.parameters["prefill_coalesce_when_idle"].default is True
    assert (
        signature.parameters["prefill_coalesce_requires_pending_builds"].default is True
    )
    assert (
        signature.parameters["prefill_coalesce_after_builds_during_decode"].default
        is False
    )
    assert signature.parameters["enable_speculative"].default is False
    assert signature.parameters["speculative_draft_model_path"].default is None
    assert signature.parameters["speculative_num_steps"].default == 3
    assert signature.parameters["speculative_num_draft_tokens"].default == 4


def test_whisper_stage_forwards_first_class_speculative_args(monkeypatch) -> None:
    from sglang_omni.models.whisper_asr import engine_builder as whisper_builder

    seen: dict[str, object] = {}

    class FakeBuilder:
        def __init__(self, **kwargs) -> None:
            seen["builder"] = kwargs

        def build(self, model_path, **kwargs):
            seen["build"] = {"model_path": model_path, **kwargs}
            return "executor"

    monkeypatch.setattr(whisper_builder, "WhisperASREngineBuilder", FakeBuilder)

    result = whisper_asr_stages.create_sglang_whisper_asr_executor(
        "/models/whisper-large-v3",
        enable_speculative=True,
        speculative_draft_model_path="/models/distil-whisper-large-v3",
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        server_args_overrides={"max_running_requests": 2},
    )

    assert result == "executor"
    builder_args = seen["builder"]
    assert builder_args["enable_speculative"] is True
    assert (
        builder_args["speculative_draft_model_path"]
        == "/models/distil-whisper-large-v3"
    )
    assert builder_args["speculative_num_steps"] == 3
    assert builder_args["speculative_num_draft_tokens"] == 4
    assert seen["build"]["server_args_overrides"] == {"max_running_requests": 2}


def test_whisper_speculative_requires_p3_unless_probe_env_is_set(monkeypatch) -> None:
    from transformers import AutoConfig

    from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder

    builder = WhisperASREngineBuilder(
        max_running_requests=4,
        max_new_tokens=32,
        mem_fraction_static=0.2,
        enable_speculative=True,
        speculative_draft_model_path="/models/distil-whisper-large-v3",
    )
    overrides = {
        "speculative_algorithm": "STANDALONE",
        "speculative_draft_model_path": "/models/distil-whisper-large-v3",
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
    }
    config = SimpleNamespace(
        architectures=["WhisperForConditionalGeneration"],
        d_model=1280,
        vocab_size=51866,
        num_mel_bins=128,
        max_source_positions=1500,
    )
    builder._target_hf_config = config
    builder.decoder_context_len = 448
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda _path: config)

    monkeypatch.delenv("SGLANG_OMNI_SPEC_ALLOW_ENCDEC", raising=False)
    with pytest.raises(RuntimeError, match=r"requires .* \(P3\)"):
        builder.adjust_overrides(dict(overrides))

    monkeypatch.setenv("SGLANG_OMNI_SPEC_ALLOW_ENCDEC", "1")
    allowed = dict(overrides)
    builder.adjust_overrides(allowed)
    assert allowed["speculative_algorithm"] == "STANDALONE"


def test_whisper_speculative_config_requires_compatible_draft() -> None:
    from sglang_omni.models.whisper_asr.engine_builder import (
        _validate_whisper_speculative_configs,
    )

    target = SimpleNamespace(
        architectures=["WhisperForConditionalGeneration"],
        d_model=1280,
        vocab_size=51866,
        num_mel_bins=128,
        max_source_positions=1500,
    )
    draft = SimpleNamespace(**vars(target))

    _validate_whisper_speculative_configs(
        target,
        draft,
        num_draft_tokens=4,
        decoder_context_len=448,
    )

    draft.architectures = ["LlamaForCausalLM"]
    with pytest.raises(ValueError, match="Whisper architecture"):
        _validate_whisper_speculative_configs(
            target,
            draft,
            num_draft_tokens=4,
            decoder_context_len=448,
        )


@pytest.mark.parametrize(
    ("field", "draft_value"),
    [
        ("d_model", 1024),
        ("vocab_size", 51865),
        ("num_mel_bins", 80),
        ("max_source_positions", 1024),
    ],
)
def test_whisper_speculative_config_rejects_target_draft_mismatch(
    field: str,
    draft_value: int,
) -> None:
    from sglang_omni.models.whisper_asr.engine_builder import (
        _validate_whisper_speculative_configs,
    )

    target = SimpleNamespace(
        architectures=["WhisperForConditionalGeneration"],
        d_model=1280,
        vocab_size=51866,
        num_mel_bins=128,
        max_source_positions=1500,
    )
    draft = SimpleNamespace(**vars(target))
    setattr(draft, field, draft_value)

    with pytest.raises(ValueError, match=field):
        _validate_whisper_speculative_configs(
            target,
            draft,
            num_draft_tokens=4,
            decoder_context_len=448,
        )


def test_whisper_speculative_config_rejects_draft_window_over_budget() -> None:
    from sglang_omni.models.whisper_asr.engine_builder import (
        _validate_whisper_speculative_configs,
    )

    config = SimpleNamespace(
        architectures=["WhisperForConditionalGeneration"],
        d_model=1280,
        vocab_size=51866,
        num_mel_bins=128,
        max_source_positions=1500,
    )

    with pytest.raises(ValueError, match="decoder budget"):
        _validate_whisper_speculative_configs(
            config,
            config,
            num_draft_tokens=449,
            decoder_context_len=448,
        )


def test_whisper_encoder_cuda_graph_setup_is_ordered_after_generation_graphs() -> None:
    from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder

    calls: list[tuple[list[int], int]] = []
    builder = WhisperASREngineBuilder(
        max_running_requests=4,
        max_new_tokens=32,
        mem_fraction_static=0.2,
        enable_encoder_cuda_graph=True,
    )
    builder.processor = SimpleNamespace(
        feature_extractor=SimpleNamespace(nb_max_frames=3000)
    )
    builder.encoder_token_count = 1500
    assert builder.encoder_graph_batch_buckets == (1, 2, 4, 8, 12, 16)
    model = SimpleNamespace(
        init_encoder_graphs=lambda buckets, feature_len: calls.append(
            (list(buckets), feature_len)
        )
    )

    builder.setup_model_resources(
        model,
        server_args=SimpleNamespace(max_prefill_tokens=4096),
        generation_cuda_graph_enabled=True,
    )
    assert calls == [([1, 2], 3000)]

    builder.setup_model_resources(
        model,
        server_args=SimpleNamespace(max_prefill_tokens=4096),
        generation_cuda_graph_enabled=False,
    )
    assert calls == [([1, 2], 3000)]


def test_whisper_encoder_compile_runs_before_graph_capture() -> None:
    from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder

    calls: list[tuple[str, object]] = []
    builder = WhisperASREngineBuilder(
        max_running_requests=4,
        max_new_tokens=32,
        mem_fraction_static=0.2,
        enable_encoder_cuda_graph=True,
        enable_encoder_torch_compile=True,
        encoder_torch_compile_mode="max-autotune-no-cudagraphs",
    )
    builder.processor = SimpleNamespace(
        feature_extractor=SimpleNamespace(nb_max_frames=3000)
    )
    builder.encoder_token_count = 1500
    model = SimpleNamespace(
        compile_encoder=lambda feature_len, mode=None: calls.append(
            ("compile", (feature_len, mode))
        ),
        init_encoder_graphs=lambda buckets, feature_len: calls.append(
            ("graphs", list(buckets))
        ),
    )

    builder.setup_model_resources(
        model,
        server_args=SimpleNamespace(max_prefill_tokens=4096),
        generation_cuda_graph_enabled=True,
    )
    assert calls == [
        ("compile", (3000, "max-autotune-no-cudagraphs")),
        ("graphs", [1, 2]),
    ]

    calls.clear()
    builder.setup_model_resources(
        model,
        server_args=SimpleNamespace(max_prefill_tokens=4096),
        generation_cuda_graph_enabled=False,
    )
    assert calls == [("compile", (3000, "max-autotune-no-cudagraphs"))]


def test_whisper_encoder_compile_is_opt_in_at_builder() -> None:
    from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder

    builder = WhisperASREngineBuilder(
        max_running_requests=4,
        max_new_tokens=32,
        mem_fraction_static=0.2,
    )
    builder.processor = SimpleNamespace(
        feature_extractor=SimpleNamespace(nb_max_frames=3000)
    )
    builder.encoder_token_count = 1500
    model = SimpleNamespace()  # no compile_encoder / init_encoder_graphs

    builder.setup_model_resources(
        model,
        server_args=SimpleNamespace(max_prefill_tokens=4096),
        generation_cuda_graph_enabled=True,
    )


def test_whisper_encoder_cuda_graph_buckets_follow_final_prefill_budget() -> None:
    from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder

    calls: list[list[int]] = []
    builder = WhisperASREngineBuilder(
        max_running_requests=16,
        max_new_tokens=32,
        mem_fraction_static=0.2,
        enable_encoder_cuda_graph=True,
        encoder_graph_batch_buckets=[8, 1, 4, 4, 16],
    )
    builder.processor = SimpleNamespace(
        feature_extractor=SimpleNamespace(nb_max_frames=3000)
    )
    builder.encoder_token_count = 1500
    model = SimpleNamespace(
        init_encoder_graphs=lambda buckets, feature_len: calls.append(list(buckets))
    )

    builder.setup_model_resources(
        model,
        server_args=SimpleNamespace(max_prefill_tokens=8192),
        generation_cuda_graph_enabled=True,
    )

    assert calls == [[1, 4]]


def test_whisper_disables_chunked_prefill_for_atomic_encoder_prefix() -> None:
    from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder

    builder = WhisperASREngineBuilder(
        max_running_requests=4,
        max_new_tokens=32,
        mem_fraction_static=0.2,
    )
    defaults = builder.generation_defaults(dtype="float16")

    assert defaults["max_prefill_tokens"] == 4096
    assert defaults["chunked_prefill_size"] == 0

    overrides = {"chunked_prefill_size": 0}
    builder.adjust_overrides(overrides)
    assert overrides["chunked_prefill_size"] == 0

    with pytest.raises(ValueError, match="encoder prefix must be admitted atomically"):
        builder.adjust_overrides({"chunked_prefill_size": 4096})


def test_whisper_prefill_coalescing_defaults_are_forwarded() -> None:
    from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder

    builder = WhisperASREngineBuilder(
        max_running_requests=16,
        max_new_tokens=32,
        mem_fraction_static=0.2,
    )

    assert builder.extra_scheduler_kwargs() == {
        "request_build_max_workers": 2,
        "request_build_max_pending": 16,
        "prefill_coalesce_requests": 2,
        "prefill_coalesce_wait_ms": 6.0,
        "prefill_coalesce_when_idle": True,
        "prefill_coalesce_requires_pending_builds": True,
        "prefill_coalesce_after_builds_during_decode": False,
    }


def test_whisper_asr_config_uses_single_batched_stage() -> None:
    config = WhisperASRPipelineConfig(model_path="openai/whisper-large-v3")

    assert config.entry_stage == "asr"
    assert [stage.name for stage in config.stages] == ["asr"]
    assert config.terminal_stages == ["asr"]
    assert config.gpu_placement == {"asr": 0}
    assert config.stages[0].factory.endswith("create_sglang_whisper_asr_executor")
    assert config.stages[0].factory_args["device"] == "cuda:0"
    assert config.stages[0].factory_args["enable_encoder_cuda_graph"] is True
    assert config.stages[0].factory_args["enable_encoder_torch_compile"] is True
    assert config.stages[0].factory_args["request_build_max_workers"] == 2
    assert config.stages[0].factory_args["request_build_max_pending"] == 16
    assert config.stages[0].factory_args["prefill_coalesce_requests"] == 2
    assert config.stages[0].factory_args["prefill_coalesce_wait_ms"] == 6.0
    assert config.stages[0].factory_args["prefill_coalesce_when_idle"] is True
    assert (
        config.stages[0].factory_args["prefill_coalesce_requires_pending_builds"]
        is True
    )
    assert (
        config.stages[0].factory_args["prefill_coalesce_after_builds_during_decode"]
        is False
    )
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("WhisperForConditionalGeneration")
        is WhisperASRPipelineConfig
    )


def test_whisper_asr_threads_explicit_cuda_graph_bs(monkeypatch) -> None:
    build_kwargs: dict[str, object] = {}
    fake_processor = SimpleNamespace(
        tokenizer=object(),
        feature_extractor=SimpleNamespace(nb_max_frames=3000),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: SimpleNamespace(
                    max_target_positions=448
                )
            ),
            AutoProcessor=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: fake_processor
            ),
            GenerationConfig=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: object()
            ),
        ),
    )
    monkeypatch.setattr(
        whisper_request_builders,
        "make_whisper_scheduler_adapters",
        lambda **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        model_runner_base,
        "ModelRunner",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        sglang_backend,
        "SGLangOutputProcessor",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        omni_scheduler,
        "OmniScheduler",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    def _fake_server_args_builder(model_path, context_length, **overrides):
        build_kwargs["context_length"] = context_length
        build_kwargs.update(overrides)
        server_args = FakeServerArgs(**overrides)
        server_args.cuda_graph_config = SimpleNamespace(
            decode=SimpleNamespace(
                max_bs=overrides["cuda_graph_max_bs"],
                bs=overrides["cuda_graph_bs"],
            ),
            prefill=SimpleNamespace(backend="disabled", bs=None, max_bs=None),
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
        sglang_backend,
        "build_sglang_server_args",
        _fake_server_args_builder,
    )
    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure_defer_cuda_graph",
        _fake_create_infrastructure,
    )

    whisper_asr_stages.create_sglang_whisper_asr_executor("dummy")

    assert build_kwargs["cuda_graph_max_bs"] == 16
    assert build_kwargs["cuda_graph_bs"] == [1, 2, 4, 8, 12, 16]
    # note (jiannan-17): context_length = encoder_token_count + max_prev_tokens + max_new_tokens + 8
    assert build_kwargs["context_length"] == 1500 + 224 + 256 + 8
    assert build_kwargs["chunked_prefill_size"] == 0
