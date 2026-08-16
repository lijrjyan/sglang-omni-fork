# SPDX-License-Identifier: Apache-2.0
"""SGLang bootstrap helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.model_runner import _hidden_capture as hidden_capture_module
from sglang_omni.model_runner import model_worker as model_worker_module
from sglang_omni.scheduling import bootstrap, sglang_backend
from tests.unit_test.fakes import FakeServerArgs


def test_runtime_configuration_reports_global_backend_for_each_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bootstrap,
        "get_visible_gpu_sm_version",
        lambda _gpu_id: 89,
        raising=False,
    )
    server_args = SimpleNamespace(
        attention_backend="flashinfer",
        decode_attention_backend=None,
        prefill_attention_backend=None,
        sampling_backend="pytorch",
        get_attention_backends=lambda: ("flashinfer", "flashinfer"),
    )

    description = bootstrap._describe_sglang_runtime_configuration(
        server_args,
        gpu_id=0,
    )

    assert description == (
        "SGLang runtime configuration: gpu_id=0, sm=89, architecture=ada, "
        "attention_backend=flashinfer, decode_attention_backend=flashinfer, "
        "prefill_attention_backend=flashinfer, sampling_backend=pytorch"
    )


def test_runtime_configuration_reports_explicit_phase_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bootstrap,
        "get_visible_gpu_sm_version",
        lambda _gpu_id: 90,
        raising=False,
    )
    server_args = SimpleNamespace(
        attention_backend="flashinfer",
        decode_attention_backend="triton",
        prefill_attention_backend="fa3",
        sampling_backend="pytorch",
        get_attention_backends=lambda: ("fa3", "triton"),
    )

    description = bootstrap._describe_sglang_runtime_configuration(
        server_args,
        gpu_id=1,
    )

    assert description == (
        "SGLang runtime configuration: gpu_id=1, sm=90, architecture=hopper, "
        "attention_backend=flashinfer, decode_attention_backend=triton, "
        "prefill_attention_backend=fa3, sampling_backend=pytorch"
    )


def test_create_sglang_infrastructure_runs_0515_initialization_phases(
    monkeypatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        bootstrap,
        "_describe_sglang_runtime_configuration",
        lambda _server_args, _gpu_id: events.append("runtime_configuration")
        or "runtime configuration",
    )

    class FakeRunner:
        model = object()

        def alloc_memory_pool(self) -> None:
            events.append("alloc_memory_pool")

        def init_attention_backends(self) -> None:
            events.append("init_attention_backends")

        def init_cuda_graphs(self) -> None:
            events.append("init_cuda_graphs")

    class FakeWorker:
        model_config = SimpleNamespace(is_multimodal=False)
        enable_prefill_input_embeds = False

        def __init__(self, **kwargs) -> None:
            del kwargs
            events.append("model_worker")
            self.model_runner = FakeRunner()

        def get_memory_pool(self):
            events.append("get_memory_pool")
            return "req_pool", "kv_pool"

    class FakePrefillManager:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def add_one_request(self, req) -> None:
            del req

    class FakeDecodeManager:
        def __init__(self, **kwargs) -> None:
            del kwargs

    monkeypatch.setattr(model_worker_module, "ModelWorker", FakeWorker)
    monkeypatch.setattr(sglang_backend, "PrefillManager", FakePrefillManager)
    monkeypatch.setattr(sglang_backend, "DecodeManager", FakeDecodeManager)
    monkeypatch.setattr(
        sglang_backend,
        "create_tree_cache",
        lambda *args: ("tree_cache", args),
    )

    server_args = SimpleNamespace(
        speculative_algorithm=None,
        attention_backend=None,
        decode_attention_backend=None,
        prefill_attention_backend=None,
        sampling_backend=None,
        page_size=1,
        disable_overlap_schedule=False,
        disable_cuda_graph=False,
        chunked_prefill_size=8,
        max_prefill_tokens=16,
    )
    infrastructure = bootstrap.create_sglang_infrastructure(server_args, 0)

    assert events == [
        "runtime_configuration",
        "model_worker",
        "alloc_memory_pool",
        "init_attention_backends",
        "init_cuda_graphs",
        "get_memory_pool",
    ]
    assert infrastructure[0].model_runner.model is FakeRunner.model


def test_speculative_bootstrap_constructs_workers_pools_and_backends_in_order(
    monkeypatch,
) -> None:
    events: list[str] = []

    class FakeRunner:
        model = object()
        model_config = SimpleNamespace(context_len=4096, vocab_size=151936)
        ps = "parallel-state"
        dist_port = 23456
        memory_pool_config = "target-pool-config"

        def alloc_memory_pool(self) -> None:
            events.append("target_pool")

        def init_attention_backends(self) -> None:
            events.append("target_backend")

        def init_cuda_graphs(self) -> None:
            events.append("target_eager_runner")

    class FakeWorker:
        model_config = FakeRunner.model_config
        enable_prefill_input_embeds = False

        def __init__(self, **kwargs) -> None:
            del kwargs
            events.append("target_worker")
            self.model_runner = FakeRunner()

        def get_memory_pool(self):
            events.append("target_get_pool")
            return "req_pool", "kv_allocator"

    class FakeTargetAdapter:
        def __init__(self, worker, *, tokenizer) -> None:
            assert isinstance(worker, FakeWorker)
            assert tokenizer == "target-tokenizer"
            events.append("target_adapter")

    class FakeDraftWorker:
        def __init__(self, **kwargs) -> None:
            assert kwargs["gpu_id"] == 0
            assert kwargs["ps"] == "parallel-state"
            assert kwargs["nccl_port"] == 23456
            assert isinstance(kwargs["target_worker"], FakeTargetAdapter)
            events.append("draft_worker")

        def alloc_memory_pool(self, **kwargs) -> None:
            assert kwargs == {
                "memory_pool_config": "target-pool-config",
                "req_to_token_pool": "req_pool",
                "token_to_kv_pool_allocator": "kv_allocator",
            }
            events.append("draft_pool")

        def init_attention_backends(self) -> None:
            events.append("draft_backend")

        def init_cuda_graphs(self) -> None:
            events.append("draft_eager_runner")

    fake_algorithm = SimpleNamespace(
        is_none=lambda: False,
        create_worker=lambda _server_args: FakeDraftWorker,
    )

    class FakePrefillManager:
        def __init__(self, **kwargs) -> None:
            assert kwargs["spec_algorithm"] is fake_algorithm

        def add_one_request(self, req) -> None:
            del req

    class FakeDecodeManager:
        def __init__(self, **kwargs) -> None:
            del kwargs

    monkeypatch.setattr(
        bootstrap,
        "_describe_sglang_runtime_configuration",
        lambda *_args: "runtime configuration",
    )
    monkeypatch.setattr(
        bootstrap,
        "_parse_speculative_algorithm",
        lambda _args: fake_algorithm,
    )
    monkeypatch.setattr(model_worker_module, "ModelWorker", FakeWorker)
    monkeypatch.setattr(
        model_worker_module,
        "SpeculativeTargetWorkerAdapter",
        FakeTargetAdapter,
    )
    monkeypatch.setattr(
        model_worker_module,
        "register_speculative_omni_models",
        lambda architecture: events.append(f"register_models:{architecture}"),
    )
    monkeypatch.setattr(sglang_backend, "PrefillManager", FakePrefillManager)
    monkeypatch.setattr(sglang_backend, "DecodeManager", FakeDecodeManager)
    monkeypatch.setattr(
        sglang_backend,
        "create_tree_cache",
        lambda *args: ("tree", args),
    )

    server_args = SimpleNamespace(
        speculative_algorithm="STANDALONE",
        speculative_draft_model_path="qwen3-asr-self-draft",
        attention_backend=None,
        sampling_backend=None,
        get_attention_backends=lambda: (None, None),
        page_size=1,
        disable_overlap_schedule=True,
        disable_cuda_graph=True,
        chunked_prefill_size=-1,
        max_prefill_tokens=4096,
    )

    infrastructure = bootstrap.create_sglang_infrastructure(
        server_args,
        0,
        target_tokenizer="target-tokenizer",
        model_arch_override="Qwen3ASRForConditionalGeneration",
    )

    assert events.index("draft_worker") < events.index("target_pool")
    assert "register_models:Qwen3ASRForConditionalGeneration" in events
    assert events.index("target_pool") < events.index("draft_pool")
    assert events.index("draft_pool") < events.index("target_backend")
    assert events.index("target_backend") < events.index("draft_backend")
    assert events.index("draft_backend") < events.index("target_eager_runner")
    assert events.index("target_eager_runner") < events.index("draft_eager_runner")
    assert server_args.disable_cuda_graph is True
    assert infrastructure[-2].__class__ is FakeTargetAdapter
    assert infrastructure[-1].__class__ is FakeDraftWorker


def test_cuda_graph_init_scopes_prefill_embedding_capture_flag() -> None:
    model_config = SimpleNamespace(is_multimodal=False)
    capture_values: list[bool] = []
    model_worker = SimpleNamespace(
        model_config=model_config,
        enable_prefill_input_embeds=True,
        model_runner=SimpleNamespace(
            init_cuda_graphs=lambda: capture_values.append(model_config.is_multimodal)
        ),
    )

    bootstrap.init_sglang_cuda_graphs(model_worker)

    assert capture_values == [True]
    assert model_config.is_multimodal is False


@pytest.mark.parametrize(
    ("server_args", "expected"),
    [
        (
            SimpleNamespace(
                chunked_prefill_size=8192,
                max_prefill_tokens=16384,
                context_length=32768,
                max_running_requests=64,
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(max_bs=128, bs=[1, 128]),
                    prefill=SimpleNamespace(max_bs=256, bs=[4, 256]),
                ),
            ),
            8192,
        ),
        (
            SimpleNamespace(
                chunked_prefill_size=-1,
                max_prefill_tokens=16384,
                context_length=8192,
                max_running_requests=64,
                cuda_graph_config=None,
            ),
            16384,
        ),
        (
            SimpleNamespace(
                chunked_prefill_size=512,
                max_prefill_tokens=16384,
                context_length=8192,
                max_running_requests=64,
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(max_bs=128, bs=[1, 128]),
                    prefill=SimpleNamespace(max_bs=1024, bs=[4, 1024]),
                ),
            ),
            1024,
        ),
    ],
)
def test_hidden_capture_max_tokens_covers_eager_and_graph_forwards(
    server_args: SimpleNamespace,
    expected: int,
) -> None:
    assert bootstrap._hidden_capture_max_tokens(server_args) == expected


def test_hidden_capture_max_tokens_covers_non_chunked_context_length() -> None:
    server_args = SimpleNamespace(
        chunked_prefill_size=-1,
        max_prefill_tokens=8192,
        context_length=32768,
        max_running_requests=64,
        cuda_graph_config=None,
    )

    assert bootstrap._hidden_capture_max_tokens(server_args) == 32768


def test_hidden_capture_max_tokens_rejects_missing_capacity_sources() -> None:
    server_args = SimpleNamespace(
        chunked_prefill_size=-1,
        max_prefill_tokens=None,
        max_running_requests=0,
        cuda_graph_config=None,
    )

    with pytest.raises(ValueError, match="hidden capture capacity"):
        bootstrap._hidden_capture_max_tokens(server_args)


def test_hidden_capture_is_installed_before_graph_initialization(monkeypatch) -> None:
    events: list[object] = []

    class FakeRunner:
        model = object()

        def alloc_memory_pool(self) -> None:
            events.append("alloc_memory_pool")

        def init_attention_backends(self) -> None:
            events.append("init_attention_backends")

        def init_cuda_graphs(self) -> None:
            events.append("init_cuda_graphs")

    class FakeWorker:
        model_config = SimpleNamespace(is_multimodal=False)
        enable_prefill_input_embeds = False

        def __init__(self, **kwargs) -> None:
            del kwargs
            self.model_runner = FakeRunner()
            events.append("model_worker")

        def get_memory_pool(self):
            events.append("get_memory_pool")
            return "req_pool", "kv_pool"

    class FakePrefillManager:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def add_one_request(self, req) -> None:
            del req

    class FakeDecodeManager:
        def __init__(self, **kwargs) -> None:
            del kwargs

    def fake_install(model, layers, *, max_tokens) -> None:
        events.append(("install_hidden_capture", model, layers, max_tokens))

    monkeypatch.setattr(
        bootstrap,
        "_describe_sglang_runtime_configuration",
        lambda *_args: "runtime configuration",
    )
    monkeypatch.setattr(model_worker_module, "ModelWorker", FakeWorker)
    monkeypatch.setattr(
        hidden_capture_module,
        "install_hidden_capture_hooks",
        fake_install,
    )
    monkeypatch.setattr(sglang_backend, "PrefillManager", FakePrefillManager)
    monkeypatch.setattr(sglang_backend, "DecodeManager", FakeDecodeManager)
    monkeypatch.setattr(
        sglang_backend,
        "create_tree_cache",
        lambda *args: ("tree_cache", args),
    )
    server_args = SimpleNamespace(
        speculative_algorithm=None,
        attention_backend=None,
        decode_attention_backend=None,
        prefill_attention_backend=None,
        sampling_backend=None,
        page_size=1,
        disable_overlap_schedule=False,
        disable_cuda_graph=False,
        chunked_prefill_size=8192,
        max_prefill_tokens=16384,
        context_length=32768,
        max_running_requests=64,
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(max_bs=128, bs=[1, 128]),
            prefill=SimpleNamespace(max_bs=256, bs=[4, 256]),
        ),
    )

    bootstrap.create_sglang_infrastructure(
        server_args,
        0,
        capture_hidden_layers=[0, 24],
    )

    assert events == [
        "model_worker",
        ("install_hidden_capture", FakeRunner.model, [0, 24], 8192),
        "alloc_memory_pool",
        "init_attention_backends",
        "init_cuda_graphs",
        "get_memory_pool",
    ]


def test_defer_cuda_graph_restores_requested_graph_capture(monkeypatch) -> None:
    server_args = FakeServerArgs(disable_cuda_graph=False)
    seen: list[bool] = []

    def fake_create_sglang_infrastructure(server_args, gpu_id, **kwargs):
        seen.append(bool(server_args.disable_cuda_graph))
        return ("infra", gpu_id, kwargs)

    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure",
        fake_create_sglang_infrastructure,
    )

    want_cuda_graph, infrastructure = (
        bootstrap.create_sglang_infrastructure_defer_cuda_graph(
            server_args,
            3,
            model_arch_override="TestModel",
        )
    )

    assert want_cuda_graph is True
    assert seen == [True]
    assert server_args.disable_cuda_graph is False
    assert infrastructure == (
        "infra",
        3,
        {
            "defer_cuda_graph_capture": True,
            "model_arch_override": "TestModel",
        },
    )


def test_defer_cuda_graph_leaves_disabled_graph_capture_disabled(monkeypatch) -> None:
    server_args = FakeServerArgs(disable_cuda_graph=True)
    seen: list[bool] = []

    def fake_create_sglang_infrastructure(server_args, gpu_id, **kwargs):
        del gpu_id, kwargs
        seen.append(bool(server_args.disable_cuda_graph))
        return object()

    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure",
        fake_create_sglang_infrastructure,
    )

    want_cuda_graph, _ = bootstrap.create_sglang_infrastructure_defer_cuda_graph(
        server_args,
        0,
    )

    assert want_cuda_graph is False
    assert seen == [True]
    assert server_args.disable_cuda_graph is True
