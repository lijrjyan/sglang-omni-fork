# SPDX-License-Identifier: Apache-2.0
"""Speculative target adapter: Omni prefill hooks and target admission."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner import model_worker as model_worker_module
from sglang_omni.model_runner.model_worker import (
    SpeculativeTargetWorkerAdapter,
    register_speculative_omni_models,
)
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
    get_omni_prefill_inputs,
)


def test_register_accepts_qwen3_thinker_without_registry_write(monkeypatch) -> None:
    import sglang.srt.models.registry as registry_module

    from sglang_omni.models.qwen3_omni.bootstrap import (
        register_thinker_speculative_target,
    )

    register_thinker_speculative_target()
    before = dict(registry_module.ModelRegistry.models)
    register_speculative_omni_models("Qwen3OmniThinkerForCausalLM")
    assert dict(registry_module.ModelRegistry.models) == before


def test_register_rejects_unknown_target_architecture() -> None:
    with pytest.raises(ValueError, match="supports target architectures"):
        register_speculative_omni_models("Qwen3TTSTalker")


class _Mode:
    def __init__(self, extend: bool, verify: bool = False) -> None:
        self._extend = extend
        self._verify = verify

    def is_extend(self) -> bool:
        return self._extend

    def is_target_verify(self) -> bool:
        return self._verify


def _adapter(monkeypatch, *, forward_mode: _Mode, batch_size: int = 2):
    forward_batch = SimpleNamespace(
        forward_mode=forward_mode,
        input_ids=torch.arange(batch_size * 3),
        input_embeds=None,
        replace_embeds=None,
        mm_input_embeds=None,
        is_prefill_only=False,
        seq_lens=torch.ones(batch_size),
        return_logprob=False,
        apply_deprecated_skip_attn_backend_init=lambda flag: None,
    )
    logits_output = SimpleNamespace(next_token_logits=None)
    runner_out = SimpleNamespace(
        logits_output=logits_output,
        can_run_graph=False,
        expert_distribution_metrics=None,
        routed_experts_output=None,
        indexer_topk_output=None,
    )
    model_runner = SimpleNamespace(
        forward=lambda fb, pp_proxy_tensors=None: runner_out,
        sample=lambda logits, fb: torch.zeros(batch_size, dtype=torch.long),
    )
    worker = SimpleNamespace(
        server_args="args",
        model_runner=model_runner,
        model_config="cfg",
        device="cpu",
        gpu_id=0,
    )
    from sglang.srt.model_executor import forward_batch_info

    monkeypatch.setattr(
        forward_batch_info.ForwardBatch,
        "init_new",
        staticmethod(lambda *args, **kwargs: forward_batch),
    )
    adapter = SpeculativeTargetWorkerAdapter(worker, tokenizer=object())
    return adapter, forward_batch


def _batch(batch_size: int = 2):
    return SimpleNamespace(
        reqs=[
            SimpleNamespace(rid=f"r{i}", _omni_data=f"d{i}") for i in range(batch_size)
        ]
    )


class _OmniRunner:
    def __init__(self, *, sidecar: bool, custom_result=None) -> None:
        self.events: list[str] = []
        self.sidecar = sidecar
        self.custom_result = custom_result

    def before_prefill(self, forward_batch, schedule_batch, requests):
        self.events.append("before")
        assert [r.request_id for r in requests] == [q.rid for q in schedule_batch.reqs]
        assert [r.data for r in requests] == [q._omni_data for q in schedule_batch.reqs]
        embeds = torch.full((len(forward_batch.input_ids), 4), 7.0)
        if self.sidecar:
            attach_omni_prefill_inputs(forward_batch, OmniPrefillInputs(embeds))
        else:
            forward_batch.input_embeds = embeds

    def custom_prefill_forward(self, forward_batch, schedule_batch, requests):
        self.events.append("custom")
        return self.custom_result

    def cleanup_prefill(self, forward_batch, schedule_batch, requests):
        self.events.append("cleanup")


@pytest.mark.parametrize("sidecar", [True, False])
def test_extend_runs_omni_prefill_hooks_and_publishes_mm_embeds(
    monkeypatch, sidecar: bool
) -> None:
    adapter, forward_batch = _adapter(monkeypatch, forward_mode=_Mode(extend=True))
    runner = _OmniRunner(sidecar=sidecar)
    adapter.bind_omni_model_runner(runner)
    published = {}
    adapter.model_runner.forward = lambda fb, pp_proxy_tensors=None: (
        published.setdefault("embeds", fb.mm_input_embeds),
        published.setdefault("sidecar", get_omni_prefill_inputs(fb)),
        SimpleNamespace(
            logits_output=SimpleNamespace(next_token_logits=None),
            can_run_graph=False,
            expert_distribution_metrics=None,
            routed_experts_output=None,
            indexer_topk_output=None,
        ),
    )[-1]

    adapter.forward_batch_generation(_batch())

    assert runner.events == ["before", "custom", "cleanup"]
    assert (
        published["embeds"].shape == (6, 4) and float(published["embeds"][0, 0]) == 7.0
    )
    assert (published["sidecar"] is not None) is sidecar
    assert get_omni_prefill_inputs(forward_batch) is None


def test_verify_skips_omni_prefill_hooks(monkeypatch) -> None:
    adapter, _ = _adapter(monkeypatch, forward_mode=_Mode(extend=True, verify=True))
    runner = _OmniRunner(sidecar=True)
    adapter.bind_omni_model_runner(runner)

    adapter.forward_batch_generation(_batch(), is_verify=True)

    assert runner.events == []


def test_runner_owned_prefill_forward_is_refused(monkeypatch) -> None:
    adapter, _ = _adapter(monkeypatch, forward_mode=_Mode(extend=True))
    runner = _OmniRunner(sidecar=False, custom_result=object())
    adapter.bind_omni_model_runner(runner)

    with pytest.raises(NotImplementedError, match="runner-owned prefill"):
        adapter.forward_batch_generation(_batch())
    assert runner.events == ["before", "custom", "cleanup"]


def test_bind_rejects_a_second_runner(monkeypatch) -> None:
    adapter, _ = _adapter(monkeypatch, forward_mode=_Mode(extend=True))
    adapter.bind_omni_model_runner(object())
    with pytest.raises(RuntimeError, match="already has an Omni model runner"):
        adapter.bind_omni_model_runner(object())
    assert model_worker_module.SpeculativeTargetWorkerAdapter is type(adapter)
