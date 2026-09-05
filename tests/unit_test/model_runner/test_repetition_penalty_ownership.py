# SPDX-License-Identifier: Apache-2.0
"""The non-speculative runner must apply the repetition penalty exactly once.

SGLang's ``model_runner.sample`` applies its device-resident
``BatchedRepetitionPenalizer`` when the batch has one prepared; Omni's host-side
``_apply_repetition_penalty`` is only a fallback for sampling infos without an
active SGLang penalizer.
"""

from types import SimpleNamespace

import torch
from sglang.srt.sampling.penaltylib import BatchedRepetitionPenalizer

from sglang_omni.model_runner.base import ModelRunner


def _forward_batch(orchestrator):
    return SimpleNamespace(
        sampling_info=SimpleNamespace(
            device="cpu", sampling_seed=None, penalizer_orchestrator=orchestrator
        ),
        top_logprobs_nums=None,
        token_ids_logprobs=None,
    )


def _orchestrator(prepared: bool, with_repetition: bool = True):
    penalizers = {}
    if with_repetition:
        penalizers[BatchedRepetitionPenalizer] = SimpleNamespace(
            is_prepared=lambda: prepared
        )
    return SimpleNamespace(penalizers=penalizers)


def test_detects_prepared_sglang_repetition_penalizer() -> None:
    assert ModelRunner._sglang_applies_repetition_penalty(
        _forward_batch(_orchestrator(prepared=True))
    )


def test_unprepared_or_missing_penalizer_falls_back_to_omni() -> None:
    assert not ModelRunner._sglang_applies_repetition_penalty(
        _forward_batch(_orchestrator(prepared=False))
    )
    assert not ModelRunner._sglang_applies_repetition_penalty(
        _forward_batch(_orchestrator(prepared=True, with_repetition=False))
    )
    assert not ModelRunner._sglang_applies_repetition_penalty(_forward_batch(None))
    assert not ModelRunner._sglang_applies_repetition_penalty(SimpleNamespace())


def _runner_with_counter():
    runner = object.__new__(ModelRunner)
    calls = {"omni": 0}

    def _omni_penalty(_logits_output, _requests):
        calls["omni"] += 1

    runner._apply_repetition_penalty = _omni_penalty
    runner._apply_codec_suppress_tokens = lambda *args: None
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            sample=lambda _logits_output, _forward_batch: torch.tensor([7])
        )
    )
    return runner, calls


def _requests():
    req = SimpleNamespace(sampling_params=SimpleNamespace(sampling_seed=None))
    data = SimpleNamespace(return_logprob=False, req=req)
    return [SimpleNamespace(data=data)]


def test_sample_skips_omni_penalty_when_sglang_penalizer_active() -> None:
    runner, calls = _runner_with_counter()
    out = runner._sample_next_token_ids(
        SimpleNamespace(next_token_logits=torch.zeros(1, 4)),
        _forward_batch(_orchestrator(prepared=True)),
        SimpleNamespace(),
        _requests(),
    )
    assert out.tolist() == [7]
    assert calls["omni"] == 0


def test_sample_applies_omni_penalty_without_sglang_penalizer() -> None:
    runner, calls = _runner_with_counter()
    runner._sample_next_token_ids(
        SimpleNamespace(next_token_logits=torch.zeros(1, 4)),
        _forward_batch(None),
        SimpleNamespace(),
        _requests(),
    )
    assert calls["omni"] == 1
