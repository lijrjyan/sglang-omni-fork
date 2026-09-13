# SPDX-License-Identifier: Apache-2.0
"""Speculative history uses each request's committed output length."""

from types import SimpleNamespace

import torch
from sglang.srt.sampling.penaltylib import (
    BatchedFrequencyPenalizer,
    BatchedMinNewTokensPenalizer,
    BatchedPresencePenalizer,
    BatchedRepetitionPenalizer,
)

from sglang_omni.scheduling.speculative_bridge import cumulate_accepted_tokens


def _batch(histories):
    reqs = [
        SimpleNamespace(
            output_ids=list(ids),
            sampling_params=SimpleNamespace(
                repetition_penalty=1.3,
                frequency_penalty=0.25,
                presence_penalty=0.5,
                min_new_tokens=2,
                stop_token_ids={7},
            ),
            tokenizer=SimpleNamespace(additional_stop_token_ids=set(), eos_token_id=7),
        )
        for ids in histories
    ]

    class Orchestrator:
        vocab_size = 8
        device = "cpu"

        def __init__(self):
            self.penalizers = {}

        def reqs(self):
            return reqs

    orchestrator = Orchestrator()
    for cls in (
        BatchedFrequencyPenalizer,
        BatchedMinNewTokensPenalizer,
        BatchedPresencePenalizer,
        BatchedRepetitionPenalizer,
    ):
        penalizer = cls(orchestrator)
        penalizer.prepare()
        orchestrator.penalizers[cls] = penalizer
    orchestrator.cumulate_output_tokens = lambda ids: [
        p.cumulate_output_tokens(ids) for p in orchestrator.penalizers.values()
    ]
    return SimpleNamespace(
        reqs=reqs,
        device="cpu",
        sampling_info=SimpleNamespace(penalizer_orchestrator=orchestrator),
    )


def _check(batch):
    states = batch.sampling_info.penalizer_orchestrator.penalizers
    minimum = states[BatchedMinNewTokensPenalizer]
    assert minimum.len_output_tokens.flatten().tolist() == [
        len(req.output_ids) for req in batch.reqs
    ]
    logits = torch.zeros(len(batch.reqs), 8)
    minimum.apply(logits)
    assert torch.isneginf(logits[:, 7]).tolist() == [
        len(req.output_ids) < 2 for req in batch.reqs
    ]
    for row, req in enumerate(batch.reqs):
        for token in range(8):
            count = req.output_ids.count(token)
            assert (
                states[BatchedFrequencyPenalizer]
                .cumulated_frequency_penalties[row, token]
                .item()
                == count * 0.25
            )
            assert states[BatchedPresencePenalizer].cumulated_presence_penalties[
                row, token
            ].item() == (0.5 if count else 0)
            torch.testing.assert_close(
                states[BatchedRepetitionPenalizer].cumulated_repetition_penalties[
                    row, token
                ],
                torch.tensor(1.3 if count else 1.0),
            )


def test_unequal_accept_lengths_and_repeated_preparation():
    batch = _batch([[2, 3, 2], [4]])
    for _ in range(2):
        cumulate_accepted_tokens(batch)
        _check(batch)
    batch.reqs[1].output_ids.append(7)
    cumulate_accepted_tokens(batch)
    _check(batch)


def test_empty_history_does_not_defer_other_requests():
    batch = _batch([[2, 3], []])
    cumulate_accepted_tokens(batch)
    _check(batch)


def test_fresh_penalizers_restore_retained_history_without_request_cursor():
    batch = _batch([[2, 2, 3], [4]])
    for req in batch.reqs:
        req._omni_penalized_len = len(req.output_ids)
    cumulate_accepted_tokens(batch)
    _check(batch)


def test_native_accumulation_is_not_counted_twice():
    batch = _batch([[2, 3, 2], [4]])
    batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
        torch.tensor([2, 4])
    )
    cumulate_accepted_tokens(batch)
    _check(batch)


def test_history_follows_rows_after_filter_and_merge():
    batch = _batch([[2, 3, 2], [4]])
    cumulate_accepted_tokens(batch)
    states = batch.sampling_info.penalizer_orchestrator.penalizers
    for penalizer in states.values():
        penalizer.filter(torch.tensor([1]))
    batch.reqs[:] = batch.reqs[1:]
    other = _batch([[5, 5], []])
    cumulate_accepted_tokens(other)
    for cls, penalizer in states.items():
        penalizer.merge(other.sampling_info.penalizer_orchestrator.penalizers[cls])
    batch.reqs.extend(other.reqs)
    batch.reqs[0].output_ids.append(6)
    cumulate_accepted_tokens(batch)
    _check(batch)


def test_empty_histories_reset_prepared_state():
    batch = _batch([[], []])
    batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
        torch.tensor([2, 4])
    )
    cumulate_accepted_tokens(batch)
    _check(batch)
