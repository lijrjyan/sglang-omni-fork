# SPDX-License-Identifier: Apache-2.0
"""Omni scheduler bridge for SGLang's native speculative workers (L0).

Omni runs its own scheduler, so the pieces SGLang's ``Scheduler`` would do
around a native V2 draft worker live here: synchronous dispatch through
``forward_batch_generation``, the result handoff (``next_draft_input`` /
``new_seq_lens``), terminal trimming by ``req.finished_len``, the STANDALONE
eager-prefill hidden-state clear, and the row-filter keyword compatibility for
the different ``SpecInput`` classes. Nothing here depends on the model family;
model-specific runtime needs go through ``speculative_model_support``.
"""

from __future__ import annotations

import inspect
from collections import Counter
from typing import Any

import torch
from sglang.srt.managers.overlap_utils import resolve_forward_inputs

from sglang_omni.scheduling.speculative_support import (
    eager_prefill_has_hidden,
    is_omni_supported_speculative_algorithm,
    requires_history_restore,
    unsupported_speculative_algorithm_message,
)


def validate_scheduler_speculative_config(
    algorithm: Any,
    *,
    requested_algorithm: Any,
    enable_overlap: bool,
    enable_async_decode: bool,
    target_worker_adapter: Any,
    draft_worker: Any,
) -> None:
    """Admit ``algorithm`` to Omni's synchronous lane or raise ``ValueError``."""
    if not is_omni_supported_speculative_algorithm(algorithm):
        raise ValueError(unsupported_speculative_algorithm_message(requested_algorithm))
    if enable_overlap:
        raise ValueError("Omni speculative decoding requires enable_overlap=False")
    if enable_async_decode:
        raise ValueError("Omni speculative decoding requires enable_async_decode=False")
    if target_worker_adapter is None or draft_worker is None:
        raise ValueError(
            "Omni speculative decoding requires prebuilt target and draft workers"
        )


def create_future_map(algorithm: Any, device: Any, req_to_token_pool: Any) -> Any:
    """Build the native future map the synchronous lane resolves against."""
    return algorithm.create_future_map(
        torch.device(device),
        req_to_token_pool,
        needs_cpu_seq_lens=True,
    )


def refuses_omni_lookahead(algorithm: Any) -> bool:
    """Omni's one-token lookahead cannot run under a native speculative worker."""
    return not algorithm.is_none()


def finished_output_ids(req: Any) -> list[int]:
    """Output ids up to the request's finish point.

    Native speculative verification can accept tokens past the stop token in
    the same step; SGLang keeps them in ``req.output_ids`` and records the
    finish position in ``req.finished_len``. Emit only the finished prefix so
    the terminal payload matches the one-token-per-step path.
    """
    output_ids = getattr(req, "output_ids_through_stop", None)
    if output_ids is None:
        output_ids = req.output_ids
        finished_len = getattr(req, "finished_len", None)
        if finished_len is not None:
            output_ids = output_ids[:finished_len]
    return list(output_ids)


def handoff_speculative_result(
    batch: Any, batch_result: Any, *, algorithm: Any
) -> None:
    """Carry the native worker's draft state and sequence lengths onto the batch."""
    batch.spec_info = batch_result.next_draft_input
    if not eager_prefill_has_hidden(algorithm):
        # STANDALONE has no recurrent state. Native extend graphs can
        # materialize it, but merging with state-free prefills cannot
        # maintain its batch dimension.
        batch.spec_info.hidden_states = None
    if batch_result.new_seq_lens is not None:
        batch.seq_lens = batch_result.new_seq_lens
        if batch.seq_lens_cpu is not None:
            batch.seq_lens_cpu = batch_result.new_seq_lens.to("cpu")
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
    batch.input_ids = None


def cumulate_accepted_tokens(batch) -> None:
    """Restore prepared penalizers from committed output histories.

    Native accumulation accepts one token per row; padding ragged histories
    changes frequency and min-length semantics. Absolute state also survives
    re-prefill, filtering and merging without a request-local cursor.
    """
    from sglang.srt.sampling.penaltylib import (
        BatchedFrequencyPenalizer,
        BatchedMinNewTokensPenalizer,
        BatchedPresencePenalizer,
        BatchedRepetitionPenalizer,
    )

    reqs = batch.reqs
    if not reqs:
        return
    orchestrator = batch.sampling_info.penalizer_orchestrator
    minimum = orchestrator.penalizers.get(BatchedMinNewTokensPenalizer)
    if minimum is not None and minimum.is_prepared():
        minimum.len_output_tokens.copy_(
            torch.tensor(
                [len(req.output_ids) for req in reqs],
                dtype=minimum.len_output_tokens.dtype,
                device=batch.device,
            ).view(-1, 1)
        )

    rows, tokens, counts = [], [], []
    for row, req in enumerate(reqs):
        history = Counter(
            token
            for value in req.output_ids
            if 0 <= (token := int(value)) < orchestrator.vocab_size
        )
        rows.extend([row] * len(history))
        tokens.extend(history)
        counts.extend(history.values())
    row_indices = torch.tensor(rows, dtype=torch.long, device=batch.device)
    token_indices = torch.tensor(tokens, dtype=torch.long, device=batch.device)
    for cls, state_name, param_name, initial in (
        (
            BatchedRepetitionPenalizer,
            "cumulated_repetition_penalties",
            "repetition_penalty",
            1,
        ),
        (
            BatchedFrequencyPenalizer,
            "cumulated_frequency_penalties",
            "frequency_penalty",
            0,
        ),
        (
            BatchedPresencePenalizer,
            "cumulated_presence_penalties",
            "presence_penalty",
            0,
        ),
    ):
        penalizer = orchestrator.penalizers.get(cls)
        if penalizer is None or not penalizer.is_prepared():
            continue
        state = getattr(penalizer, state_name)
        state.fill_(initial)
        values = [
            float(getattr(reqs[row].sampling_params, param_name))
            * (count if cls is BatchedFrequencyPenalizer else 1)
            for row, count in zip(rows, counts)
        ]
        state[row_indices, token_indices] = torch.tensor(
            values, dtype=state.dtype, device=batch.device
        )


def run_speculative_batch(scheduler: Any, batch: Any) -> Any:
    """Run one native synchronous speculative batch without reconstruction."""
    scheduler._emit_prefill_start_for_batch(batch)
    scheduler._stamp_batch_launch(batch)
    resolve_forward_inputs(batch, scheduler.future_map)
    prepare_speculative_batch(scheduler, batch)
    with scheduler._forward_isolation(batch, overlap=False):
        batch_result = scheduler.draft_worker.forward_batch_generation(batch)
    handoff_speculative_result(batch, batch_result, algorithm=scheduler.spec_algorithm)
    scheduler.update_cache_from_scheduler(batch, batch_result)
    batch_result.copy_done = scheduler.device_module.Event()
    batch_result.copy_to_cpu(
        return_logprob=batch.return_logprob,
        return_hidden_states=batch.return_hidden_states,
    )
    if batch.return_logprob or batch.return_hidden_states:
        batch_result.extend_input_len_per_req = [
            req.extend_range.length if req.extend_range is not None else 0
            for req in batch.reqs
        ]
    else:
        batch_result.extend_input_len_per_req = None
    if batch.return_logprob:
        batch_result.extend_logprob_start_len_per_req = batch.extend_logprob_start_lens
    else:
        batch_result.extend_logprob_start_len_per_req = None
    scheduler._emit_prefill_end_for_batch(batch)
    return batch_result


def clear_spec_info_rows(spec_info: Any, device: Any) -> None:
    """Drop every draft-state row after the batch was filtered to nothing.

    ``EagleDraftInput.filter_batch`` takes ``has_been_filtered``;
    ``DFlashDraftInputV2.filter_batch`` does not. Pass the keyword only when
    the ``SpecInput`` class accepts it.
    """
    empty_device_indices = torch.empty(0, dtype=torch.long, device=device)
    filter_kwargs: dict[str, Any] = {
        "new_indices": empty_device_indices,
        "new_indices_cpu": [],
    }
    if "has_been_filtered" in inspect.signature(spec_info.filter_batch).parameters:
        filter_kwargs["has_been_filtered"] = False
    spec_info.filter_batch(**filter_kwargs)


__all__ = [
    "clear_spec_info_rows",
    "create_future_map",
    "cumulate_accepted_tokens",
    "finished_output_ids",
    "handoff_speculative_result",
    "refuses_omni_lookahead",
    "run_speculative_batch",
    "validate_scheduler_speculative_config",
]


def prepare_speculative_batch(scheduler: Any, batch: Any) -> None:
    """Prepare embeddings, restore history, then publish live sampling state."""
    runner = getattr(scheduler, "_model_runner", None)
    forward_mode = getattr(batch, "forward_mode", None)
    if (
        runner is not None
        and forward_mode is not None
        and forward_mode.is_extend()
        and not forward_mode.is_idle()
    ):
        output = scheduler._build_sched_output(batch)
        embeds = runner.speculative_prefill_input_embeds(batch, output.requests)
        if embeds is not None:
            batch.input_embeds = embeds
    info = getattr(batch, "sampling_info", None)
    orchestrator = getattr(info, "penalizer_orchestrator", None)
    if orchestrator is not None and orchestrator.is_required:
        if requires_history_restore(scheduler.spec_algorithm):
            cumulate_accepted_tokens(batch)
        info.update_penalties()
    worker = getattr(scheduler, "model_worker", None)
    if worker is not None:
        publish = getattr(
            worker.model_runner.model, "set_speculative_sampling_info", None
        )
        if publish is not None:
            publish(info)
