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
from typing import Any

import torch
from sglang.srt.managers.overlap_utils import resolve_forward_inputs

from sglang_omni.scheduling.speculative_support import (
    eager_prefill_has_hidden,
    is_omni_supported_speculative_algorithm,
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


def cumulate_accepted_tokens(batch: Any, batch_result: Any) -> None:
    """Reserved for lanes with exact per-prefix sampling processors.

    Family admission owns the sampling restrictions; this hook restores no
    history. Greedy sampling alone does not make history penalties safe to omit.
    Replaying between steps also cannot repair native block-broadcast penalties.
    """
    del batch, batch_result


def run_speculative_batch(scheduler: Any, batch: Any) -> Any:
    """Run one native synchronous speculative batch without reconstruction."""
    scheduler._emit_prefill_start_for_batch(batch)
    scheduler._stamp_batch_launch(batch)
    resolve_forward_inputs(batch, scheduler.future_map)
    with scheduler._forward_isolation(batch, overlap=False):
        batch_result = scheduler.draft_worker.forward_batch_generation(batch)
    handoff_speculative_result(batch, batch_result, algorithm=scheduler.spec_algorithm)
    cumulate_accepted_tokens(batch, batch_result)
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
