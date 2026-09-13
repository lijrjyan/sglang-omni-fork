# SPDX-License-Identifier: Apache-2.0
"""Omni lifecycle adapter for native DFlash's publication and result protocol.

Only consecutive decode forwards overlap. Prefill results retire before the next
batch selection, preserving Omni's same-iteration chunk boundary contract.
"""
import time
from collections import deque
from functools import partial

from sglang.srt.environ import envs


class NativeSpeculativeOverlap:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.pending = deque()
        scheduler.result_queue = self.pending
        scheduler.batch_record_buf = [None, None]
        scheduler.batch_record_ct = 0
        device = scheduler.device_module
        scheduler.schedule_stream = device.current_stream()
        scheduler.forward_stream = device.Stream()
        scheduler.copy_stream = device.Stream()
        scheduler.forward_stream_ctx = device.stream(scheduler.forward_stream)
        scheduler.copy_stream_ctx = device.stream(scheduler.copy_stream)

    def synchronize(self):
        self.scheduler.forward_stream.synchronize()
        self.scheduler.copy_stream.synchronize()

    def barrier(self):
        # Match native Scheduler._apply_war_barrier, selecting the speculative
        # worker's last reader. Clear even when forced coarse so an eager or
        # prefill step cannot reuse the preceding graph's read-done event.
        scheduler = self.scheduler
        runner = scheduler.draft_worker.last_shared_read_runner
        event = runner.shared_read_done_event
        runner.shared_read_done_event = None
        if event is not None and not envs.SGLANG_FORCE_COARSE_WAR_BARRIER.get():
            scheduler.schedule_stream.wait_event(event)
        else:
            scheduler.schedule_stream.wait_stream(scheduler.forward_stream)

    def drain(self):
        while self.pending:
            batch, result = self.pending.popleft()
            self.scheduler.process_batch_result(batch, result)

    def forward(self, batch):
        from sglang.srt.managers.overlap_utils import resolve_forward_inputs

        scheduler = self.scheduler
        scheduler.future_map.resolve_seq_lens_cpu(batch)
        with scheduler.forward_stream_ctx:
            scheduler.forward_stream.wait_stream(scheduler.schedule_stream)
            resolve_forward_inputs(batch, scheduler.future_map)
            with scheduler._forward_isolation(batch, overlap=True):
                indices = batch.req_pool_indices
                result = scheduler.draft_worker.forward_batch_generation(
                    batch,
                    on_publish=partial(scheduler.future_map.publish, indices),
                    grammar_barrier=scheduler._advance_pending_grammar,
                )
                if result.extra_keep_alive_refs:
                    scheduler.batch_record_buf[scheduler.batch_record_ct].extend(
                        result.extra_keep_alive_refs
                    )
                if result.delay_sample_func is not None:
                    raise RuntimeError("Native DFlash overlap cannot delay sampling")
                result.copy_done = scheduler.device_module.Event()
                scheduler._relay_forward_payload(indices, result)
                scheduler.copy_stream.wait_stream(scheduler.forward_stream)
                with scheduler.copy_stream_ctx:
                    result.copy_to_cpu(
                        return_logprob=batch.return_logprob,
                        return_hidden_states=batch.return_hidden_states,
                    )
        batch.input_ids = None
        batch.spec_info = result.next_draft_input
        batch.spec_info.future_dsa_topk_indices_available = (
            batch.spec_info.dsa_topk_indices is not None
        )
        batch.spec_info.future_indices = indices
        result.extend_input_len_per_req = (
            [
                req.extend_range.length if req.extend_range is not None else 0
                for req in batch.reqs
            ]
            if batch.return_logprob or batch.return_hidden_states
            else None
        )
        result.extend_logprob_start_len_per_req = (
            batch.extend_logprob_start_lens if batch.return_logprob else None
        )
        return result

    def run(self):
        scheduler = self.scheduler
        try:
            while scheduler._running:
                scheduler._process_admin_requests()
                recv = scheduler.recv_requests()
                recv.extend(scheduler._take_deferred_request_payloads())
                # Admission and prefill selection observe committed Req state.
                if recv or scheduler.waiting_queue or scheduler.chunked_req is not None:
                    self.drain()
                scheduler.process_input_requests(recv)
                if scheduler._engine_paused:
                    self.drain()
                    time.sleep(0.001)
                    continue
                batch = scheduler.get_next_batch_to_run()
                scheduler.cur_batch = batch
                if batch is not None:
                    result = scheduler.run_batch(batch)
                    if result is getattr(scheduler, "_failed_batch_result", None):
                        # Failure handler has fenced GPU work and retired pending
                        # results before any request resources were released.
                        scheduler.last_batch = None
                        continue
                    self.barrier()
                    self.pending.append((batch.copy(), result))
                    if len(self.pending) > 1:
                        previous, output = self.pending.popleft()
                        scheduler.process_batch_result(previous, output)
                    if not batch.forward_mode.is_decode():
                        self.drain()
                else:
                    self.drain()
                    scheduler._sched_idled = True
                    scheduler.self_check_during_idle()
                    scheduler._sleep_during_idle()
                scheduler.last_batch = batch
                if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                    scheduler.self_check_during_busy()
        finally:
            self.synchronize()
            self.drain()
