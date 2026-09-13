# SPDX-License-Identifier: Apache-2.0
from collections import deque
from types import SimpleNamespace

import pytest

from sglang_omni.scheduling.speculative_overlap import NativeSpeculativeOverlap


class Batch:
    def __init__(self, name, decode=True):
        self.name = name
        self.forward_mode = SimpleNamespace(is_decode=lambda: decode)

    def copy(self):
        return self


def make_bridge(batches):
    events = []
    bridge = object.__new__(NativeSpeculativeOverlap)
    bridge.pending = deque()
    iterator = iter(batches)

    def select():
        events.append("select")
        batch = next(iterator, None)
        if batch is None:
            scheduler._running = False
        return batch

    scheduler = SimpleNamespace(
        _running=True,
        _engine_paused=False,
        waiting_queue=[],
        chunked_req=None,
        _process_admin_requests=lambda: None,
        recv_requests=lambda: [],
        _take_deferred_request_payloads=lambda: [],
        process_input_requests=lambda _: None,
        get_next_batch_to_run=select,
        run_batch=lambda b: events.append("launch:" + b.name) or b.name,
        process_batch_result=lambda b, r: events.append("process:" + r),
        self_check_during_idle=lambda: None,
        _sleep_during_idle=lambda: None,
    )
    bridge.scheduler = scheduler
    bridge.synchronize = lambda: events.append("sync")
    bridge.barrier = lambda: events.append("barrier")
    return bridge, events


def test_decode_launch_overlaps_previous_result_processing():
    bridge, events = make_bridge([Batch("a"), Batch("b")])
    bridge.run()
    assert events.index("launch:b") < events.index("process:a")
    assert events.index("barrier") < events.index("process:a")
    assert events.count("process:a") == events.count("process:b") == 1
    assert not bridge.pending


def test_prefill_drained_before_next_selection():
    bridge, events = make_bridge([Batch("prefill", False), Batch("decode")])
    bridge.run()
    assert events[:5] == [
        "select",
        "launch:prefill",
        "barrier",
        "process:prefill",
        "select",
    ]


def test_waiting_admission_drains_before_selection():
    bridge, events = make_bridge([Batch("new", False)])
    bridge.pending.append((Batch("old"), "old"))
    bridge.scheduler.waiting_queue.append(object())
    bridge.run()
    assert events.index("process:old") < events.index("select")


def test_drain_removes_pending_before_callback():
    bridge, events = make_bridge([])
    bridge.pending.append((Batch("old"), "old"))
    bridge.scheduler.process_batch_result = lambda b, r: bridge.drain()
    bridge.drain()
    assert not bridge.pending


def test_failed_loop_synchronizes_and_drains():
    bridge, events = make_bridge([Batch("bad")])
    bridge.pending.append((Batch("old"), "old"))

    def fail(b):
        raise RuntimeError("forward failed")

    bridge.scheduler.run_batch = fail
    with pytest.raises(RuntimeError, match="forward failed"):
        bridge.run()
    assert events.index("sync") < events.index("process:old")
    assert not bridge.pending


def test_native_forward_preserves_publication_result_and_lifetimes(monkeypatch):
    from contextlib import nullcontext

    from sglang.srt.managers import overlap_utils

    calls = []
    indices = object()
    refs = [object()]
    next_input = SimpleNamespace(dsa_topk_indices=None)
    result = SimpleNamespace(
        extra_keep_alive_refs=refs,
        delay_sample_func=None,
        next_draft_input=next_input,
        copy_to_cpu=lambda **kw: calls.append("copy"),
    )
    batch = SimpleNamespace(
        req_pool_indices=indices,
        return_logprob=False,
        return_hidden_states=False,
        input_ids=object(),
    )

    def worker(batch, on_publish, grammar_barrier):
        calls.append("worker")
        on_publish("accepted-lengths")
        return result

    scheduler = SimpleNamespace(
        future_map=SimpleNamespace(
            resolve_seq_lens_cpu=lambda b: calls.append("cpu-capability"),
            publish=lambda rows, lengths: calls.append(("publish", rows, lengths)),
        ),
        forward_stream_ctx=nullcontext(),
        copy_stream_ctx=nullcontext(),
        forward_stream=SimpleNamespace(
            wait_stream=lambda s: calls.append("wait-schedule")
        ),
        schedule_stream=object(),
        copy_stream=SimpleNamespace(wait_stream=lambda s: calls.append("wait-forward")),
        _forward_isolation=lambda b, overlap: nullcontext(),
        draft_worker=SimpleNamespace(forward_batch_generation=worker),
        _advance_pending_grammar=lambda: None,
        batch_record_buf=[[]],
        batch_record_ct=0,
        device_module=SimpleNamespace(Event=object),
        _relay_forward_payload=lambda rows, r: calls.append(("relay", rows, r)),
    )
    monkeypatch.setattr(
        overlap_utils, "resolve_forward_inputs", lambda b, m: calls.append("resolve")
    )
    bridge = object.__new__(NativeSpeculativeOverlap)
    bridge.scheduler = scheduler
    assert bridge.forward(batch) is result
    assert calls[:3] == ["cpu-capability", "wait-schedule", "resolve"]
    assert calls[4] == ("publish", indices, "accepted-lengths")
    assert calls[5] == ("relay", indices, result)
    assert calls[-2:] == ["wait-forward", "copy"]
    assert scheduler.batch_record_buf[0] == refs
    assert batch.spec_info is next_input
    assert next_input.future_indices is indices
    assert batch.input_ids is None


@pytest.mark.parametrize("force_coarse", [False, True])
def test_shared_read_fence_consumed_once_with_coarse_fallback(
    monkeypatch, force_coarse
):
    from sglang.srt.environ import envs

    monkeypatch.setenv("SGLANG_FORCE_COARSE_WAR_BARRIER", "1" if force_coarse else "0")
    assert envs.SGLANG_FORCE_COARSE_WAR_BARRIER.get() is force_coarse
    event = object()
    runner = SimpleNamespace(shared_read_done_event=event)
    calls = []
    forward_stream = object()
    bridge = object.__new__(NativeSpeculativeOverlap)
    bridge.scheduler = SimpleNamespace(
        draft_worker=SimpleNamespace(last_shared_read_runner=runner),
        forward_stream=forward_stream,
        schedule_stream=SimpleNamespace(
            wait_event=lambda e: calls.append(("event", e)),
            wait_stream=lambda s: calls.append(("stream", s)),
        ),
    )
    bridge.barrier()
    assert runner.shared_read_done_event is None
    assert calls == [("stream", forward_stream) if force_coarse else ("event", event)]
    # A following eager/prefill forward must not reuse an earlier graph event.
    bridge.barrier()
    assert calls[-1] == ("stream", forward_stream)
    assert len(calls) == 2
