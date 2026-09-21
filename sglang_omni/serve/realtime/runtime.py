"""Bounded, transport independent realtime session ownership and media clocks."""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import uuid
from collections.abc import AsyncIterator, Callable
from contextvars import ContextVar
from typing import Literal, cast

from sglang_omni.serve.realtime.control import (
    Accepted,
    Cleared,
    Closed,
    ControlEvent,
    Created,
    Drained,
    Ended,
    Failure,
    UnitCompleted,
    Updated,
)
from sglang_omni.serve.realtime.negotiation import SessionNegotiation
from sglang_omni.serve.realtime.output import OutputEvent, ResponseStatus, TurnFailure
from sglang_omni.serve.realtime.output_buffer import OutputBuffer
from sglang_omni.serve.realtime.schema import GrantedCapabilities, SessionConfiguration
from sglang_omni.serve.realtime.task_cleanup import cancel_local_tasks
from sglang_omni.serve.realtime.types import (
    Capabilities,
    Envelope,
    InteractionAdapter,
    ProtocolError,
    RuntimeLimits,
    Unit,
)

logger = logging.getLogger(__name__)

MAX_FAILURE_MESSAGE_CHARS = 512


class SessionRuntime:
    def __init__(
        self,
        model: str,
        capabilities: Capabilities,
        adapter_factory: Callable[[], InteractionAdapter],
        limits: RuntimeLimits,
    ) -> None:
        self.session_id = "sess_" + uuid.uuid4().hex
        self.model, self.capabilities, self.limits = model, capabilities, limits
        self.factory = adapter_factory
        self.negotiation = SessionNegotiation(
            model=model, capabilities=capabilities, limits=limits
        )
        self.adapter: InteractionAdapter | None = None
        self.state: Literal["CREATED", "OPEN", "CLOSING", "CLOSED"] = "CREATED"
        self.config: SessionConfiguration = {}
        self.granted: GrantedCapabilities = {}
        self.next_seq = 0
        self.accepted_samples = 0
        self.consumed_samples = 0
        self.discarded_samples = 0
        self.padding_samples = 0
        self.pending = bytearray()
        self.unit_seq = 0
        self.eos = False
        self.eos_event_id: str | None = None
        self.lock = asyncio.Lock()
        self.wake = asyncio.Event()
        self.output_buffer = OutputBuffer(limits)
        self.worker: asyncio.Task[None] | None = None
        self.close_task: asyncio.Task[None] | None = None
        self.producer_unit: ContextVar[Unit | None] = ContextVar(
            "realtime_unit", default=None
        )

    def notify(self, event: ControlEvent) -> None:
        self.output_buffer.enqueue(Envelope(event, True))

    def created(self) -> None:
        self.notify(
            Created(
                self.session_id,
                self.model,
                "realtime",
            )
        )

    async def outputs(self) -> AsyncIterator[Envelope]:
        while True:
            while self.output_buffer.output:
                value, size = self.output_buffer.output.popleft()
                self.output_buffer.output_bytes -= size
                yield value
            if self.state == "CLOSED":
                return
            else:
                self.output_buffer.output_wake.clear()
                await self.output_buffer.output_wake.wait()

    async def emit(self, event: OutputEvent, unit: Unit | None = None) -> None:
        if self.state != "OPEN":
            return
        elif isinstance(event, TurnFailure):
            self.fail(event.message, event.code)
        else:
            unit = unit or self.producer_unit.get()
            modalities = (
                unit.output_modalities if unit is not None else None
            ) or tuple(
                self.granted.get(
                    "output_modalities", self.capabilities.output_modalities
                )
            )
            self.output_buffer.emit(event, unit, modalities)

    def ms(self, samples: int) -> float:
        return samples * 1000 / self.capabilities.input_rate

    def require_open(self) -> None:
        if self.state != "OPEN":
            raise ProtocolError("invalid_state", "session is not OPEN")

    async def update(self, patch: object, event_id: str) -> None:
        async with self.lock:
            if self.state not in ("CREATED", "OPEN"):
                raise ProtocolError("invalid_state", "session is closing")
            candidate, grant = self.negotiation.negotiate(
                self.config, self.state, patch
            )
            if self.state == "CREATED":
                adapter = self.factory()
                adapter.set_limits(self.limits)
                try:
                    await asyncio.wait_for(
                        adapter.open(self.session_id, candidate, self.emit),
                        self.limits.cleanup_timeout_s,
                    )
                except Exception as exc:
                    try:
                        await asyncio.wait_for(
                            adapter.close(), self.limits.cleanup_timeout_s
                        )
                    except Exception:
                        self.adapter = adapter
                        raise RuntimeError("admission cleanup failed") from exc
                    raise ProtocolError("admission_rejected", str(exc)) from exc
                self.adapter = adapter
                if self.close_task is not None:
                    # Note (Junnan Li): Admission cleanup belongs to the closing owner; do not publish OPEN here.
                    return
                self.state = "OPEN"
                self.worker = asyncio.create_task(self.pump())
            self.config, self.granted = candidate, grant
            self.notify(
                Updated(
                    self.session_id,
                    self.model,
                    candidate["type"],
                    grant,
                    event_id,
                    copy.deepcopy(candidate),
                )
            )

    async def append(
        self, pcm: bytes, seq: object, start_ms: object, event_id: str
    ) -> None:
        async with self.lock:
            self.require_open()
            if self.eos:
                raise ProtocolError("invalid_state", "audio input has ended")
            if type(seq) is not int or seq != self.next_seq:
                raise ProtocolError("invalid_state", "audio seq must be contiguous")
            if not pcm or len(pcm) % 2:
                raise ProtocolError(
                    "invalid_request", "audio must contain whole PCM16 samples"
                )
            if start_ms is not None and (
                type(start_ms) not in (int, float)
                or not math.isfinite(cast(float, start_ms))
                or not math.isclose(
                    cast(float, start_ms),
                    self.ms(self.accepted_samples),
                    rel_tol=0,
                    abs_tol=1e-7,
                )
            ):
                raise ProtocolError(
                    "invalid_state", "input media time must be sample-contiguous"
                )
            if (
                self.accepted_samples - self.consumed_samples - self.discarded_samples
            ) * 2 + len(pcm) > self.limits.max_input_bytes:
                raise ProtocolError(
                    "buffer_overflow", "input budget exhausted; retry this seq"
                )
            self.pending.extend(pcm)
            self.accepted_samples += len(pcm) // 2
            self.next_seq += 1
            self.notify(Accepted(seq, self.ms(self.accepted_samples), event_id))
            self.wake.set()

    async def clear(self, event_id: str) -> None:
        async with self.lock:
            self.require_open()
            discarded = len(self.pending) // 2
            assert self.adapter is not None
            discarded += await self.adapter.clear()
            self.pending.clear()
            self.discarded_samples += discarded
            self.notify(Cleared(self.ms(discarded), event_id))

    async def end(self, event_id: str) -> None:
        async with self.lock:
            self.require_open()
            if self.eos:
                raise ProtocolError("invalid_state", "audio input already ended")
            unit_bytes = (
                self.capabilities.input_rate
                * self.capabilities.native_unit_ms
                // 1000
                * 2
            )
            if (
                self.capabilities.tail_policy == "reject"
                and len(self.pending) % unit_bytes
            ):
                raise ProtocolError(
                    "invalid_state", "partial native unit; append more audio before EOS"
                )
            self.eos = True
            self.eos_event_id = event_id
            self.notify(
                Ended(
                    self.ms(self.accepted_samples),
                    self.capabilities.tail_policy,
                    event_id,
                )
            )
            self.wake.set()

    async def pump(self) -> None:
        assert self.adapter is not None
        unit_bytes = (
            self.capabilities.input_rate * self.capabilities.native_unit_ms // 1000 * 2
        )
        try:
            while self.state == "OPEN":
                await self.wake.wait()
                async with self.lock:
                    self.wake.clear()
                    if self.state != "OPEN":
                        return
                    if len(self.pending) < unit_bytes and not self.eos:
                        continue
                    size = min(unit_bytes, len(self.pending))
                    start = self.accepted_samples - len(self.pending) // 2
                    pcm = bytes(self.pending[:size])
                    del self.pending[:size]
                    real = len(pcm) // 2
                    eos = self.eos and not self.pending
                    if (
                        real
                        and size < unit_bytes
                        and self.capabilities.tail_policy == "pad"
                    ):
                        self.padding_samples += (unit_bytes - size) // 2
                        pcm += b"\0" * (unit_bytes - size)
                    unit = Unit(
                        self.unit_seq,
                        start,
                        pcm,
                        real,
                        eos,
                        tuple(self.granted["output_modalities"]),
                    )
                    self.unit_seq += 1
                self.producer_unit.set(unit)
                consumed = await self.adapter.process(unit)
                if isinstance(consumed, tuple):
                    consumed, discarded = consumed
                else:
                    discarded = real - consumed
                if (
                    any(type(x) is not int or x < 0 for x in (consumed, discarded))
                    or self.consumed_samples
                    + self.discarded_samples
                    + consumed
                    + discarded
                    > self.accepted_samples
                ):
                    raise RuntimeError(
                        "adapter did not provide valid media consumption"
                    )
                self.consumed_samples += consumed
                self.discarded_samples += discarded
                if self.close_task is None:
                    self.output_buffer.enqueue(
                        Envelope(UnitCompleted(f"unit_{unit.seq}"), unit=unit)
                    )
                if eos:
                    if self.close_task is not None:
                        return
                    assert self.eos_event_id is not None
                    self.notify(
                        Drained(
                            self.ms(self.accepted_samples),
                            self.ms(self.consumed_samples),
                            self.ms(self.discarded_samples),
                            self.ms(self.padding_samples),
                            self.eos_event_id,
                        )
                    )
                    return
                self.wake.set()
        except asyncio.CancelledError:
            raise
        except ProtocolError as exc:
            self.fail(str(exc), exc.code)
        except Exception as exc:
            logger.exception(f"Realtime session {self.session_id} input pump failed")
            self.fail(str(exc))

    def fail(
        self, message: str, code: str = "internal", event_id: str | None = None
    ) -> None:
        if self.close_task is None:
            self.output_buffer.terminal(
                Failure(code, message[:MAX_FAILURE_MESSAGE_CHARS], True, event_id)
            )
            self.close_task = asyncio.create_task(self.run_close(code))

    async def close(self, reason: str, event_id: str | None = None) -> None:
        if self.close_task is None:
            self.close_task = asyncio.create_task(self.run_close(reason, event_id))
        await asyncio.shield(self.close_task)

    async def run_close(self, reason: str, event_id: str | None = None) -> None:
        # Note (Junnan Li): Set CLOSING under the command lock, then release it: adapter
        # teardown can run a VAD callback that must observe CLOSING.
        async with self.lock:
            self.state = "CLOSING"
        await self.close_state(reason, event_id)

    async def close_state(self, reason: str, event_id: str | None = None) -> None:
        self.discarded_samples += len(self.pending) // 2
        self.pending.clear()
        self.wake.set()
        cleanup_error = None
        try:
            if self.adapter is not None:
                await asyncio.wait_for(
                    self.adapter.close(), self.limits.cleanup_timeout_s
                )
        except Exception as exc:
            cleanup_error = exc
        finally:
            try:
                await cancel_local_tasks([self.worker], self.limits.cleanup_timeout_s)
            except Exception as exc:
                cleanup_error = cleanup_error or exc
        try:
            if cleanup_error is not None:
                self.output_buffer.output.clear()
                self.output_buffer.output_bytes = 0
                self.output_buffer.terminal(
                    Failure(
                        "cleanup_timeout",
                        str(cleanup_error)[:MAX_FAILURE_MESSAGE_CHARS],
                        True,
                        event_id,
                    )
                )
            else:
                status: ResponseStatus = (
                    "cancelled"
                    if reason in ("client_closed", "disconnect")
                    else "failed"
                )
                self.output_buffer.finish_responses(status, reason)
                self.output_buffer.terminal(Closed(reason, event_id))
        finally:
            self.state = "CLOSED"
            self.output_buffer.output_wake.set()
