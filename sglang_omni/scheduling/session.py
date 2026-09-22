# SPDX-License-Identifier: Apache-2.0
"""Persistent state for session-aware pipeline stages."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Literal, Protocol

from sglang_omni.admission import QueueFullError
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    ResourceUsage,
    SessionOperation,
    SessionRef,
    TimedChunk,
    find_session_operation,
)
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

DEFAULT_MAX_OPEN_SESSIONS = 64
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_MAX_STATE_BYTES = 1 << 30


class ChunkEmitter(Protocol):
    def __call__(self, chunk: TimedChunk) -> None: ...


class OperationRegistrar(Protocol):
    def __call__(self, message: IncomingMessage) -> None: ...


class StageCompute(Protocol):
    def __call__(self, payload: StagePayload) -> StagePayload: ...


@dataclass(kw_only=True)
class SessionContext:
    ref: SessionRef
    cancelled: threading.Event
    emit: ChunkEmitter


class SessionHooks:
    """Hooks run serially per session; different sessions may run concurrently.

    Each hook keeps the state it created. The scheduler only passes SessionRef.
    Failed open releases allocations it has not returned. close is idempotent.
    """

    def open(self, ref: SessionRef, request: OmniRequest) -> None:
        raise NotImplementedError

    def append(
        self,
        chunk: TimedChunk,
        payload: StagePayload,
        context: SessionContext,
    ) -> StagePayload:
        raise NotImplementedError

    def close(self, ref: SessionRef) -> None:
        raise NotImplementedError

    def usage(self, ref: SessionRef) -> ResourceUsage:
        return ResourceUsage()


@dataclass(kw_only=True)
class StageSession:
    is_open: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    usage: ResourceUsage = field(default_factory=ResourceUsage)


@dataclass(frozen=True, kw_only=True)
class OperationArrival:
    """Arrival position of one accepted operation inside its session."""

    ref: SessionRef
    operation: Literal["open", "append", "close"]
    sequence: int


@dataclass(kw_only=True)
class SessionOperationCursor:
    """Arrival cursor for one session. An operation starts when its sequence is runnable."""

    next_sequence: int = 0
    runnable_sequence: int = 0
    completed_sequences: set[int] = field(default_factory=set)


class SessionInbox(queue.Queue[IncomingMessage]):
    def __init__(self, register: OperationRegistrar) -> None:
        super().__init__()
        self.register = register

    def put(
        self,
        message: IncomingMessage,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        if message.type == "new_request":
            self.register(message)
        super().put(message, block, timeout)


class SessionScheduler(SimpleScheduler):
    """Opt-in scheduler for persistent hooks, with bounded stage admission."""

    def __init__(
        self,
        session_hooks: SessionHooks,
        *,
        compute_fn: StageCompute | None = None,
        max_open_sessions: int = DEFAULT_MAX_OPEN_SESSIONS,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        max_state_bytes: int = DEFAULT_MAX_STATE_BYTES,
    ) -> None:
        self.session_hooks = session_hooks
        self.request_compute = compute_fn
        self.max_open_sessions = max_open_sessions
        self.max_state_bytes = max_state_bytes
        self.open_sessions: dict[SessionRef, StageSession] = {}
        self.append_cancel_events: dict[str, threading.Event] = {}
        self.session_table_lock = threading.Lock()
        self.is_shutting_down = False
        self.arrivals_by_request_id: dict[str, OperationArrival] = {}
        self.cursors_by_session: dict[SessionRef, SessionOperationCursor] = {}
        self.operation_finished: threading.Condition = threading.Condition(
            self.session_table_lock
        )
        super().__init__(
            self.compute,
            max_concurrency=max_concurrency,
            abort_callback=self.cancel_operation,
            shutdown_callback=self.shutdown_sessions,
        )
        self.inbox = SessionInbox(self.register_operation)

    def register_operation(self, message: IncomingMessage) -> None:
        try:
            session_operation = find_session_operation(message.data.request.metadata)
        except ValueError:
            # Note (Junnan Li): put() runs on the stage loop; compute reports the malformed operation.
            return
        if session_operation is None:
            return
        ref = session_operation.ref
        with self.session_table_lock:
            cursor = self.cursors_by_session.setdefault(ref, SessionOperationCursor())
            self.arrivals_by_request_id[message.request_id] = OperationArrival(
                ref=ref,
                operation=session_operation.operation,
                sequence=cursor.next_sequence,
            )
            cursor.next_sequence += 1

    def finish_operation(self, request_id: str) -> None:
        with self.operation_finished:
            arrival = self.arrivals_by_request_id.pop(request_id, None)
            if arrival is None:
                return
            cursor = self.cursors_by_session.get(arrival.ref)
            if cursor is None:
                return
            # Note (Junnan Li): An aborted operation can finish before its predecessors ran.
            cursor.completed_sequences.add(arrival.sequence)
            while cursor.runnable_sequence in cursor.completed_sequences:
                cursor.completed_sequences.discard(cursor.runnable_sequence)
                cursor.runnable_sequence += 1
            if cursor.runnable_sequence == cursor.next_sequence:
                del self.cursors_by_session[arrival.ref]
            self.operation_finished.notify_all()

    def consume_if_aborted(self, request_id: str) -> bool:
        aborted = super().consume_if_aborted(request_id)
        with self.session_table_lock:
            arrival = self.arrivals_by_request_id.get(request_id)
            is_close_operation = arrival is not None and arrival.operation == "close"
        if aborted and is_close_operation:
            # Note (Junnan Li): A timed-out close is request-aborted; skipping it would leak the state.
            return False
        if aborted:
            self.finish_operation(request_id)
        return aborted

    def compute(self, payload: StagePayload) -> StagePayload:
        session_operation = find_session_operation(payload.request.metadata)
        if session_operation is None:
            if self.request_compute is None:
                raise ValueError("this stage has no compute_fn for ordinary requests")
            return self.request_compute(payload)
        ref = session_operation.ref
        try:
            with self.operation_finished:
                # Note (Junnan Li): A request-level abort may already have consumed the arrival.
                arrival = self.arrivals_by_request_id.get(payload.request_id)
                if arrival is not None:
                    ref = arrival.ref
                    sequence = arrival.sequence
                    self.operation_finished.wait_for(
                        lambda: (
                            (cursor := self.cursors_by_session.get(ref)) is None
                            or cursor.runnable_sequence >= sequence
                        )
                    )
            return self.compute_session(payload, session_operation)
        finally:
            try:
                # Note (Junnan Li): stop skips a session whose hook is running; it is closed here.
                with self.session_table_lock:
                    session = (
                        self.open_sessions.get(ref) if self.is_shutting_down else None
                    )
                if session is not None:
                    with session.lock:
                        self.close_session(ref, session)
            finally:
                self.finish_operation(payload.request_id)

    def cancel_operation(self, request_id: str) -> None:
        with self.session_table_lock:
            cancel_event = self.append_cancel_events.get(request_id)
            if cancel_event is not None:
                cancel_event.set()

    def shutdown_sessions(self) -> None:
        with self.session_table_lock:
            self.is_shutting_down = True
            for cancel_event in self.append_cancel_events.values():
                cancel_event.set()
            open_sessions = list(self.open_sessions.items())
        errors: list[Exception] = []
        for ref, session in open_sessions:
            if session.lock.acquire(blocking=False):
                try:
                    self.close_session(ref, session)
                except Exception as exc:
                    errors.append(exc)
                finally:
                    session.lock.release()
        if errors:
            raise RuntimeError("session shutdown cleanup failed") from errors[0]

    def close_session(self, ref: SessionRef, session: StageSession) -> None:
        if session.is_open:
            self.session_hooks.close(ref)
            session.is_open = False
        with self.session_table_lock:
            self.open_sessions.pop(ref, None)

    def update_usage(self, session: StageSession, ref: SessionRef) -> None:
        usage = self.session_hooks.usage(ref)
        with self.session_table_lock:
            session.usage = usage
            if (
                sum(
                    stage_session.usage.bytes
                    for stage_session in self.open_sessions.values()
                )
                > self.max_state_bytes
            ):
                raise QueueFullError()

    def open_session(self, ref: SessionRef, request: OmniRequest) -> None:
        session = StageSession()
        session.lock.acquire()
        with self.session_table_lock:
            if self.is_shutting_down:
                session.lock.release()
                raise RuntimeError("session scheduler is stopping")
            if ref in self.open_sessions:
                session.lock.release()
                raise ValueError("session already opened")
            if len(self.open_sessions) >= self.max_open_sessions:
                session.lock.release()
                raise QueueFullError()
            self.open_sessions[ref] = session
        try:
            self.session_hooks.open(ref, request)
            session.is_open = True
            self.update_usage(session, ref)
            if self.is_shutting_down:
                raise RuntimeError("session scheduler is stopping")
        except BaseException:
            self.close_session(ref, session)
            raise
        finally:
            session.lock.release()

    def compute_session(
        self, payload: StagePayload, session_operation: SessionOperation
    ) -> StagePayload:
        ref = session_operation.ref
        operation = session_operation.operation
        if operation == "open":
            self.open_session(ref, payload.request)
            payload.data = {"opened": True}
            return payload

        with self.session_table_lock:
            session = self.open_sessions.get(ref)
        if session is None:
            if operation == "close":
                payload.data = {"closed": True}
                return payload
            else:
                raise ValueError("unknown session incarnation")
        with session.lock:
            if operation == "close":
                self.close_session(ref, session)
                payload.data = {"closed": True}
                return payload
            elif self.is_shutting_down:
                raise RuntimeError("session scheduler is stopping")
            else:
                input_chunk = session_operation.chunk
                assert input_chunk is not None, "append operation carries no chunk"
                cancel_event = threading.Event()
                with self.session_table_lock:
                    self.append_cancel_events[payload.request_id] = cancel_event
                with self._abort_lock:
                    if payload.request_id in self._aborted:
                        cancel_event.set()

                def emit(chunk: TimedChunk) -> None:
                    if not cancel_event.is_set():
                        self.outbox.put(
                            OutgoingMessage(
                                request_id=payload.request_id,
                                type="stream",
                                data=chunk.to_dict(),
                                metadata={"modality": chunk.modality},
                            )
                        )

                try:
                    updated_payload = self.session_hooks.append(
                        input_chunk,
                        payload,
                        SessionContext(ref=ref, cancelled=cancel_event, emit=emit),
                    )
                    self.update_usage(session, ref)
                    return updated_payload
                finally:
                    with self.session_table_lock:
                        self.append_cancel_events.pop(payload.request_id, None)
