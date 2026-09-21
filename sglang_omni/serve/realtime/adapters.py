"""Interaction bridges. No transport parsing or model-name dispatch."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable

from sglang_omni.client.client import Client
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import OutputChunk, SessionLimits, SessionRef, TimedChunk
from sglang_omni.serve.realtime.output import OutputEvent, TurnFailure
from sglang_omni.serve.realtime.schema import SessionConfiguration
from sglang_omni.serve.realtime.task_cleanup import cancel_local_tasks
from sglang_omni.serve.realtime.types import (
    DEFAULT_AUDIO_RATE,
    InteractionAdapter,
    OutputSink,
    RuntimeLimits,
    Unit,
)

logger = logging.getLogger(__name__)


class CoordinatorAdapter(InteractionAdapter):
    """Publish output after unit success; cancel preserves its consumption receipt."""

    def __init__(
        self,
        client: Client,
        *,
        stages: list[str],
        request_builder: Callable[[SessionConfiguration], OmniRequest],
        output_converter: Callable[[OutputChunk], Iterable[OutputEvent]],
        input_rate: int = DEFAULT_AUDIO_RATE,
        atomic_consumption: bool = False,
        limits: SessionLimits | None = None,
    ) -> None:
        if not atomic_consumption:
            raise ValueError("a producer atomic-consumption contract is required")
        self.client = client
        self.stages = stages
        self.request_builder = request_builder
        self.convert = output_converter
        self.rate = input_rate
        self.limits = limits or SessionLimits()
        self.local_cleanup_timeout = self.limits.command_timeout_s
        self.ref: SessionRef | None = None
        self.reader: asyncio.Task[None] | None = None
        self.active: Unit | None = None
        self.future: asyncio.Future[int] | None = None
        self.buffer: list[OutputEvent] = []
        self.buffer_bytes = 0
        self.emit: OutputSink | None = None
        self.epoch = 0
        self.reader_error: Exception | None = None
        self.closing = False

    def set_limits(self, limits: RuntimeLimits) -> None:
        self.local_cleanup_timeout = limits.cleanup_timeout_s

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        self.emit = emit
        self.ref = await self.client.open_session(
            self.request_builder(config),
            stages=self.stages,
            limits=self.limits,
            session_id=session_id,
        )
        self.reader = asyncio.create_task(self.read())

    async def read(self) -> None:
        assert self.ref is not None and self.emit is not None
        try:
            # Note (Junnan Li): Keep the output iterator across abort; closing it closes the coordinator session.
            async for output in self.client.session_outputs(self.ref):
                if self.active is None or output.input_seq != self.active.seq:
                    continue
                if output.kind == "input_done":
                    if output.ref.epoch == self.epoch:
                        for event in self.buffer:
                            await self.emit(event, self.epoch, self.active)
                    self.buffer.clear()
                    self.buffer_bytes = 0
                    if self.future is not None and not self.future.done():
                        self.future.set_result(self.active.real_samples)
                elif output.ref.epoch == self.epoch:
                    for event in self.convert(output):
                        size = len(repr(event).encode())
                        if (
                            len(self.buffer) >= self.limits.max_output_chunks
                            or self.buffer_bytes + size > self.limits.max_output_bytes
                        ):
                            raise RuntimeError("native unit output budget exhausted")
                        self.buffer.append(event)
                        self.buffer_bytes += size
            if not self.closing:
                raise RuntimeError("session output stream closed")
        except Exception as exc:
            logger.exception("Realtime session output reader failed")
            self.reader_error = exc
            if self.future is not None and not self.future.done():
                self.future.set_exception(exc)
            else:
                await self.emit(
                    TurnFailure("server_error", "internal", str(exc)), self.epoch
                )

    async def process(self, unit: Unit, epoch: int) -> int:
        assert self.ref is not None
        if self.reader_error is not None:
            raise self.reader_error
        self.active = unit
        self.epoch = epoch
        self.future = asyncio.get_running_loop().create_future()
        chunk = TimedChunk(
            "audio",
            unit.start_sample * 1000 / self.rate,
            unit.real_samples * 1000 / self.rate,
            unit.seq,
            unit.pcm,
            format="pcm16",
            eos=unit.eos,
        )
        try:
            await self.client.append_session(self.ref, chunk)
            return await self.future
        finally:
            self.active = None
            self.future = None
            self.buffer.clear()
            self.buffer_bytes = 0

    async def cancel(self) -> None:
        assert self.ref is not None
        # Note (Junnan Li): The coordinator still completes the unit and preserves its receipt.
        future = self.future
        self.ref = await self.client.abort_session(self.ref)
        if future is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(future), self.local_cleanup_timeout
                )
            except asyncio.TimeoutError as exc:
                raise RuntimeError(
                    "completed unit consumption receipt is missing"
                ) from exc
        self.epoch = self.ref.epoch
        self.buffer.clear()
        self.buffer_bytes = 0

    async def close(self) -> None:
        self.closing = True
        try:
            if self.ref is not None:
                await self.client.close_session(self.ref)
        finally:
            await cancel_local_tasks([self.reader], self.local_cleanup_timeout)
