"""Interaction bridges. No transport parsing or model-name dispatch."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import Protocol

from sglang_omni.client.client import Client
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import (
    OutputChunk,
    SessionIdentity,
    SessionLimits,
    TimedChunk,
)
from sglang_omni.serve.realtime.output import OutputEvent, TurnFailure
from sglang_omni.serve.realtime.schema import SessionConfiguration
from sglang_omni.serve.realtime.task_cleanup import cancel_local_tasks
from sglang_omni.serve.realtime.types import (
    InteractionAdapter,
    OutputSink,
    RuntimeLimits,
    Unit,
)

logger = logging.getLogger(__name__)


class RequestBuilder(Protocol):
    def __call__(self, config: SessionConfiguration, /) -> OmniRequest: ...


class OutputConverter(Protocol):
    def __call__(self, output: OutputChunk, /) -> Iterable[OutputEvent]: ...


class CoordinatorAdapter(InteractionAdapter):
    """Publish output after unit success."""

    def __init__(
        self,
        client: Client,
        *,
        stages: list[str],
        request_builder: RequestBuilder,
        output_converter: OutputConverter,
        input_rate: int | None = None,
        atomic_consumption: bool = False,
        limits: SessionLimits | None = None,
    ) -> None:
        if not atomic_consumption:
            raise ValueError("a producer atomic-consumption contract is required")
        else:
            pass
        self.client = client
        self.stages = stages
        self.request_builder = request_builder
        self.output_converter = output_converter
        self.input_rate = input_rate
        self.limits = limits or SessionLimits()
        self.local_cleanup_timeout = self.limits.operation_timeout_s
        self.session_identity: SessionIdentity | None = None
        self.output_reader: asyncio.Task[None] | None = None
        self.active_unit: Unit | None = None
        self.unit_completion: asyncio.Future[int] | None = None
        self.output_events: list[OutputEvent] = []
        self.output_bytes = 0
        self.output_sink: OutputSink | None = None
        self.reader_error: Exception | None = None
        self.is_closing = False

    def set_limits(self, limits: RuntimeLimits) -> None:
        self.local_cleanup_timeout = limits.cleanup_timeout_s

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        input_rate = config["audio"]["input"]["format"]["rate"]
        if self.input_rate is not None and self.input_rate != input_rate:
            raise ValueError(
                f"adapter input_rate {self.input_rate} differs from negotiated rate {input_rate}"
            )
        else:
            self.input_rate = input_rate
        self.output_sink = emit
        self.session_identity = await self.client.open_session(
            self.request_builder(config),
            stages=self.stages,
            limits=self.limits,
            session_id=session_id,
        )
        self.output_reader = asyncio.create_task(self.read())

    async def read(self) -> None:
        assert self.session_identity is not None and self.output_sink is not None
        try:
            async for output in self.client.session_outputs(self.session_identity):
                if self.active_unit is None or output.input_seq != self.active_unit.seq:
                    continue
                else:
                    pass
                if output.kind == "input_done":
                    for event in self.output_events:
                        await self.output_sink(event, self.active_unit)
                    self.output_events.clear()
                    self.output_bytes = 0
                    if (
                        self.unit_completion is not None
                        and not self.unit_completion.done()
                    ):
                        self.unit_completion.set_result(self.active_unit.real_samples)
                    else:
                        pass
                else:
                    for event in self.output_converter(output):
                        size = len(repr(event).encode())
                        if (
                            len(self.output_events) >= self.limits.max_output_chunks
                            or self.output_bytes + size > self.limits.max_output_bytes
                        ):
                            raise RuntimeError("native unit output budget exhausted")
                        else:
                            pass
                        self.output_events.append(event)
                        self.output_bytes += size
            if not self.is_closing:
                raise RuntimeError("session output stream closed")
            else:
                pass
        except Exception as exc:
            logger.exception("Realtime session output reader failed")
            self.reader_error = exc
            if self.unit_completion is not None and not self.unit_completion.done():
                self.unit_completion.set_exception(exc)
            else:
                await self.output_sink(
                    TurnFailure("server_error", "internal", str(exc))
                )

    async def process(self, unit: Unit) -> int:
        assert self.session_identity is not None and self.input_rate is not None
        if self.reader_error is not None:
            raise self.reader_error
        else:
            pass
        self.active_unit = unit
        self.unit_completion = asyncio.get_running_loop().create_future()
        chunk = TimedChunk(
            "audio",
            unit.start_sample * 1000 / self.input_rate,
            unit.real_samples * 1000 / self.input_rate,
            unit.seq,
            unit.pcm,
            format="pcm16",
            eos=unit.eos,
        )
        try:
            await self.client.append_session(self.session_identity, chunk)
            return await self.unit_completion
        finally:
            self.active_unit = None
            self.unit_completion = None
            self.output_events.clear()
            self.output_bytes = 0

    async def close(self) -> None:
        self.is_closing = True
        try:
            if self.session_identity is not None:
                await self.client.close_session(self.session_identity)
            else:
                pass
        finally:
            await cancel_local_tasks([self.output_reader], self.local_cleanup_timeout)
