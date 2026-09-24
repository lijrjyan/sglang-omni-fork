"""Strict direct-WebSocket projection of the shared realtime subset."""

import asyncio
import base64
import binascii
import json
import logging
import uuid
from typing import Literal

from pydantic import ValidationError
from starlette.websockets import WebSocket, WebSocketDisconnect

from sglang_omni.serve.realtime.control import (
    Accepted,
    Cleared,
    Closed,
    Created,
    Drained,
    Ended,
    Failure,
    UnitCompleted,
    Updated,
)
from sglang_omni.serve.realtime.projection import project_control, project_output
from sglang_omni.serve.realtime.runtime import SessionRuntime
from sglang_omni.serve.realtime.schema import (
    CLIENT_EVENT,
    MAX_EVENT_ID_LENGTH,
    AudioAppendEvent,
    JsonObject,
    SessionUpdateEvent,
)
from sglang_omni.serve.realtime.types import ProtocolError

logger = logging.getLogger(__name__)


class SharedRealtimeSession:
    def __init__(self, websocket: WebSocket, runtime: SessionRuntime) -> None:
        self.websocket = websocket
        self.runtime = runtime
        self.session_id = runtime.session_id

    async def run(self) -> None:
        self.runtime.created()
        reader = asyncio.create_task(self.read())
        sender = asyncio.create_task(self.send())
        disconnected = False
        try:
            done, _ = await asyncio.wait(
                (reader, sender), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                try:
                    result = task.result()
                    if task is reader and result == "disconnect":
                        disconnected = True
                    else:
                        pass
                except WebSocketDisconnect:
                    disconnected = True
        finally:
            await self.runtime.close("disconnect")
            if not disconnected and not sender.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(sender), self.runtime.limits.cleanup_timeout_s
                    )
                except (asyncio.TimeoutError, WebSocketDisconnect):
                    pass
            else:
                pass
            for task in (reader, sender):
                task.cancel()
            await asyncio.gather(reader, sender, return_exceptions=True)

    async def teardown(self) -> None:
        await self.runtime.close("disconnect")

    async def send(self) -> None:
        async for envelope in self.runtime.outputs():
            if isinstance(
                envelope.event,
                (
                    Accepted,
                    Cleared,
                    Closed,
                    Created,
                    Drained,
                    Ended,
                    Failure,
                    UnitCompleted,
                    Updated,
                ),
            ):
                event = project_control(envelope.event)
            else:
                event = project_output(
                    envelope.event,
                    output_modalities=(
                        list(envelope.output_modalities)
                        if envelope.output_modalities is not None
                        else None
                    ),
                )
            event["event_id"] = "evt_" + uuid.uuid4().hex
            if envelope.unit is not None:
                unit = envelope.unit
                metadata = event.setdefault("sglang", {})
                metadata.update(
                    {
                        "unit_id": f"unit_{unit.seq}",
                        "chunk_seq": envelope.chunk_seq,
                        "media_time": dict(
                            t_start_ms=self.runtime.ms(unit.start_sample),
                            duration_ms=self.runtime.ms(unit.real_samples),
                        ),
                    }
                )
            else:
                pass
            self.runtime.output_buffer.before_send(envelope)
            await self.websocket.send_text(json.dumps(event, allow_nan=False))
            self.runtime.output_buffer.sent(envelope)
        await self.websocket.close()

    async def read(self) -> Literal["disconnect"] | None:
        while self.runtime.state != "CLOSED":
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                return "disconnect"
            else:
                pass
            event_id = None
            try:
                if message.get("bytes") is not None:
                    raise ProtocolError(
                        "invalid_request", "binary frames are unsupported"
                    )
                else:
                    pass
                try:
                    raw = json.loads(
                        message.get("text", ""),
                        parse_constant=lambda _: (_ for _ in ()).throw(
                            ValueError("nonfinite JSON number")
                        ),
                    )
                except (ValueError, TypeError) as exc:
                    raise ProtocolError("invalid_request", "invalid JSON") from exc
                event_id = raw.get("event_id") if isinstance(raw, dict) else None
                if (
                    not isinstance(event_id, str)
                    or not 0 < len(event_id) <= MAX_EVENT_ID_LENGTH
                ):
                    event_id = None
                else:
                    pass
                await self.dispatch(raw)
            except ProtocolError as exc:
                try:
                    self.runtime.notify(
                        Failure(exc.code, str(exc), False, event_id, exc.param)
                    )
                except RuntimeError:
                    self.runtime.fail("outbound event budget exhausted")
                    return None
            except Exception as exc:
                logger.exception(f"Realtime session {self.session_id} dispatch failed")
                self.runtime.fail(str(exc), event_id=event_id)
                return None

        return None

    async def dispatch(self, raw: JsonObject) -> None:
        try:
            event = CLIENT_EVENT.validate_python(raw)
        except ValidationError as exc:
            error = exc.errors()[0]
            code = (
                "not_supported"
                if error["type"] == "union_tag_invalid"
                else "invalid_request"
            )
            raise ProtocolError(
                code, error["msg"], ".".join(str(part) for part in error["loc"])
            ) from exc
        if isinstance(event, SessionUpdateEvent):
            await self.runtime.update(event.session, event.event_id)
        elif isinstance(event, AudioAppendEvent):
            if len(event.audio) > (self.runtime.limits.max_input_bytes + 2) // 3 * 4:
                raise ProtocolError(
                    "buffer_overflow", "encoded audio exceeds input budget"
                )
            else:
                pass
            try:
                pcm = base64.b64decode(event.audio, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ProtocolError(
                    "invalid_request", "invalid base64 audio", "audio"
                ) from exc
            await self.runtime.append(
                pcm, event.sglang.seq, event.sglang.t_start_ms, event.event_id
            )
        elif event.type == "input_audio_buffer.clear":
            await self.runtime.clear(event.event_id)
        elif event.type == "sglang.input_audio.end":
            await self.runtime.end(event.event_id)
        elif event.type == "session.close":
            await self.runtime.close("client_closed", event.event_id)
        else:
            self.runtime.require_open()
            raise ProtocolError(
                "not_applicable", "manual turn commands are not granted"
            )
