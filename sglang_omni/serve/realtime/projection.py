"""Wire projections; computation emits raw PCM and concrete event values."""

import base64
from dataclasses import asdict
from typing import cast

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
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    AudioFinished,
    OutputEvent,
    ResponseFinished,
    ResponseStarted,
    TextDelta,
    TextFinished,
    TurnFailure,
)
from sglang_omni.serve.realtime.schema import JsonObject, JsonValue


def project_output(
    event: OutputEvent,
    *,
    output_modalities: list[str] | None = None,
) -> JsonObject:
    if isinstance(event, ResponseStarted):
        return dict(
            type="response.created",
            response=dict(
                id=event.response_id,
                object="realtime.response",
                status="in_progress",
                output=[],
            ),
        )
    elif isinstance(event, ResponseFinished):
        content: list[JsonValue] = [dict(type="output_text", text=event.text)]
        if output_modalities is not None and "text" not in output_modalities:
            content = []
        if event.include_audio:
            content.append(dict(type="output_audio", transcript=event.text))
        return dict(
            type="response.done",
            response=dict(
                id=event.response_id,
                object="realtime.response",
                status=event.status,
                status_details=dict(reason=event.reason),
                output=[
                    dict(
                        id=event.item_id,
                        object="realtime.item",
                        type="message",
                        role="assistant",
                        content=content,
                    )
                ],
                usage=cast(JsonValue, event.usage),
            ),
        )
    elif isinstance(event, (TextDelta, TextFinished, AudioDelta, AudioFinished)):
        audio = isinstance(event, (AudioDelta, AudioFinished))
        done = isinstance(event, (TextFinished, AudioFinished))
        if audio:
            name = "output_audio"
        elif output_modalities == ["audio"]:
            name = "output_audio_transcript"
        else:
            name = "output_text"
        result: JsonObject = dict(
            type=f'response.{name}.{"done" if done else "delta"}',
            response_id=event.response_id,
            item_id=event.item_id,
            output_index=0,
            content_index=0,
        )
        if isinstance(event, AudioDelta):
            result["delta"] = base64.b64encode(event.pcm).decode("ascii")
        elif isinstance(event, (TextDelta, TextFinished)):
            result[
                (
                    ("transcript" if name == "output_audio_transcript" else "text")
                    if done
                    else "delta"
                )
            ] = event.text
        return result
    elif isinstance(event, TurnFailure):
        return dict(
            type="error",
            error=dict(type=event.type, code=event.code, message=event.message),
        )
    else:
        raise TypeError(f"Unsupported typed output: {type(event)}")


def project_control(event: ControlEvent) -> JsonObject:

    if isinstance(event, UnitCompleted):
        return dict(type="sglang.unit.done", unit_id=event.unit_id)
    elif isinstance(event, Created):
        return dict(
            type="session.created",
            session=dict(
                id=event.session_id,
                object="realtime.session",
                type=event.session_type,
                model=event.model,
                sglang=dict(granted=None),
            ),
        )
    elif isinstance(event, Updated):
        return dict(
            type="session.updated",
            client_event_id=event.client_event_id,
            session={
                **cast(JsonObject, event.config or {}),
                "id": event.session_id,
                "object": "realtime.session",
                "model": event.model,
                "type": event.session_type,
                "sglang": {
                    **cast(JsonObject, (event.config or {}).get("sglang", {})),
                    "granted": cast(JsonObject, event.granted),
                },
            },
        )
    elif isinstance(event, Failure):
        return dict(
            type="error",
            sglang=dict(fatal=event.fatal),
            error=dict(
                type="server_error" if event.fatal else "invalid_request_error",
                code=event.code,
                message=event.message,
                event_id=event.client_event_id,
                param=event.param,
            ),
        )
    elif isinstance(event, Closed):
        return dict(
            type="session.closed",
            reason=event.reason,
            client_event_id=event.client_event_id,
        )
    else:
        names = {
            Accepted: "sglang.input_audio.accepted",
            Cleared: "input_audio_buffer.cleared",
            Ended: "sglang.input_audio.ended",
            Drained: "sglang.input_audio.drained",
        }
        result = dict(type=names[type(event)], **asdict(event))
        if isinstance(event, Cleared):
            result["sglang"] = dict(discarded_ms=result.pop("discarded_ms"))
        return result
