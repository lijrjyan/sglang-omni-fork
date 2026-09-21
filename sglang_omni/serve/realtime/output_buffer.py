"""Realtime output delivery and completion state."""

import asyncio
from collections import deque
from dataclasses import dataclass, replace

from sglang_omni.serve.realtime.control import Closed, Failure
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    AudioFinished,
    ContextLimitError,
    OutputEvent,
    ResponseEvent,
    ResponseFinished,
    ResponseStarted,
    ResponseStatus,
    TextDelta,
    TextFinished,
)
from sglang_omni.serve.realtime.types import Envelope, RuntimeLimits, Unit


@dataclass(kw_only=True)
class ResponseState:
    output_modalities: tuple[str, ...]
    terminal: bool
    visible: bool
    terminal_sent: bool
    text_done_sent: bool
    audio_done_sent: bool
    audio_visible: bool
    item_id: str
    text: str
    audio: bool


class OutputBuffer:
    def __init__(self, limits: RuntimeLimits) -> None:
        self.limits = limits
        self.output: deque[tuple[Envelope, int]] = deque()
        self.output_bytes = 0
        self.output_wake = asyncio.Event()
        self.output_seq = 0
        self.responses: dict[str, ResponseState] = {}

    def enqueue(self, envelope: Envelope) -> None:
        size = len(repr(envelope).encode())
        if (
            len(self.output) >= self.limits.max_output_events
            or self.output_bytes + size > self.limits.max_output_bytes
        ):
            raise RuntimeError("outbound event budget exhausted")
        self.output.append((envelope, size))
        self.output_bytes += size
        self.output_wake.set()

    def emit(
        self,
        event: OutputEvent,
        unit: Unit | None,
        output_modalities: tuple[str, ...],
    ) -> None:
        rid = event.response_id if isinstance(event, ResponseEvent) else None
        # Note (Junnan Li): Keep each response's negotiated modalities across hot updates.
        response = self.responses.get(rid) if rid is not None else None
        modalities = (
            response.output_modalities if response is not None else None
        ) or output_modalities
        if isinstance(event, (AudioDelta, AudioFinished)) and "audio" not in modalities:
            return
        if isinstance(event, ResponseFinished) and "audio" not in modalities:
            event = replace(event, include_audio=False)
        if isinstance(event, ResponseStarted):
            if rid in self.responses:
                raise RuntimeError("duplicate response creation")
            self.responses[event.response_id] = ResponseState(
                output_modalities=modalities,
                terminal=False,
                visible=False,
                terminal_sent=False,
                text_done_sent=False,
                audio_done_sent=False,
                audio_visible=False,
                item_id="",
                text="",
                audio=False,
            )
        elif isinstance(
            event,
            (TextDelta, TextFinished, AudioDelta, AudioFinished, ResponseFinished),
        ):
            state = self.responses.get(event.response_id)
            if state is None:
                raise RuntimeError("response output precedes creation")
            if state.terminal:
                return
            if state.item_id and event.item_id != state.item_id:
                raise RuntimeError("only one message item per response is supported")
            state.item_id = event.item_id
            if isinstance(event, (TextDelta, TextFinished)):
                state.text = (
                    state.text + event.text
                    if isinstance(event, TextDelta)
                    else event.text
                )
                if len(state.text) > self.limits.max_history_chars:
                    raise ContextLimitError("response text context limit")
            if isinstance(event, AudioDelta):
                if len(event.pcm) % 2:
                    raise RuntimeError("producer emitted invalid PCM16")
                state.audio = True
            if isinstance(event, ResponseFinished):
                state.terminal = True
        self.enqueue(
            Envelope(
                event,
                unit=unit,
                chunk_seq=self.output_seq,
                output_modalities=tuple(modalities),
            )
        )
        self.output_seq += 1

    def finish_responses(self, status: ResponseStatus, reason: str) -> None:
        for rid, state in list(self.responses.items()):
            if state.terminal_sent:
                continue
            elif not state.visible:
                del self.responses[rid]
                continue
            state.terminal = True
            item_id = state.item_id or "item_" + rid
            events: list[OutputEvent] = []
            if not state.text_done_sent and state.output_modalities:
                events.append(TextFinished(rid, item_id, state.text))
            if state.audio_visible and not state.audio_done_sent:
                events.append(AudioFinished(rid, item_id))
            events.append(
                ResponseFinished(
                    rid, item_id, state.text, state.audio_visible, status, reason
                )
            )
            for event in events:
                envelope = Envelope(
                    event,
                    True,
                    output_modalities=tuple(state.output_modalities),
                )
                size = len(repr(envelope).encode())
                self.output.append((envelope, size))
                self.output_bytes += size
            self.output_wake.set()

    def before_send(self, envelope: Envelope) -> None:
        """Close must observe lifecycle visibility before the socket send yields."""
        event = envelope.event
        if isinstance(event, ResponseStarted):
            self.responses[event.response_id].visible = True
        elif isinstance(event, ResponseFinished):
            self.responses[event.response_id].terminal_sent = True
        elif isinstance(event, TextFinished):
            self.responses[event.response_id].text_done_sent = True
        elif isinstance(event, AudioFinished):
            self.responses[event.response_id].audio_done_sent = True
        if isinstance(event, AudioDelta):
            self.responses[event.response_id].audio_visible = True

    def sent(self, envelope: Envelope) -> None:
        event = envelope.event
        if isinstance(event, ResponseFinished):
            del self.responses[event.response_id]
        elif isinstance(event, Closed):
            self.responses.clear()

    def terminal(self, event: Failure | Closed) -> None:
        # Note (Junnan Li): Terminal notifications must remain deliverable after media overflow.
        terminals = [
            (env, size)
            for env, size in self.output
            if isinstance(env.event, Failure)
            or (
                env.control
                and isinstance(
                    env.event, (TextFinished, AudioFinished, ResponseFinished)
                )
            )
        ]
        self.output = deque(terminals)
        self.output_bytes = sum(size for _, size in self.output)
        envelope = Envelope(event, True)
        size = len(repr(envelope).encode())
        self.output.append((envelope, size))
        self.output_bytes += size
        self.output_wake.set()
