"""Realtime unit identity, adapter contracts and resource configuration."""

from __future__ import annotations

import math
from collections.abc import Awaitable
from dataclasses import asdict, dataclass
from typing import Protocol

from sglang_omni.serve.realtime.control import ControlEvent
from sglang_omni.serve.realtime.output import OutputEvent
from sglang_omni.serve.realtime.schema import (
    GrantedCapabilities,
    Interaction,
    PartialStyle,
    SessionConfiguration,
    TailPolicy,
)


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str, param: str | None = None) -> None:
        self.code = code
        self.param = param
        super().__init__(message)


DEFAULT_AUDIO_RATE = 16000
PCM16_BYTES_PER_SAMPLE = 2


@dataclass(frozen=True)
class RuntimeLimits:
    max_input_bytes: int = 60 * DEFAULT_AUDIO_RATE * PCM16_BYTES_PER_SAMPLE
    max_output_bytes: int = 4 * 1024 * 1024
    max_output_events: int = 256
    max_history_chars: int = 64 * 1024
    cleanup_timeout_s: float = 30

    def __post_init__(self) -> None:
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("runtime limits must be finite and positive")


@dataclass(frozen=True)
class Capabilities:
    interaction: Interaction = "native"
    input_rate: int = DEFAULT_AUDIO_RATE
    output_rate: int = DEFAULT_AUDIO_RATE
    output_modalities: tuple[str, ...] = ("text",)
    native_unit_ms: int = 20
    tail_policy: TailPolicy = "flush"
    cancel_is_noop: bool = False
    partial_style: PartialStyle = "append_only"

    def __post_init__(self) -> None:
        if self.interaction != "native":
            raise ValueError("unsupported interaction")
        if self.input_rate <= 0 or self.output_rate <= 0 or self.native_unit_ms <= 0:
            raise ValueError("positive rates and cadence required")
        if self.input_rate * self.native_unit_ms % 1000:
            raise ValueError("native cadence must contain whole samples")
        if self.tail_policy not in ("flush", "pad", "reject"):
            raise ValueError("unsupported tail policy")
        if not self.output_modalities or set(self.output_modalities) - {
            "text",
            "audio",
        }:
            raise ValueError("unsupported output modalities")

    def describe(self) -> GrantedCapabilities:
        return dict(
            interaction=self.interaction,
            native_full_duplex=self.interaction == "native",
            proactive_output=False,
            turn_control=[None],
            client_commit=False,
            input_modalities=["audio"],
            output_modalities=list(self.output_modalities),
            input_audio_format=dict(type="audio/pcm", rate=self.input_rate),
            output_audio_format=dict(type="audio/pcm", rate=self.output_rate),
            native_unit_ms=self.native_unit_ms,
            first_unit_ms=self.native_unit_ms,
            microturn_ms="variable",
            tail_policy=self.tail_policy,
            supports_server_interrupt=False,
            cancel_is_noop=self.cancel_is_noop,
            supports_truncate=False,
            supports_resume=False,
            partial_style=self.partial_style,
            pressure_policy="reject",
            strict_order=True,
        )


@dataclass(frozen=True)
class Unit:
    seq: int
    start_sample: int
    pcm: bytes
    real_samples: int
    eos: bool = False
    output_modalities: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Envelope:
    epoch: int
    event: OutputEvent | ControlEvent
    # Note (Junnan Li): Control acknowledgements and response terminals survive an epoch advance.
    control: bool = False
    unit: Unit | None = None
    chunk_seq: int = 0
    output_modalities: tuple[str, ...] | None = None


class OutputSink(Protocol):
    def __call__(
        self, event: OutputEvent, epoch: int, unit: Unit | None = None
    ) -> Awaitable[None]: ...


class InteractionAdapter:
    """Adapters override what they support; the runtime never probes for methods."""

    def set_limits(self, limits: RuntimeLimits) -> None:
        pass

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        raise NotImplementedError

    async def process(self, unit: Unit, epoch: int) -> int | tuple[int, int]:
        raise NotImplementedError

    async def clear(self) -> int:
        return 0

    async def cancel(self) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError
