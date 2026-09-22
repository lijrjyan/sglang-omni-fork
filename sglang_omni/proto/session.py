# SPDX-License-Identifier: Apache-2.0
"""Model-independent, bounded session command and output contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import msgpack
import msgspec

SESSION_METADATA_KEY = "omni_session"
# Note (Junnan Li): msgspec encodes bytes as base64 text by default; keep them native on both sides.
BUILTIN_TYPES = (bytes,)
SessionOp = Literal["open", "append", "close"]
DEFAULT_MAX_MODALITIES = 8
DEFAULT_MAX_PENDING_CHUNKS = 16
DEFAULT_MAX_PENDING_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_OUTPUT_CHUNKS = 64
DEFAULT_MAX_OUTPUT_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_CHUNK_BYTES = 1024 * 1024
DEFAULT_COMMAND_TIMEOUT_S = 30.0
DEFAULT_IDLE_TIMEOUT_S = 300.0
# Note (Junnan Li): Msgpack bin headers grow by 1 byte at 256 bytes and 3 bytes at 65536.
MSGPACK_BIN8_LIMIT = 256
MSGPACK_BIN16_LIMIT = 65536
MSGPACK_BIN16_HEADER_GROWTH = 1
MSGPACK_BIN32_HEADER_GROWTH = 3


@dataclass(frozen=True)
class SessionRef:
    session_id: str
    incarnation: int = 1


@dataclass(frozen=True)
class TimedChunk:
    """Input seq is global across modalities within a session incarnation."""

    modality: str
    t_start_ms: float
    duration_ms: float
    seq: int
    payload: bytes | dict[str, Any] | None
    format: str | None = None
    eos: bool = False

    def to_dict(self) -> dict[str, Any]:
        return msgspec.to_builtins(self, builtin_types=BUILTIN_TYPES)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimedChunk:
        return msgspec.convert(data, type=cls, strict=True, builtin_types=BUILTIN_TYPES)


@dataclass(frozen=True)
class OutputChunk:
    """input_seq identifies the originating pipeline input, not a stream seq."""

    ref: SessionRef
    seq: int
    input_seq: int
    modality: str
    t_start_ms: float
    duration_ms: float
    payload: bytes | dict[str, Any] | None
    format: str | None = None
    eos: bool = False
    kind: Literal["data", "input_done"] = "data"

    def to_dict(self) -> dict[str, Any]:
        return msgspec.to_builtins(self, builtin_types=BUILTIN_TYPES)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutputChunk:
        return msgspec.convert(data, type=cls, strict=True, builtin_types=BUILTIN_TYPES)


@dataclass(frozen=True)
class ResourceUsage:
    kv_tokens: int = 0
    slots: dict[str, int] = field(default_factory=dict)
    bytes: int = 0


@dataclass(frozen=True)
class SessionLimits:
    max_modalities: int = DEFAULT_MAX_MODALITIES
    max_pending_chunks: int = DEFAULT_MAX_PENDING_CHUNKS
    max_pending_bytes: int = DEFAULT_MAX_PENDING_BYTES
    max_output_chunks: int = DEFAULT_MAX_OUTPUT_CHUNKS
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES
    command_timeout_s: float = DEFAULT_COMMAND_TIMEOUT_S
    idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S


@dataclass(frozen=True)
class SessionCommand:
    """Coordinator-to-stage session command, carried in request metadata."""

    op: SessionOp
    ref: SessionRef
    stages: tuple[str, ...]
    chunk: TimedChunk | None = None

    def to_dict(self) -> dict[str, Any]:
        return msgspec.to_builtins(self, builtin_types=BUILTIN_TYPES)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionCommand:
        return msgspec.convert(data, type=cls, strict=True, builtin_types=BUILTIN_TYPES)


def find_session_command(metadata: dict[str, Any]) -> SessionCommand | None:
    """Return the command in request metadata, or None for an ordinary request."""
    command_fields = metadata.get(SESSION_METADATA_KEY)
    if command_fields is None:
        return None
    return SessionCommand.from_dict(command_fields)


def wire_size(chunk_fields: dict[str, Any]) -> int:
    """Return the msgpack wire size of a chunk dict without copying a binary payload."""
    payload = chunk_fields["payload"]
    if isinstance(payload, bytes):
        payload_size = len(payload)
        if payload_size < MSGPACK_BIN8_LIMIT:
            header_growth = 0
        elif payload_size < MSGPACK_BIN16_LIMIT:
            header_growth = MSGPACK_BIN16_HEADER_GROWTH
        else:
            header_growth = MSGPACK_BIN32_HEADER_GROWTH
        packed_without_payload = msgpack.packb(
            {**chunk_fields, "payload": b""}, use_bin_type=True
        )
        return len(packed_without_payload) + payload_size + header_growth
    else:
        return len(msgpack.packb(chunk_fields, use_bin_type=True))
