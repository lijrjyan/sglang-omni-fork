# SPDX-License-Identifier: Apache-2.0
"""Shared audio preparation for ASR request builders.

Resolves the audio source from the ``StagePayload``, decodes/resamples it to
the model's sample rate, then derives the clip duration and cache fingerprint.

Low-level mechanics (decode, load, resample, fingerprint) stay in
``sglang_omni.utils.audio``.

Model-specific: ``source_name`` used in error messages, duration limit where
the model has one, and the custom ``source_resolver`` when the model accepts
sources beyond the default payload keys (e.g. MOSS-Transcribe-Diarize).
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from sglang_omni.utils.audio import audio_fingerprint, audio_fingerprint_int, load_audio

if TYPE_CHECKING:
    from sglang_omni.proto import StagePayload

DEFAULT_TARGET_SAMPLE_RATE = 16000

# Byte-like sources take precedence over path-like sources; within each group,
# order is the lookup precedence.
_BYTES_SOURCE_KEYS = ("audio_bytes", "bytes", "file")
_PATH_SOURCE_KEYS = ("audio_path", "path", "url")


def resolve_audio_source(payload: StagePayload) -> Any:
    """Default source resolver shared by the ASR request builders."""
    inputs = payload.request.inputs
    if isinstance(inputs, dict):
        for key in _BYTES_SOURCE_KEYS:
            value = inputs.get(key)
            if value is not None:
                return value
        for key in _PATH_SOURCE_KEYS:
            value = inputs.get(key)
            if value is not None:
                return value
    return inputs


@dataclass(frozen=True)
class PreparedAudio:
    """Decoded waveform plus the derived per-request audio metadata."""

    waveform: np.ndarray
    sample_rate: int
    duration_s: float
    fingerprint: str

    @property
    def fingerprint_int(self) -> int:
        return audio_fingerprint_int(self.fingerprint)


@dataclass(frozen=True)
class Chunk:
    """Half-open sample range for one transcription window."""

    start_sample: int
    end_sample: int

    @property
    def num_samples(self) -> int:
        return self.end_sample - self.start_sample


def detect_silence_boundaries(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    min_silence_s: float = 0.3,
    frame_duration_s: float = 0.02,
    relative_threshold: float = 0.1,
) -> list[int]:
    """Return sample midpoints of internal low-energy runs.

    This offline detector intentionally avoids the realtime Silero VAD's model
    lifecycle. It provides deterministic silence candidates; ``plan_chunks``
    still enforces the hard maximum when no candidate is usable.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    frame_samples = round(frame_duration_s * sample_rate)
    min_silence_samples = round(min_silence_s * sample_rate)
    if frame_samples <= 0 or min_silence_samples <= 0:
        raise ValueError("silence durations must produce at least one sample")
    if not 0.0 <= relative_threshold < 1.0:
        raise ValueError("relative_threshold must be in [0, 1)")
    if len(waveform) < frame_samples * 3:
        return []

    rms: list[float] = []
    for start in range(0, len(waveform), frame_samples):
        frame = waveform[start : start + frame_samples]
        rms.append(float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))))
    peak = max(rms, default=0.0)
    if peak == 0.0:
        return []
    silent = [value <= peak * relative_threshold for value in rms]

    boundaries: list[int] = []
    run_start: int | None = None
    for index, is_silent in enumerate([*silent, False]):
        if is_silent and run_start is None:
            run_start = index
            continue
        if is_silent or run_start is None:
            continue
        run_end = index
        start_sample = run_start * frame_samples
        end_sample = min(run_end * frame_samples, len(waveform))
        is_internal = run_start > 0 and run_end < len(silent)
        if is_internal and end_sample - start_sample >= min_silence_samples:
            boundaries.append((start_sample + end_sample) // 2)
        run_start = None
    return boundaries


def plan_chunks(
    num_samples: int,
    sample_rate: int,
    max_window_s: float,
    vad_boundaries: Sequence[int],
    *,
    boundary_search_s: float = 5.0,
) -> list[Chunk]:
    """Plan non-overlapping windows, preferring nearby silence before each limit."""

    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    max_samples = round(max_window_s * sample_rate)
    if max_samples <= 0:
        raise ValueError("max_window_s must produce at least one sample")
    if not math.isfinite(boundary_search_s) or boundary_search_s < 0:
        raise ValueError("boundary_search_s must be a finite non-negative number")
    # note (Junnan Li): The 5-second default keeps silence cuts near the hard
    # limit; fall back to the hard limit instead of overlapping when none exists.
    search_samples = min(
        round(boundary_search_s * sample_rate),
        max_samples // 2,
    )
    if num_samples == 0:
        return []
    if num_samples <= max_samples:
        return [Chunk(0, num_samples)]

    boundaries = sorted(
        {int(boundary) for boundary in vad_boundaries if 0 < boundary < num_samples}
    )
    chunks: list[Chunk] = []
    start = 0
    while start < num_samples:
        hard_end = min(start + max_samples, num_samples)
        if hard_end == num_samples:
            end = hard_end
        else:
            search_start = hard_end - search_samples
            candidates = [
                boundary
                for boundary in boundaries
                if search_start <= boundary <= hard_end
            ]
            end = candidates[-1] if candidates else hard_end
        chunks.append(Chunk(start, end))
        if end == num_samples:
            break
        start = end
    return chunks


def stitch_transcripts(pieces: Sequence[str]) -> str:
    """Join chunk text with a language-aware separator and no de-duplication."""
    stitched = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        stitched += _transcript_separator(stitched, piece) + piece
    return stitched


def _transcript_separator(previous: str, current: str) -> str:
    if not previous or not current or previous[-1].isspace() or current[0].isspace():
        return ""
    if _is_no_space_boundary(previous[-1]) or _is_no_space_boundary(current[0]):
        return ""
    if unicodedata.category(current[0]).startswith("P"):
        return ""
    return " "


def _is_no_space_boundary(character: str) -> bool:
    return _is_cjk_character(character) or (
        unicodedata.category(character).startswith("P")
        and unicodedata.east_asian_width(character) in {"F", "W"}
    )


def _is_cjk_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
        or 0x3040 <= codepoint <= 0x30FF
        or 0xAC00 <= codepoint <= 0xD7AF
    )


def prepare_audio(
    payload: StagePayload,
    *,
    source_name: str,
    target_sample_rate: int = DEFAULT_TARGET_SAMPLE_RATE,
    source_resolver: Callable[[StagePayload], Any] = resolve_audio_source,
    max_duration_s: float | None = None,
    max_duration_message: str | None = None,
) -> PreparedAudio:
    """Resolve, load, and fingerprint the payload's audio for one request."""

    source = source_resolver(payload)
    waveform = load_audio(
        source,
        source_name=source_name,
        target_sample_rate=target_sample_rate,
    )
    duration_s = float(len(waveform) / target_sample_rate)
    if max_duration_s is not None and duration_s > max_duration_s:
        raise ValueError(
            max_duration_message
            or (
                f"{source_name} accepts audio up to {max_duration_s} seconds, "
                f"got {duration_s:.3f} seconds"
            )
        )
    return PreparedAudio(
        waveform=waveform,
        sample_rate=target_sample_rate,
        duration_s=duration_s,
        fingerprint=audio_fingerprint(waveform),
    )


__all__ = [
    "Chunk",
    "DEFAULT_TARGET_SAMPLE_RATE",
    "PreparedAudio",
    "detect_silence_boundaries",
    "plan_chunks",
    "prepare_audio",
    "resolve_audio_source",
    "stitch_transcripts",
]
