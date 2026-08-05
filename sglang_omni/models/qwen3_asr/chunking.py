# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR-owned long-audio chunking policy."""

from __future__ import annotations

QWEN3_ASR_AUTO_CHUNK_DEFAULT = True

# The serving context currently admits 30-second requests, which is smaller
# than Qwen3-ASR's native audio envelope.
QWEN3_ASR_CHUNK_WINDOW_SECONDS = 30.0
QWEN3_ASR_CHUNK_OVERLAP_SECONDS = 2.0


def validate_qwen3_asr_chunking(max_seconds: float, overlap_seconds: float) -> None:
    if max_seconds <= 0:
        raise ValueError("asr_chunk_max_seconds must be positive")
    if overlap_seconds < 0:
        raise ValueError("asr_chunk_overlap_seconds must be non-negative")
    if overlap_seconds >= max_seconds:
        raise ValueError(
            "asr_chunk_overlap_seconds must be smaller than asr_chunk_max_seconds"
        )


__all__ = [
    "QWEN3_ASR_AUTO_CHUNK_DEFAULT",
    "QWEN3_ASR_CHUNK_OVERLAP_SECONDS",
    "QWEN3_ASR_CHUNK_WINDOW_SECONDS",
    "validate_qwen3_asr_chunking",
]
