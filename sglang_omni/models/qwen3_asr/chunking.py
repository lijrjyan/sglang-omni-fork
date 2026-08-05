# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR-owned long-audio chunking policy."""

from __future__ import annotations

import math

QWEN3_ASR_AUTO_CHUNK_DEFAULT = True

# note (Junnan Li): Keep the chunk ceiling aligned with the deployed 30-second
# serving context; raise it when that context limit changes.
QWEN3_ASR_CHUNK_WINDOW_SECONDS = 30.0


def validate_qwen3_asr_chunking(max_seconds: float) -> None:
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("asr_chunk_max_seconds must be a finite positive number")


__all__ = [
    "QWEN3_ASR_AUTO_CHUNK_DEFAULT",
    "QWEN3_ASR_CHUNK_WINDOW_SECONDS",
    "validate_qwen3_asr_chunking",
]
