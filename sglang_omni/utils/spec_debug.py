# SPDX-License-Identifier: Apache-2.0
"""File-backed trace for native speculative decoding bring-up.

Stage worker processes reconfigure logging after startup, so a plain logger
is not a reliable channel for per-step diagnostics. This helper appends to
``SGLANG_OMNI_SPEC_TRACE`` when ``SGLANG_OMNI_SPEC_DEBUG=1`` and is a no-op
otherwise.
"""

from __future__ import annotations

import os
import time

SPEC_DEBUG = os.environ.get("SGLANG_OMNI_SPEC_DEBUG", "0") == "1"


def spec_trace(message: str) -> None:
    if not SPEC_DEBUG:
        return
    path = os.environ.get("SGLANG_OMNI_SPEC_TRACE", "/tmp/sglang_omni_spec_trace.log")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{time.time():.3f} pid={os.getpid()} {message}\n")
