# SPDX-License-Identifier: Apache-2.0
"""Python GC control helpers shared by the launcher, stage runtime and admin path.

``gc.freeze()`` moves every object currently tracked by the cyclic collector
into a permanent generation that later collections skip.  After model load and
CUDA graph capture a serving process holds millions of long-lived objects
(weights, graph runners, tokenizer tables); freezing them keeps gen2 collections
from re-scanning that static set on every request, which shows up as tail
latency on the scheduler thread.  Freezing is idempotent and never affects
correctness: objects created afterwards stay in the regular generations.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

FREEZE_GC_AFTER_STARTUP_ENV = "SGLANG_OMNI_FREEZE_GC_AFTER_STARTUP"


def freeze_gc_after_startup_enabled() -> bool:
    """Return whether the launcher should freeze GC once the pipeline is ready.

    Enabled by default; set ``SGLANG_OMNI_FREEZE_GC_AFTER_STARTUP=0`` to keep the
    pre-existing behaviour (no freeze).  ``POST /freeze_gc`` stays available
    either way.
    """
    raw = os.environ.get(FREEZE_GC_AFTER_STARTUP_ENV, "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def gc_object_counts() -> tuple[int, int, int]:
    return tuple(len(gc.get_objects(generation=i)) for i in range(3))  # type: ignore[return-value]


def freeze_gc(context: str) -> dict[str, Any]:
    """Freeze the cyclic GC in this process and return before/after generation sizes."""
    before = gc_object_counts()
    gc.freeze()
    after = gc_object_counts()
    frozen = gc.get_freeze_count()
    logger.info(
        "Freezing GC in %s process (pid=%d): gen0 %d->%d, gen1 %d->%d, gen2 %d->%d, frozen=%d",
        context,
        os.getpid(),
        before[0],
        after[0],
        before[1],
        after[1],
        before[2],
        after[2],
        frozen,
    )
    return {
        "context": context,
        "pid": os.getpid(),
        "before": {"gen0": before[0], "gen1": before[1], "gen2": before[2]},
        "after": {"gen0": after[0], "gen1": after[1], "gen2": after[2]},
        "frozen": frozen,
    }
