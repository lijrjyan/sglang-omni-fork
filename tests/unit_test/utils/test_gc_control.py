# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gc

from sglang_omni.utils.gc_control import (
    FREEZE_GC_AFTER_STARTUP_ENV,
    freeze_gc,
    freeze_gc_after_startup_enabled,
)


def test_freeze_gc_reports_counts_and_is_idempotent() -> None:
    junk = [[i] for i in range(1000)]  # tracked containers in gen0
    first = freeze_gc("test")
    # The interpreter may track/untrack a handful of its own objects between
    # the freeze and the readback; only the order of magnitude is meaningful.
    assert first["frozen"] > 0
    assert abs(first["frozen"] - gc.get_freeze_count()) < 100
    assert first["after"]["gen0"] <= first["before"]["gen0"]
    second = freeze_gc("test")
    assert second["frozen"] >= first["frozen"]
    del junk
    gc.unfreeze()


def test_freeze_gc_after_startup_env_parsing(monkeypatch) -> None:
    monkeypatch.delenv(FREEZE_GC_AFTER_STARTUP_ENV, raising=False)
    assert freeze_gc_after_startup_enabled() is True
    for raw in ("0", "false", "NO", " off "):
        monkeypatch.setenv(FREEZE_GC_AFTER_STARTUP_ENV, raw)
        assert freeze_gc_after_startup_enabled() is False
    monkeypatch.setenv(FREEZE_GC_AFTER_STARTUP_ENV, "1")
    assert freeze_gc_after_startup_enabled() is True
