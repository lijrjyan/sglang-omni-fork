# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only (env-gated): record the non-speculative sampling post-processing
chain (raw logits -> Omni repetition-penalty step -> SGLang ``apply_logits_bias``)
for the decode row at one position, plus the sampled token.

Env: SGLANG_OMNI_SAMPLEPROBE_DIR (output dir; unset = disabled),
     SGLANG_OMNI_SAMPLEPROBE_POS (decode position, default 131; -1 = every decode row),
     SGLANG_OMNI_SAMPLEPROBE_ARM (label),
     SGLANG_OMNI_SAMPLEPROBE_DOUBLE=1 forces Omni's host-side penalty even when
     SGLang's penalizer is prepared (reproduces the pre-fix double penalty).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import torch

_DIR = os.environ.get("SGLANG_OMNI_SAMPLEPROBE_DIR")
_POS = int(os.environ.get("SGLANG_OMNI_SAMPLEPROBE_POS", "131"))
_ARM = os.environ.get("SGLANG_OMNI_SAMPLEPROBE_ARM", "arm")
ENABLED = bool(_DIR)
FORCE_DOUBLE = os.environ.get("SGLANG_OMNI_SAMPLEPROBE_DOUBLE", "0") == "1"


def _top(logits: torch.Tensor, row: int, k: int = 5) -> list[list[float]]:
    v, i = torch.topk(logits[row].float(), k)
    return [[int(a), round(float(b), 4)] for a, b in zip(i.tolist(), v.tolist())]


def match(forward_batch: Any) -> int | None:
    """Row index of the decode row at the probe position, or None."""
    if not ENABLED:
        return None
    positions = getattr(forward_batch, "positions", None)
    if positions is None:
        return None
    pos = positions.detach().cpu().tolist()
    if len(pos) != getattr(forward_batch, "batch_size", len(pos)):
        return None
    for r, p in enumerate(pos):
        if _POS < 0 or int(p) == _POS:
            return r
    return None


class Recorder:
    def __init__(self, forward_batch: Any, row: int, sglang_owns: bool) -> None:
        self.fb = forward_batch
        self.row = row
        self.rec: dict[str, Any] = {
            "arm": _ARM,
            "pos": int(forward_batch.positions[row]),
            "row": row,
            "input_id": int(forward_batch.input_ids[row]),
            "t": time.time(),
            "sglang_owns_penalty": sglang_owns,
            "force_double": FORCE_DOUBLE,
        }
        self._orig = None

    def stage(self, name: str, logits_output: Any) -> None:
        self.rec[name] = _top(logits_output.next_token_logits, self.row)

    def wrap_bias(self) -> None:
        si = getattr(self.fb, "sampling_info", None)
        if si is None:
            return
        orig = si.apply_logits_bias
        self._orig = orig
        row = self.row
        rec = self.rec

        def wrapped(logits: torch.Tensor):
            out = orig(logits)
            rec["post_sglang_bias"] = _top(logits, row)
            si.apply_logits_bias = orig
            return out

        si.apply_logits_bias = wrapped

    def finish(self, next_token_ids: Any) -> None:
        si = getattr(self.fb, "sampling_info", None)
        if si is not None and self._orig is not None:
            si.apply_logits_bias = self._orig
        try:
            self.rec["sampled"] = int(next_token_ids[self.row])
        except Exception:  # pragma: no cover - diagnostic
            self.rec["sampled"] = None
        os.makedirs(_DIR, exist_ok=True)
        with open(os.path.join(_DIR, f"sampleprobe-{_ARM}.jsonl"), "a") as f:
            f.write(json.dumps(self.rec) + "\n")
        print(f"[sampleprobe] {json.dumps(self.rec)}", flush=True)
