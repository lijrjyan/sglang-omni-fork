# SPDX-License-Identifier: Apache-2.0
"""Hand the target's Whisper encoder output to a speculative draft that shares the encoder."""

from __future__ import annotations

from collections import OrderedDict

import torch


class EncoderStateShare:
    """Per-request encoder states published by the target model, consumed once by the draft.

    Keyed by ``req_pool_idx`` because target and draft workers share the request table.
    """

    def __init__(self, max_entries: int = 256) -> None:
        self._states: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._max_entries = max_entries
        self.published = 0
        self.consumed = 0
        self.misses = 0

    def publish(
        self, req_pool_indices: list[int], flat: torch.Tensor, encoder_lens: list[int]
    ) -> None:
        start = 0
        for idx, length in zip(req_pool_indices, encoder_lens):
            self._states[idx] = flat[start : start + length]
            start += length
            self.published += 1
        while len(self._states) > self._max_entries:
            self._states.popitem(last=False)

    def take(self, req_pool_indices: list[int]) -> torch.Tensor | None:
        """Return the concatenated states for the batch, or None if any request is missing."""
        if any(idx not in self._states for idx in req_pool_indices):
            self.misses += 1
            return None
        pieces = [self._states.pop(idx) for idx in req_pool_indices]
        self.consumed += len(pieces)
        return torch.cat(pieces, dim=0)


__all__ = ["EncoderStateShare"]
