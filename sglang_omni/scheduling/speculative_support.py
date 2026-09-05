# SPDX-License-Identifier: Apache-2.0
"""Which native SGLang speculative algorithms Omni's synchronous lane admits."""

from __future__ import annotations

from typing import Any

# Native SGLang V2 draft workers that share one scheduler-facing interface
# (``forward_batch_generation`` returning ``next_draft_input``/``new_seq_lens``).
OMNI_SPECULATIVE_ALGORITHMS: tuple[str, ...] = ("STANDALONE", "EAGLE3", "DFLASH")


def is_omni_supported_speculative_algorithm(algorithm: Any) -> bool:
    """Return whether a parsed ``SpeculativeAlgorithm`` runs in Omni's lane."""
    return bool(
        algorithm.is_standalone() or algorithm.is_eagle3() or algorithm.is_dflash()
    )


def unsupported_speculative_algorithm_message(algorithm: Any = None) -> str:
    supported = ", ".join(OMNI_SPECULATIVE_ALGORITHMS)
    message = (
        f"sglang-omni only supports {supported} speculative decoding in the "
        "synchronous lane"
    )
    if algorithm is not None:
        message += f"; got speculative_algorithm={algorithm!r}"
    return message


__all__ = [
    "OMNI_SPECULATIVE_ALGORITHMS",
    "is_omni_supported_speculative_algorithm",
    "unsupported_speculative_algorithm_message",
]
