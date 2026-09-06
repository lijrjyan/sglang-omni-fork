# SPDX-License-Identifier: Apache-2.0
"""Admission and per-algorithm rules for Omni's synchronous speculative lane (L1).

Admission uses SGLang's own ``SpeculativeAlgorithm`` predicates; nothing here
re-describes what the native workers already declare. The rules that Omni has
to know about an algorithm are kept together in one place so the engine
factory and bootstrap never branch on algorithm names themselves:

* the window rule (how ``speculative_num_steps`` / ``speculative_num_draft_tokens``
  relate: chain drafts verify ``steps + 1`` tokens, block drafts verify one
  block and take no step count);
* whether an eager prefill leaves recurrent draft hidden state (STANDALONE: no);
* which draft graph the algorithm needs from a stock ``init_cuda_graphs``;
* the graph policy applied to the stage's ``cuda_graph_config``.

Adding an algorithm means one predicate in ``is_omni_supported_speculative_algorithm``
plus one row in each rule table below. ``DSPARK`` is the documented next slot:
it is a block draft (``steps == 1``, ``N == block_size + 1`` resolvable from
the draft checkpoint) whose worker has no ``speculative_algorithm``
attribute of its own; admit it with ``algorithm.is_dspark()`` and give it the
``BLOCK`` window rule and the ``VERIFY`` draft graph kind.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

# Native SGLang V2 draft workers that share one scheduler-facing interface
# (``forward_batch_generation`` returning ``next_draft_input``/``new_seq_lens``).
OMNI_SPECULATIVE_ALGORITHMS: tuple[str, ...] = ("STANDALONE", "EAGLE3", "DFLASH")


def is_omni_supported_speculative_algorithm(algorithm: Any) -> bool:
    """Return whether a parsed ``SpeculativeAlgorithm`` runs in Omni's lane."""
    # DSPARK slot: ``or algorithm.is_dspark()`` once the CosyVoice3 contract lands.
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


def normalize_speculative_algorithm(algorithm: Any) -> str:
    """Upper-case an algorithm name, refusing anything outside the lane."""
    normalized = str(algorithm).upper()
    if normalized not in OMNI_SPECULATIVE_ALGORITHMS:
        raise ValueError(unsupported_speculative_algorithm_message(algorithm))
    return normalized


class WindowRule(Enum):
    """How the draft window is sized from the factory knobs."""

    # ``speculative_num_steps`` chain steps, ``N == steps + 1``, topk fixed to 1.
    CHAIN = "chain"
    # ``speculative_num_draft_tokens`` is the block size; no step count.
    BLOCK = "block"


_WINDOW_RULES: dict[str, WindowRule] = {
    "STANDALONE": WindowRule.CHAIN,
    "EAGLE3": WindowRule.CHAIN,
    "DFLASH": WindowRule.BLOCK,
    # "DSPARK": WindowRule.BLOCK  (steps forced to 1, N resolvable from checkpoint)
}


def window_rule(algorithm: str) -> WindowRule:
    return _WINDOW_RULES[normalize_speculative_algorithm(algorithm)]


def takes_step_count(algorithm: str) -> bool:
    """Whether ``speculative_num_steps`` is a meaningful knob for ``algorithm``."""
    return window_rule(algorithm) is WindowRule.CHAIN


def apply_window_rule(configured: dict[str, Any], algorithm: str) -> None:
    """Validate and normalise the window knobs in ``configured`` in place."""
    if window_rule(algorithm) is WindowRule.BLOCK:
        block_size = int(configured.setdefault("speculative_num_draft_tokens", 8))
        if block_size < 1:
            raise ValueError(
                f"{algorithm} speculative_num_draft_tokens is the draft block size "
                f"and must be >= 1; got {block_size}"
            )
        configured["speculative_num_draft_tokens"] = block_size
        configured.pop("speculative_num_steps", None)
        return
    num_steps = int(configured.setdefault("speculative_num_steps", 3))
    num_draft_tokens = int(configured.setdefault("speculative_num_draft_tokens", 4))
    if num_steps < 0:
        raise ValueError(f"speculative_num_steps must be >= 0, got {num_steps}")
    if num_draft_tokens != num_steps + 1:
        raise ValueError(
            "topk=1 requires speculative_num_draft_tokens == "
            f"speculative_num_steps + 1; got {num_draft_tokens} and {num_steps}"
        )
    configured["speculative_num_steps"] = num_steps
    configured["speculative_num_draft_tokens"] = num_draft_tokens


def eager_prefill_has_hidden(algorithm: Any) -> bool:
    """Whether an eager (non-graph) prefill leaves recurrent draft hidden state.

    EAGLE-family drafts (EAGLE3, DFLASH) carry the target's captured hidden
    state into the next step; a STANDALONE draft is a full model and has none,
    so the scheduler bridge clears the field before merging batches.
    """
    return not algorithm.is_standalone()


class DraftGraphKind(Enum):
    """What a stock ``init_cuda_graphs()`` captures for the draft."""

    # Chain drafts capture a decode graph (only when steps > 1) plus the
    # draft-extend initialisation.
    DECODE_AND_EXTEND = "decode+extend"
    # Block drafts capture one verify-shaped graph on the inner worker.
    VERIFY = "verify"


_DRAFT_GRAPH_KINDS: dict[str, DraftGraphKind] = {
    "STANDALONE": DraftGraphKind.DECODE_AND_EXTEND,
    "EAGLE3": DraftGraphKind.DECODE_AND_EXTEND,
    "DFLASH": DraftGraphKind.VERIFY,
    # "DSPARK": DraftGraphKind.VERIFY
}


def draft_graph_kind(algorithm: str) -> DraftGraphKind:
    return _DRAFT_GRAPH_KINDS[normalize_speculative_algorithm(algorithm)]


def apply_graph_policy(
    configured: dict[str, Any],
    *,
    speculative_cuda_graph: bool,
    speculative_prefill_cuda_graph: bool,
) -> None:
    """Pin the stage's graph settings to the speculative lane's policy in place.

    Target verify (decode-phase) graphs follow ``speculative_cuda_graph``; the
    prefill graph stays disabled unless a decoder-only builder opts in with
    ``speculative_prefill_cuda_graph`` (default off, D13).
    """
    # note (Junnan Li): target and draft graph capture follow multi-token parity.
    configured["disable_overlap_schedule"] = True
    configured["disable_cuda_graph"] = not speculative_cuda_graph
    # SGLang 0.5.18 gives nested graph settings precedence over flat flags.
    # The speculative path uses target verify graphs; the stage's prefill graph
    # policy is kept only when a decoder-only builder opts in explicitly.
    if "cuda_graph_config" in configured:
        from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig

        graph_config = configured["cuda_graph_config"]
        graph_config = (
            graph_config.to_dict()
            if isinstance(graph_config, CudaGraphConfig)
            else dict(graph_config)
        )
        if not speculative_prefill_cuda_graph:
            graph_config["prefill"] = {"backend": "disabled"}
        graph_config["decode"] = {
            **graph_config.get("decode", {}),
            "backend": "full" if speculative_cuda_graph else "disabled",
        }
        configured["cuda_graph_config"] = graph_config
    if "cuda_graph_backend_decode" in configured:
        configured["cuda_graph_backend_decode"] = (
            "full" if speculative_cuda_graph else "disabled"
        )
    if (
        "cuda_graph_backend_prefill" in configured
        and not speculative_prefill_cuda_graph
    ):
        configured["cuda_graph_backend_prefill"] = "disabled"
        configured.pop("cuda_graph_bs_prefill", None)
        configured.pop("cuda_graph_max_bs_prefill", None)


__all__ = [
    "DraftGraphKind",
    "OMNI_SPECULATIVE_ALGORITHMS",
    "WindowRule",
    "apply_graph_policy",
    "apply_window_rule",
    "draft_graph_kind",
    "eager_prefill_has_hidden",
    "is_omni_supported_speculative_algorithm",
    "normalize_speculative_algorithm",
    "takes_step_count",
    "unsupported_speculative_algorithm_message",
    "window_rule",
]
