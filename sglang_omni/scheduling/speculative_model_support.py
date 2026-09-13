# SPDX-License-Identifier: Apache-2.0
"""The narrow per-family runtime interface for Omni's speculative lane (L2).

SGLang already owns the model-side duck-typed hooks (``get_embed_and_head``,
``set_eagle3_layers_to_capture`` / ``set_dflash_layers_to_capture``); draft
descriptions (aux layer ids, block size, vocab maps) live in the draft
checkpoint. What remains family-specific, and has no home elsewhere, is
collected here with defaults so a text-only decoder family needs nothing:

* config-time validation of the draft against the target family
  (``validate_speculative_draft``, one signature for every family);
* request admission (``reject_speculative_request``: return the refusal
  message, ``None`` to admit — the caller raises so the error mapping is
  unchanged);
* the draft's view of a multimodal request when the draft carries its own
  encoder and must re-encode (``draft_view_of_request``; identity by default,
  Qwen3-ASR keeps the mel and salts the mm hash). Target and draft share one
  ``ForwardBatch``, so the view is applied by the draft model's forward, not
  by the scheduler bridge;
* the encoder-decoder surface: ``position_reserve`` (Whisper reserves two
  draft windows of decoder positions) and encoder-output sharing, wired by
  ``link_models`` together with the ``spec_role`` marker ("target"/"draft").
"""

from __future__ import annotations

from typing import Any


class SpeculativeModelSupport:
    """Defaults: text-only decoder target, stock draft, nothing to reject."""

    def normalize_draft_checkpoint(self, path: str, algorithm: str) -> str:
        return path

    def validate_speculative_draft(
        self,
        target_cfg: Any,
        draft_cfg: Any,
        algorithm: str,
        num_draft_tokens: int,
    ) -> None:
        del target_cfg, draft_cfg, algorithm, num_draft_tokens

    def reject_speculative_request(self, request: Any) -> str | None:
        del request
        return None

    def draft_view_of_request(self, mm_inputs: Any) -> Any:
        return mm_inputs

    def position_reserve(self, num_draft_tokens: int) -> int:
        del num_draft_tokens
        return 0

    def link_models(self, target_model: Any, draft_model: Any) -> None:
        """Mark the roles once both models are loaded; families extend this."""
        assign_spec_role(target_model, "target")
        assign_spec_role(draft_model, "draft")


def assign_spec_role(model: Any, role: str) -> None:
    """Set the unified ``spec_role`` marker (``encoder_share_role`` is its alias)."""
    model.spec_role = role
    # Back-compat alias for readers of the pre-unification Whisper attribute.
    model.encoder_share_role = role


__all__ = ["SpeculativeModelSupport", "assign_spec_role"]
