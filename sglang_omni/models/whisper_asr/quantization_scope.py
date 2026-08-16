# SPDX-License-Identifier: Apache-2.0
"""Restrict the server-wide quantization to the Whisper encoder."""

from __future__ import annotations

from typing import Any

QUANTIZATION_SCOPE_ATTR = "sglang_omni_quantization_scope"
QUANTIZATION_SCOPES = ("all", "encoder")


def validate_quantization_scope(scope: str) -> str:
    if scope not in QUANTIZATION_SCOPES:
        raise ValueError(
            f"quantization_scope must be one of {QUANTIZATION_SCOPES}, got {scope!r}"
        )
    return scope


def decoder_quant_config(config: Any, quant_config: Any) -> Any:
    """Return the decoder's quantization config under the configured scope."""
    scope = validate_quantization_scope(getattr(config, QUANTIZATION_SCOPE_ATTR, "all"))
    return None if scope == "encoder" else quant_config


__all__ = [
    "QUANTIZATION_SCOPE_ATTR",
    "QUANTIZATION_SCOPES",
    "decoder_quant_config",
    "validate_quantization_scope",
]
