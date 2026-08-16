# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from sglang_omni.models.whisper_asr import quantization_scope
from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder


def _builder(**kwargs) -> WhisperASREngineBuilder:
    return WhisperASREngineBuilder(
        max_running_requests=4, max_new_tokens=32, mem_fraction_static=0.2, **kwargs
    )


def test_encoder_scope_rides_on_json_model_override_args() -> None:
    overrides = {"json_model_override_args": json.dumps({"keep": 1})}

    _builder(quantization_scope="encoder").adjust_overrides(overrides)

    assert json.loads(overrides["json_model_override_args"]) == {
        "keep": 1,
        quantization_scope.QUANTIZATION_SCOPE_ATTR: "encoder",
    }


def test_default_scope_leaves_model_overrides_alone() -> None:
    overrides: dict = {}

    _builder().adjust_overrides(overrides)

    assert "json_model_override_args" not in overrides


def test_unknown_scope_is_rejected_at_the_builder() -> None:
    with pytest.raises(ValueError, match="quantization_scope"):
        _builder(quantization_scope="decoder")


def test_decoder_quant_config_follows_scope() -> None:
    quant = object()
    plain = SimpleNamespace()
    encoder_only = SimpleNamespace(
        **{quantization_scope.QUANTIZATION_SCOPE_ATTR: "encoder"}
    )

    assert quantization_scope.decoder_quant_config(plain, quant) is quant
    assert quantization_scope.decoder_quant_config(encoder_only, quant) is None
