# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Qwen3-Omni text-only request construction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")
pytest.importorskip("xxhash")

from sglang_omni.models.qwen3_omni.merge import merge_for_thinker
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.request_builders import (
    _compute_mrope_positions,
    build_sglang_thinker_request,
    build_thinker_request,
)
from tests.unit_test.fixtures.qwen_fakes import (
    FakeQwenTokenizer,
    make_qwen_payload,
    make_qwen_state,
)


def _patch_sampling_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.normalize",
        lambda self, tokenizer: None,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )


def test_empty_model_inputs_are_not_replaced_by_the_flat_state(monkeypatch):
    _patch_sampling_validation(monkeypatch)
    merged = merge_for_thinker({"preprocessing": make_qwen_payload(make_qwen_state())})
    state = Qwen3OmniPipelineState.from_dict(merged.data)
    assert state.thinker_inputs == {"model_inputs": {}}

    generic = build_thinker_request(state, params={"max_new_tokens": 3})
    assert generic.model_inputs == {}

    sglang_request = build_sglang_thinker_request(
        state,
        params={"max_new_tokens": 3},
        tokenizer=FakeQwenTokenizer(),
        vocab_size=256,
        request_id="text-only",
        thinker_config=SimpleNamespace(
            image_token_id=55,
            video_token_id=66,
            audio_token_id=77,
        ),
    )

    assert sglang_request.model_inputs == {}
    assert sglang_request.req.omni_model_inputs is None
    assert getattr(sglang_request.req, "multimodal_inputs", None) is None
    assert getattr(sglang_request.req, "_omni_mm_positions", None) is None


@pytest.mark.parametrize("entrypoint", ["generic", "sglang"])
def test_nested_non_dict_model_inputs_fail_loudly(monkeypatch, entrypoint):
    _patch_sampling_validation(monkeypatch)
    state = make_qwen_state(thinker_inputs={"model_inputs": ["malformed"]})

    with pytest.raises(
        TypeError,
        match="Qwen3-Omni thinker model_inputs must be a dict when provided",
    ):
        if entrypoint == "generic":
            build_thinker_request(state, params={"max_new_tokens": 3})
        else:
            build_sglang_thinker_request(
                state,
                params={"max_new_tokens": 3},
                tokenizer=FakeQwenTokenizer(),
                vocab_size=256,
                request_id="malformed-model-inputs",
                thinker_config=SimpleNamespace(
                    image_token_id=55,
                    video_token_id=66,
                    audio_token_id=77,
                ),
            )


def test_legacy_flat_payloads_still_reach_the_model_input_field():
    audio_embeds = torch.ones((1, 4))
    state = make_qwen_state(
        thinker_inputs={
            "audio_embeds": audio_embeds,
            "media_cache_keys": {"audio": "audio:cache"},
        }
    )

    request = build_thinker_request(state, params={"max_new_tokens": 3})

    assert request.model_inputs == {"audio_embeds": audio_embeds}


def test_pure_text_qwen_mrope_is_ordinary_sequential_positions():
    """Text-only M-RoPE metadata is redundant with ordinary positions."""
    sequence_length = 5
    config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2, tokens_per_second=25),
        image_token_id=55,
        video_token_id=66,
        vision_start_token_id=44,
        audio_token_id=77,
        audio_start_token_id=88,
        position_id_per_seconds=25,
    )

    positions, _delta = _compute_mrope_positions(
        torch.arange(sequence_length, dtype=torch.long),
        {},
        config,
    )

    torch.testing.assert_close(
        positions,
        torch.arange(sequence_length, dtype=torch.long).repeat(3, 1),
    )


@pytest.mark.parametrize(
    "visual_key", ["image_embeds", "video_embeds", "image_grid_thw", "video_grid_thw"]
)
def test_speculative_admission_rejects_visual_before_request_construction(
    monkeypatch, visual_key
):
    from sglang_omni.models.qwen3_omni import request_builders as builders

    called = []

    def build(state, **kwargs):
        called.append(state)
        return SimpleNamespace()

    monkeypatch.setattr(builders, "build_sglang_thinker_request", build)
    request_builder, _ = builders.make_thinker_scheduler_adapters(
        tokenizer=FakeQwenTokenizer(), vocab_size=256, enable_speculative=True
    )
    visual = make_qwen_payload(
        make_qwen_state(thinker_inputs={"model_inputs": {visual_key: torch.ones(1)}})
    )
    with pytest.raises(ValueError, match="audio and text input"):
        request_builder(visual)
    assert not called
    request_builder(make_qwen_payload(make_qwen_state()))
    assert len(called) == 1


@pytest.mark.parametrize(
    "params",
    [
        {"temperature": 0.7},
        {"repetition_penalty": 1.3},
        {"frequency_penalty": 0.5},
        {"presence_penalty": 0.5},
        {"min_new_tokens": 2},
    ],
)
def test_speculative_sampling_contract_rejects_before_construction(monkeypatch, params):
    from sglang_omni.models.qwen3_omni import request_builders as builders

    monkeypatch.setattr(
        builders,
        "build_sglang_thinker_request",
        lambda *args, **kwargs: pytest.fail(
            "unsupported sampling reached construction"
        ),
    )
    request_builder, _ = builders.make_thinker_scheduler_adapters(
        tokenizer=FakeQwenTokenizer(), vocab_size=256, enable_speculative=True
    )
    payload = make_qwen_payload(make_qwen_state())
    payload.request.params.update(params)
    with pytest.raises(ValueError, match="speculative decoding requires"):
        request_builder(payload)


def test_non_speculative_sampling_keeps_request_parameters(monkeypatch):
    from sglang_omni.models.qwen3_omni import request_builders as builders

    seen = []

    def build(state, **kwargs):
        seen.append(kwargs["params"])
        return SimpleNamespace()

    monkeypatch.setattr(builders, "build_sglang_thinker_request", build)
    request_builder, _ = builders.make_thinker_scheduler_adapters(
        tokenizer=FakeQwenTokenizer(), vocab_size=256
    )
    payload = make_qwen_payload(make_qwen_state())
    payload.request.params.update(temperature=0.7, repetition_penalty=1.3)
    request_builder(payload)
    assert seen == [payload.request.params]
