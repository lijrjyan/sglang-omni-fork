# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import WhisperFeatureExtractor

import sglang_omni.preprocessing.transcription as transcription
from sglang_omni.models.qwen3_asr.audio_lengths import (
    qwen3_asr_audio_token_lengths,
    qwen3_asr_num_audio_tokens,
)
from sglang_omni.models.qwen3_asr.configuration_qwen3_asr import Qwen3ASRProcessor
from sglang_omni.models.qwen3_asr.request_builders import (
    QWEN3_ASR_MAX_AUDIO_SECONDS,
    Qwen3ASRRequestData,
    make_qwen3_asr_scheduler_adapters,
    qwen3_asr_auto_output_budget,
    qwen3_asr_request_output_budget,
)
from sglang_omni.proto import OmniRequest, StagePayload


class _FakeTokenizer:
    eos_token_id = 2
    vocab_size = 1000

    def __init__(self) -> None:
        self.encode_calls: list[str] = []
        self.decode_calls: list[dict] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|audio_pad|>"
        return 42

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        self.encode_calls.append(text)
        assert text == "<asr_text>"
        return [100, 101]

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        assert not add_special_tokens
        audio_pad_count = text.count("<|audio_pad|>")
        return SimpleNamespace(input_ids=[11] + [42] * audio_pad_count + [12, 13, 14])

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        self.decode_calls.append(
            {
                "token_ids": list(token_ids),
                "skip_special_tokens": skip_special_tokens,
                "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
            }
        )
        pieces = {
            10: "language English",
            100: "<asr_text>",
            101: "",
            20: " leading",
            21: "\u00a0middle",
            22: "  ",
            99: "<|endoftext|>",
        }
        text = "".join(pieces[token_id] for token_id in token_ids)
        if skip_special_tokens:
            text = text.replace("<|endoftext|>", "")
        return text


def test_qwen3_asr_audio_token_length_formula_is_shared() -> None:
    lengths = torch.tensor([0, 1, 99, 100, 101, 3000], dtype=torch.long)
    expected = torch.tensor([0, 1, 13, 13, 14, 390], dtype=torch.long)

    processor = object.__new__(Qwen3ASRProcessor)

    assert torch.equal(qwen3_asr_audio_token_lengths(lengths), expected)
    assert torch.equal(processor._get_feat_extract_output_lengths(lengths), expected)
    assert qwen3_asr_num_audio_tokens(3000) == 390


def test_qwen3_asr_auto_output_budget_matches_native_long_audio_envelope() -> None:
    assert QWEN3_ASR_MAX_AUDIO_SECONDS == 1200.0
    assert qwen3_asr_auto_output_budget(1.0) == 4096
    assert qwen3_asr_auto_output_budget(409.6) == 4096
    assert qwen3_asr_auto_output_budget(1200.0) == 12000


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "12", "not-an-int"])
def test_qwen3_asr_request_output_budget_rejects_invalid_explicit_values(
    invalid,
) -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        qwen3_asr_request_output_budget(
            {"max_new_tokens": invalid},
            audio_duration_s=1.0,
            default_max_new_tokens=None,
        )


def test_qwen3_asr_request_output_budget_precedence() -> None:
    assert (
        qwen3_asr_request_output_budget(
            {}, audio_duration_s=1200.0, default_max_new_tokens=None
        )
        == 12000
    )
    assert (
        qwen3_asr_request_output_budget(
            {}, audio_duration_s=1200.0, default_max_new_tokens=6000
        )
        == 6000
    )
    assert (
        qwen3_asr_request_output_budget(
            {"max_new_tokens": 7000},
            audio_duration_s=1200.0,
            default_max_new_tokens=6000,
        )
        == 7000
    )


def test_qwen3_asr_request_builder_records_inclusive_audio_offsets(monkeypatch) -> None:
    num_mel_frames = 101
    num_audio_tokens = qwen3_asr_num_audio_tokens(num_mel_frames)

    def feature_extractor(*args, **kwargs):
        return SimpleNamespace(
            input_features=torch.zeros((1, 128, 3000)),
            attention_mask=torch.ones((1, num_mel_frames), dtype=torch.long),
        )

    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600, dtype=np.float32),
    )
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=feature_extractor,
    )
    payload = StagePayload(
        request_id="req-asr",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )

    data = request_builder(payload)

    audio_item = data.req.multimodal_inputs.mm_items[0]
    start, end = audio_item.offsets[0]
    assert audio_item.feature_attention_mask.shape == (1, num_mel_frames)
    assert end - start + 1 == num_audio_tokens
    assert data.prompt_token_ids[start : end + 1] == (
        [audio_item.pad_value] * num_audio_tokens
    )


def test_qwen3_asr_request_builder_preserves_audio_beyond_30_seconds(
    monkeypatch,
) -> None:
    """The request builder must not apply Whisper's default 30-second truncation."""
    sample_rate = 16000
    audio_duration_s = 31
    feature_extractor = WhisperFeatureExtractor(
        feature_size=128,
        sampling_rate=sample_rate,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
    )
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(
            sample_rate * audio_duration_s, dtype=np.float32
        ),
    )
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=feature_extractor,
    )
    payload = StagePayload(
        request_id="req-long-asr",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )

    data = request_builder(payload)

    audio_item = data.req.multimodal_inputs.mm_items[0]
    assert audio_item.feature.shape == (1, 128, 3100)
    assert int(audio_item.feature_attention_mask.sum().item()) == 3100
    assert data.audio_duration_s == audio_duration_s


def test_qwen3_asr_request_builder_accepts_native_1200_second_envelope(
    monkeypatch,
) -> None:
    sample_rate = 16000
    audio_duration_s = 1200
    num_mel_frames = audio_duration_s * 100
    extractor_calls: list[dict] = []

    def _feature_extractor(*args, **kwargs):
        extractor_calls.append(kwargs)
        return SimpleNamespace(
            input_features=torch.empty(
                (1, 128, num_mel_frames), device="meta", dtype=torch.float32
            ),
            attention_mask=torch.ones((1, num_mel_frames), dtype=torch.long),
        )

    # note (Junnan Li): Avoid materializing the full waveform so this boundary
    # test remains memory-light.
    audio = np.broadcast_to(
        np.zeros(1, dtype=np.float32), (sample_rate * audio_duration_s,)
    )
    monkeypatch.setattr(transcription, "load_audio", lambda source, **kwargs: audio)
    monkeypatch.setattr(transcription, "audio_fingerprint", lambda audio: "0f")
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=None,
        feature_extractor=_feature_extractor,
    )
    payload = StagePayload(
        request_id="req-native-max-asr",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )

    data = request_builder(payload)

    assert extractor_calls == [
        {
            "sampling_rate": sample_rate,
            "return_tensors": "pt",
            "return_attention_mask": True,
            "padding": "longest",
            "truncation": False,
        }
    ]
    assert data.audio_duration_s == audio_duration_s
    assert data.max_new_tokens == 12000
    assert data.req.sampling_params.max_new_tokens == 12000
    assert data.enforce_request_limits is True
    assert len(data.prompt_token_ids) == qwen3_asr_num_audio_tokens(num_mel_frames) + 4


def test_qwen3_asr_result_adapter_decodes_without_text_round_trip() -> None:
    tokenizer = _FakeTokenizer()
    _, result_adapter = make_qwen3_asr_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=32,
        feature_extractor=object(),
    )
    payload = StagePayload(
        request_id="req-asr",
        request=OmniRequest(inputs={}),
        data={},
    )
    data = Qwen3ASRRequestData(
        output_ids=[10, 100, 101, 20, 21, 22, 99],
        stage_payload=payload,
        language="en",
        audio_duration_s=1.25,
    )

    result = result_adapter(data)

    assert result.data["text"] == " leading\u00a0middle  "
    assert tokenizer.encode_calls == ["<asr_text>"]
    assert tokenizer.decode_calls[-1] == {
        "token_ids": [20, 21, 22, 99],
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }
