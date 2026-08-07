# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from sglang_omni.serve.speech_to_text import (
    assemble_speech_to_text_response,
    build_transcription_generate_request,
    validate_speech_to_text_response_format,
)


def _build_request(*, task: str = "transcribe"):
    return build_transcription_generate_request(
        audio_bytes=b"RIFF",
        filename="sample.wav",
        content_type="audio/wav",
        model="openai/whisper-large-v3",
        language="en",
        prompt=None,
        temperature=None,
        task=task,
    )


def test_build_request_defaults_to_transcribe_task() -> None:
    assert _build_request().extra_params["task"] == "transcribe"


def test_build_request_accepts_sibling_endpoint_task() -> None:
    assert _build_request(task="translate").extra_params["task"] == "translate"


def test_response_format_validation_preserves_endpoint_error_contract() -> None:
    with pytest.raises(HTTPException) as exc_info:
        validate_speech_to_text_response_format(
            " SRT ",
            stream=False,
            endpoint_path="/v1/audio/transcriptions",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == (
        "Unsupported response_format for /v1/audio/transcriptions: ' SRT '"
    )


def test_verbose_response_uses_requested_task() -> None:
    response = assemble_speech_to_text_response(
        text="hello world",
        response_format="verbose_json",
        endpoint_path="/v1/audio/transcriptions",
        task="translate",
        language="en",
        audio_bytes=b"not-a-real-audio-file",
        architectures=None,
    )

    assert json.loads(response.body)["task"] == "translate"
