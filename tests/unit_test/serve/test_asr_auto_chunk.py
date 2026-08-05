# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException

from sglang_omni.client.types import CompletionResult, GenerateRequest
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import _probe_audio_duration


class _TranscriptionClient:
    def __init__(self, texts: list[str]) -> None:
        self.requests: list[GenerateRequest] = []
        self._texts = iter(texts)

    def health(self) -> dict[str, bool]:
        return {"running": True}

    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> CompletionResult:
        del audio_format
        self.requests.append(request)
        return CompletionResult(request_id=request_id, text=next(self._texts))


class _FailingTranscriptionClient(_TranscriptionClient):
    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> CompletionResult:
        if len(self.requests) == 1:
            self.requests.append(request)
            raise RuntimeError("child completion failed")
        return await super().completion(
            request, request_id=request_id, audio_format=audio_format
        )


class _DirectUpload:
    filename = "sample.wav"
    content_type = "audio/wav"

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read(self) -> bytes:
        return self._data


def _wav_bytes(duration_s: float, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    samples = np.zeros(round(duration_s * sample_rate), dtype=np.float32)
    sf.write(buffer, samples, sample_rate, format="WAV")
    return buffer.getvalue()


def _endpoint(app):
    return next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/v1/audio/transcriptions"
    )


def _app(client, *, enabled: bool):
    return create_app(
        client,
        model_name="Qwen/Qwen3-ASR-1.7B",
        asr_auto_chunk=enabled,
        asr_chunk_max_seconds=4.0,
        asr_chunk_overlap_seconds=1.0,
    )


async def _call(app, audio_bytes: bytes, *, stream: bool = False):
    return await _endpoint(app)(
        request=object(),
        file=_DirectUpload(audio_bytes),
        model="Qwen/Qwen3-ASR-1.7B",
        language="en",
        prompt=None,
        response_format="json",
        temperature=None,
        max_new_tokens=None,
        stream=stream,
    )


@pytest.mark.asyncio
async def test_endpoint_auto_chunks_and_stitches_long_audio() -> None:
    client = _TranscriptionClient(["hello boundary", "boundary world", "world done"])

    response = await _call(_app(client, enabled=True), _wav_bytes(8.0))

    assert response.status_code == 200
    assert json.loads(response.body) == {
        "text": "hello boundary world done",
        "usage": {"type": "duration", "seconds": 8},
    }
    assert len(client.requests) == 3
    assert all(
        _probe_audio_duration(request.prompt["audio_bytes"]) <= 4.0
        for request in client.requests
    )


@pytest.mark.asyncio
async def test_endpoint_keeps_short_audio_on_original_single_path() -> None:
    audio_bytes = _wav_bytes(3.5)
    client = _TranscriptionClient(["hello world"])

    response = await _call(_app(client, enabled=True), audio_bytes)

    assert response.status_code == 200
    assert len(client.requests) == 1
    assert client.requests[0].prompt["audio_bytes"] == audio_bytes


@pytest.mark.asyncio
async def test_endpoint_rejects_long_audio_when_auto_chunk_disabled() -> None:
    client = _TranscriptionClient([])

    with pytest.raises(HTTPException) as exc_info:
        await _call(_app(client, enabled=False), _wav_bytes(8.0))

    assert exc_info.value.status_code == 400
    assert "8.000 seconds" in exc_info.value.detail
    assert "4.000 seconds" in exc_info.value.detail
    assert client.requests == []


@pytest.mark.asyncio
async def test_endpoint_streams_stitched_long_audio() -> None:
    client = _TranscriptionClient(["hello boundary", "boundary world", "world done"])

    response = await _call(_app(client, enabled=True), _wav_bytes(8.0), stream=True)
    body = "".join([line async for line in response.body_iterator])

    assert response.status_code == 200
    assert '"delta":"hello boundary world done"' in body
    assert '"text":"hello boundary world done"' in body
    assert body.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_endpoint_abandons_remaining_chunks_and_returns_500_on_child_failure() -> (
    None
):
    client = _FailingTranscriptionClient(["first chunk"])

    with pytest.raises(HTTPException) as exc_info:
        await _call(_app(client, enabled=True), _wav_bytes(8.0))

    assert exc_info.value.status_code == 500
    assert "child completion failed" in exc_info.value.detail
    assert len(client.requests) == 2
