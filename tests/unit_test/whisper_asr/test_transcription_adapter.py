# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Whisper segment-timestamp adapter."""

from __future__ import annotations

from sglang_omni.serve.transcription_adapters.whisper_asr import WhisperASRAdapter


def test_parse_whisper_timestamp_segments() -> None:
    adapter = WhisperASRAdapter()

    response = adapter.build_verbose_response(
        text="<|0.00|> Hello there.<|1.20|><|1.20|> Goodbye.<|2.40|>",
        language="en",
        audio_duration_s=2.4,
    )

    assert [
        (segment.id, segment.start, segment.end, segment.text)
        for segment in response.segments
    ] == [
        (0, 0.0, 1.2, "Hello there."),
        (1, 1.2, 2.4, "Goodbye."),
    ]
    assert response.text == "Hello there. Goodbye."


def test_postprocess_strips_whisper_timestamp_markers() -> None:
    adapter = WhisperASRAdapter()

    assert adapter.postprocess_text("<|0.00|> hello<|1.20|>") == "hello"


def test_markerless_text_falls_back_to_single_segment() -> None:
    adapter = WhisperASRAdapter()

    response = adapter.build_verbose_response(
        text="plain transcript", language="en", audio_duration_s=3.5
    )

    assert [
        (segment.id, segment.start, segment.end, segment.text)
        for segment in response.segments
    ] == [(0, 0.0, 3.5, "plain transcript")]
