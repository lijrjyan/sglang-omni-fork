# Whisper ASR

Whisper ASR checkpoints can be started through the OpenAI-compatible `/v1/audio/transcriptions` endpoint, but this path is experimental in the current SGLang-Omni tree. Prefer [Qwen3-ASR](qwen3_asr.md) for validated ASR serving.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download a Whisper checkpoint:

```bash
hf download openai/whisper-large-v3
```

## Server Configuration

Whisper ASR runs a single ASR stage on one GPU.

```bash
sgl-omni serve \
  --model-path openai/whisper-large-v3 \
  --port 8000
```

## Transcribe Audio

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=openai/whisper-large-v3 \
  -F file=@tests/data/query_to_cars.wav \
  -F response_format=json
```

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "openai/whisper-large-v3",
            "response_format": "json",
        },
        files={"file": ("query_to_cars.wav", f, "audio/wav")},
        timeout=300,
    )

resp.raise_for_status()
print(resp.json()["text"])
```

## Translate Audio

Whisper multilingual checkpoints can translate source speech to English via
`/v1/audio/translations`. Use a multilingual, non-turbo checkpoint: `*.en`
checkpoints have no translate task, and `whisper-large-v3-turbo` was distilled
without it.

```bash
curl -X POST http://localhost:8000/v1/audio/translations \
  -F model=openai/whisper-large-v3 \
  -F file=@tests/data/query_to_cars.wav \
  -F language=fr \
  -F response_format=json
```

For this endpoint, `language` is an optional source-language hint and a
**SGLang-Omni extension**. OpenAI's official audio translations request schema
does not include `language`; the translation target is English in both APIs.
See the [audio translation support matrix](../basic_usage/audio_translations.md)
for response formats and other ASR models.

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file uploaded as multipart form data |
| `model` | string | server default | Model identifier |
| `language` | string | unset | Optional source-language hint; on translations this is a SGLang-Omni extension |
| `response_format` | string | `json` | `json`, `verbose_json`, or raw `text`; translation `srt`/`vtt` require segment timestamps and return HTTP 400 |
| `temperature` | float | `0.0` | Sampling temperature; defaults to greedy decoding |

The serving route selects the internal `task` from the endpoint (`transcribe`
or `translate`); it is not a public form field. The route uses the ASR stage
default unless the pipeline is configured another way. For smoke tests, keep
the request minimal and use `response_format=json`.

## Known Limitations

- This path is experimental and not yet correctness-validated. Prefer Qwen3-ASR
  for validated ASR serving.
- Keep Whisper ASR at encoder batch size 1.
- `verbose_json` contains one duration-based placeholder segment; `srt` and
  `vtt` are rejected until real segment timestamps exist.
- First startup can take several minutes.
- The endpoint accepts one uploaded file per request.
- Audio is resampled to 16 kHz before transcription.
- `prompt` is accepted by the HTTP endpoint for OpenAI compatibility, but
  Whisper ASR currently does not pass it into decoding.
