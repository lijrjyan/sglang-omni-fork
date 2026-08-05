# Qwen3-ASR

[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) is an audio transcription model served through the OpenAI-compatible `/v1/audio/transcriptions` endpoint. It accepts one uploaded audio file per request and returns text.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download the model:

```bash
hf download Qwen/Qwen3-ASR-1.7B
```

## Server Configuration

Qwen3-ASR runs a single ASR stage on one GPU.
Async decode is enabled by default for decode batches of at least two requests,
allowing the shared one-step-lookahead path to overlap host-side result
processing with the next GPU decode forward. Use `--decode-mode sync` to disable
it, or tune the crossover with `--async-lookahead-min-batch-size`.

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --port 8000
```

For example, force synchronous decode when comparing modes:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --decode-mode sync \
  --port 8000
```

## Transcribe Audio

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F file=@tests/data/query_to_cars.wav \
  -F language=en \
  -F response_format=json
```

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "Qwen/Qwen3-ASR-1.7B",
            "language": "en",
            "response_format": "json",
        },
        files={"file": ("query_to_cars.wav", f, "audio/wav")},
        timeout=300,
    )

resp.raise_for_status()
print(resp.json()["text"])
```

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file uploaded as multipart form data |
| `model` | string | server default | Model identifier |
| `language` | string | `en` | Language hint; `zh`/`cn` select Chinese, other values use English prompting |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text` |
| `temperature` | float | `0.01` effective | Sampling temperature; `0` is converted to near-greedy `0.01` |
| `max_new_tokens` | integer | automatic | Explicit positive transcription-token budget |

`verbose_json` is accepted, but currently returns the same minimal JSON shape as `json`:
`{"text": "..."}`.

## Long Audio

The default stage supports Qwen3-ASR's native 1200-second single-forward
envelope without truncation or transcript stitching. Its automatic output
budget keeps the upstream 4096-token floor and grows to 12000 tokens at that
boundary. Context and prefill capacity include the corresponding 15600 audio
tokens and tokenized prompt overhead.

Operators can pin `max_new_tokens` or change the guaranteed native envelope
with the ASR stage's `max_audio_s` factory argument. A request-level
`max_new_tokens` takes precedence over the operator default. The scheduler
still clamps an oversized explicit request to the real per-request context and
KV capacity instead of rejecting a request that otherwise fits.

## Benchmarking

Use `benchmarks/eval/benchmark_asr_seedtts.py` to sweep ASR concurrency on
SeedTTS reference audio through `/v1/audio/transcriptions`. It defaults to
`--model-path Qwen/Qwen3-ASR-1.7B`; the shared request and metric logic lives in
`benchmarks.tasks.asr` and also supports Fun-ASR through `--model-path`.
The report includes RTF (processing time divided by audio duration) and RTFx
(successful input-audio seconds divided by wall-clock seconds).

```bash
sgl-omni serve --model-path Qwen/Qwen3-ASR-1.7B --port 8000

# Sweep the full SeedTTS EN set (1088 clips) at 1..64 concurrency, 3 repeats:
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --concurrencies 1,2,4,8,16,32,64 --repeats 3 --warmup
```

The ASR CI gate runs the selected ASR CI model preset on this same benchmark
entry point (`tests/test_model/test_asr_ci_seedtts.py`). Qwen3-ASR remains
the transcriber for the TTS and talker WER stages.

## Known Limitations

- The endpoint accepts one uploaded file per request.
- `prompt` is accepted by the HTTP endpoint for OpenAI compatibility, but Qwen3-ASR currently ignores it.
- Audio is resampled to 16 kHz before transcription.
- The automatic output budget is a bounded guard, not an EOS guarantee for
  pathological or highly repetitive audio. Set `max_new_tokens` explicitly
  when a workload needs a larger budget and the configured context permits it.
