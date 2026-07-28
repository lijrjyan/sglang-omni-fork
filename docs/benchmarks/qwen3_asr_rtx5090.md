# Qwen3-ASR 1.7B on one RTX 5090

This page records an independent Qwen3-ASR 1.7B BF16 validation on one
32 GB RTX 5090. It covers full SeedTTS English and Chinese quality sweeps,
concurrency scaling, long audio, memory and power sampling, request lifecycle
checks, a 30-minute mixed workload, and bounded cleanup.

The complete benchmark report and exit criteria are recorded in #1212. This
documentation change publishes target-hardware evidence only and does not
duplicate runtime changes from #1154, #1176, or #1193.

## Environment

| Item | Value |
| --- | --- |
| Platform | Linux 6.8, x86_64, glibc 2.39 |
| GPU | 1× NVIDIA GeForce RTX 5090, 32,607 MiB, SM120 |
| GPU power limit | 600 W |
| Driver / CUDA | 580.65.06 / 13.0 |
| PyTorch | 2.11.0+cu130 |
| SGLang | 0.5.12.post1 |
| SGLang-Omni | 0.1.0 |
| Transformers | 5.6.0 |
| FlashInfer | 0.6.11.post1 |
| CPU | AMD Ryzen 9 7950X, 32 logical CPUs |
| Host memory | 124.9 GiB |
| Base commit | `06a6fab66034c17e3aaf346c4cdd24f572d6a5d9` |
| Qualification commit | `cc573def13a96aaf5bb68c4cec7df4d6a4d83c85` |
| Qualification tree | `04d3f052aa03249f64498c617e41fd8c38cb4c66` |
| Model revision | `7278e1e70fe206f11671096ffdd38061171dd6e5` |
| SeedTTS revision | `27f4c1adee83b5b29b7c4b375f6b976324bda308` |
| Long-audio revision | `164d3b41852b1eebe89f1dc0e6e0042f16835ea0` |
| Container image | `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel` |
| Image digest | `sha256:6e8a7a6dedf900096f90190f66f988e7e658cda4f1e6cbc7c17e3a38980a4f89` |

The tested source was a clean frozen qualification stack. Later upstream
heads require separate validation.

## Validated profile

- BF16 with no quantization
- FlashInfer language-model attention
- Triton multimodal attention
- CUDA Graph batches `[1, 2, 4, 8, 12, 16]`
- `max_running_requests=16`
- `mem_fraction_static=0.65`
- `torch.compile` disabled
- `CUDA_VISIBLE_DEVICES=0`

The runtime reported SM120 and did not infer an SM100 policy. Concurrency 32
was queueing stress above the 16-request admission limit, not a 32-request
resident configuration.

Launch the colocated pipeline:

```bash
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --config examples/configs/qwen3_asr_rtx4090.yaml \
  --model-path /path/to/Qwen3-ASR-1.7B \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --host 127.0.0.1 \
  --port 8000 \
  --mem-fraction-static 0.65 \
  --max-running-requests 16 \
  --cuda-graph-max-bs 16
```

The profile filename reflects its original 24 GB target. On the tested RTX
5090, startup logs must report SM120 with FlashInfer and Triton rather than
selecting an SM100-specific policy.

## Quality and performance

Every measured cell ran the complete split three times after warmup and
completed with zero skips and zero request errors.

### SeedTTS English

| Concurrency | Samples/s | Mean latency (s) | P95 latency (s) | Corpus WER range |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 13.766 | 0.073 | 0.090 | 0.01226–0.01241 |
| 2 | 21.506 | 0.093 | 0.115 | 0.01196–0.01211 |
| 4 | 35.013 | 0.114 | 0.146 | 0.01196–0.01211 |
| 8 | 55.770 | 0.143 | 0.186 | 0.01196–0.01211 |
| 16 | 80.001 | 0.199 | 0.259 | 0.01196–0.01203 |
| 32 | 81.072 | 0.391 | 0.466 | 0.01203–0.01211 |

The complete 1,088-clip EN quality range was 0.0120–0.0124, consistent with
the 0.0122 H100 BF16 reference reported in #1151.

### SeedTTS Chinese

| Concurrency | Samples/s | Mean latency (s) | P95 latency (s) | Corpus CER range |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 15.185 | 0.066 | 0.086 | 0.00635–0.00635 |
| 2 | 24.849 | 0.080 | 0.105 | 0.00632–0.00647 |
| 4 | 43.464 | 0.092 | 0.121 | 0.00635–0.00641 |
| 8 | 67.377 | 0.119 | 0.159 | 0.00641–0.00647 |
| 16 | 97.668 | 0.163 | 0.222 | 0.00635–0.00647 |
| 32 | 93.279 | 0.341 | 0.420 | 0.00632–0.00638 |

The complete 2,020-clip ZH quality range was 0.0063–0.0065. No matching
published H100 Qwen3-ASR ZH reference is available.

## Long audio

Post-#1176/#1193 validation used the unmodified
`distil-whisper/librispeech_long` source cropped to 29.9, 30.1, 31, and
60 seconds. `max_new_tokens=512` isolated input preservation from the
128-token stage default.

- 31 seconds: 3/3 HTTP 200 with the same 444-character, 80-word transcript;
  warm latency was 0.648–0.661 seconds.
- 60 seconds: 3/3 HTTP 200 with the same 812-character, 145-word transcript;
  warm latency was 1.230–1.236 seconds.
- 31- and 60-second concurrency `c=1/2/4/8`, three repeats per cell:
  zero request errors, zero timeouts, and deterministic transcripts.
- 124.91 seconds with the default 128-token output budget: 3/3 actionable
  HTTP 400 responses before preprocessing because
  `1637 prompt/audio tokens + 128 max_new_tokens > 1636`.
- Long-audio concurrency cells recorded 22,932 MiB before and after cooldown,
  with no retained-memory delta.

The 29.9- and 30.1-second fixtures each returned 3/3 HTTP 200 but produced the
same 427-character, 76-word transcript. The exact 30.1-second boundary fixture
therefore did not independently prove recognizable content after 30.0
seconds; that boundary case remains tracked by #1173 and is not an RTX 5090
profile failure.

## Memory, startup, and stability

| Checkpoint | Device memory | Process memory |
| --- | ---: | ---: |
| Pre-model | 1,004 MiB | 502 MiB |
| Post-weight-load | 5,161 MiB | 4,506 MiB |
| Post-static/KV allocation | 22,077 MiB | 21,576 MiB |
| Post-CUDA-Graph capture | 22,200 MiB | 21,699 MiB |
| Post-first-request | 22,433 MiB | 21,926 MiB |
| Warm steady state | 22,942 MiB | 22,932 MiB |
| Post-soak cooldown | 22,948 MiB | 22,938 MiB |
| Post-shutdown | 1 MiB | 0 MiB |

Cold and three warm starts each reached health in about 12.53 seconds.

The 1,800-second mixed EN/ZH stability workload completed 98,057/98,057
requests with zero unexpected errors:

| Concurrency | Duration (s) | Requests | Requests/s | Mean latency (s) | P95 latency (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 450.03 | 6,413 | 14.250 | 0.070 | 0.128 |
| 4 | 450.07 | 18,576 | 41.273 | 0.097 | 0.161 |
| 8 | 450.12 | 29,208 | 64.890 | 0.123 | 0.194 |
| 16 | 450.15 | 43,860 | 97.434 | 0.164 | 0.272 |

Every stage covered malformed input, cancellation/reconnect, context
overflow, and health checks. Peak sampled soak device/process memory was
23,445/22,938 MiB, peak sampled soak power was 307.965 W, and final health
returned HTTP 200.

## Functional and lifecycle behavior

- English and Chinese requests passed.
- Streaming and non-streaming final text matched.
- Empty, corrupt, and invalid audio paths returned HTTP 400.
- Context overflow returned actionable HTTP 400 before expensive processing.
- Active cancellation and reconnect were observed while the backend request
  was active.
- In-flight SIGTERM was observed against an active request.
- A fresh recovery server passed health and deterministic sentinel
  transcription.
- Three lifecycle cycles left no listener, descendant process, or GPU process.

Behavior cards B01–B08 passed. B06 safely bounded an intentionally oversized
CUDA Graph request from batch 8192 to 4096 without emitting a warning. The
validated profile itself uses graph batches only through 16.

## Test evidence

- Qualification/core: 188 passed, 1 skipped because the qualification
  container ran as root; the skipped POSIX permission test was rerun as a
  non-root user and passed 1/1.
- FishAudio/dependency contracts: 46 passed.
- Publication archive replay: 107/107 SHA256 entries passed.

## Limitations

- This validates BF16 on one RTX 5090/SM120 host. It does not validate NVFP4,
  FP8, quantization, TP, DP, multi-GPU, Windows, or WSL.
- Concurrency 32 is queueing stress above the 16-request resident limit.
- B06 bounded the oversized graph request without emitting a warning.
- The recorded SSE path emitted only a terminal transcript event. Incremental
  partial-transcript cadence remains blocked by the current upstream
  Qwen3-ASR streaming implementation.

## Raw artifacts

https://gist.github.com/lijrjyan/283f96693148b0fcb365daef3b4cc2ae
