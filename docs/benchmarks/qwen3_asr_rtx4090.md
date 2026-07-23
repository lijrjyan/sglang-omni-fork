# Qwen3-ASR 1.7B on RTX 4090

This report records a reproducible Qwen3-ASR 1.7B BF16 validation on one
24 GB RTX 4090. It covers full SeedTTS English and Chinese quality sweeps,
concurrency scaling, memory sampling, request lifecycle checks, a 30-minute
mixed workload, and bounded cleanup.

## Implementation dependencies

The reproduction commands use work proposed in the incremental validation
series:

- [#1154](https://github.com/sgl-project/sglang-omni/pull/1154) provides the
  Qwen3-ASR SM89 profile and input guards.
- [#1155](https://github.com/sgl-project/sglang-omni/pull/1155) provides the
  pinned benchmark provenance, resource monitor, and stability harness.
- [#1152](https://github.com/sgl-project/sglang-omni/pull/1152) is the umbrella
  draft that preserves the complete validation stack.

The current base does not contain
`examples/configs/qwen3_asr_rtx4090.yaml` or
`benchmarks/eval/benchmark_asr_stability.py`. Apply the corresponding
incremental changes before running the commands in this report. This
documentation change does not add or duplicate those implementation files.

## Environment

| Item | Value |
| --- | --- |
| Platform | Linux 6.8, glibc 2.39 |
| GPU | 1× NVIDIA GeForce RTX 4090, 24,564 MiB, SM89 |
| GPU power limit | 400 W current, 480 W VBIOS default |
| Driver / CUDA | 590.48.01 / 13.0 |
| PyTorch | 2.11.0+cu130 |
| SGLang | 0.5.12.post1 |
| SGLang-Omni | 0.1.0 |
| Transformers | 5.6.0 |
| CPU | AMD EPYC 7402 24-Core Processor |
| Source commit | `e61712b29b2f373e7d2b77236858777d97e52afa` |
| Source tree | `77a686c5e514efac2a17646bc8f855688c198e35` |
| Model revision | `7278e1e70fe206f11671096ffdd38061171dd6e5` |
| SeedTTS revision | `27f4c1adee83b5b29b7c4b375f6b976324bda308` |

The source checkout was clean. The model and dataset were materialized at the
locked revisions and served from local snapshots.

## Validated profile

- BF16
- FlashInfer LM attention
- Triton multimodal attention
- CUDA Graph enabled
- `torch.compile` disabled
- `max_running_requests=16`
- `mem_fraction_static=0.65`

Launch the complete colocated pipeline:

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

sgl-omni serve \
  --config examples/configs/qwen3_asr_rtx4090.yaml \
  --model-path /path/to/Qwen3-ASR-1.7B@7278e1e \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port 8000
```

Wait for `GET http://127.0.0.1:8000/health` to return HTTP 200.

## Reproduction

The full sweep uses the pinned local SeedTTS materialization. Each concurrency
level runs three measured repetitions after warmup.

### English

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 \
  --meta /path/to/seedtts/en/meta.lst \
  --lang en \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --model-revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --concurrencies 1,2,4,8,16,32 \
  --repeats 3 \
  --warmup \
  --dtype bfloat16 \
  --attention-backend flashinfer \
  --mm-attention-backend triton_attn \
  --cuda-graph \
  --no-torch-compile \
  --max-running-requests 16 \
  --mem-fraction-static 0.65 \
  --monitor-interval-s 0.2 \
  --launch-command 'CUDA_VISIBLE_DEVICES=0 sgl-omni serve --config examples/configs/qwen3_asr_rtx4090.yaml --model-path <locked-local-snapshot> --model-name Qwen/Qwen3-ASR-1.7B --port 8000' \
  --output asr-seedtts-en-full.json
```

### Chinese

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 \
  --meta /path/to/seedtts/zh/meta.lst \
  --lang zh \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --model-revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --concurrencies 1,2,4,8,16,32 \
  --repeats 3 \
  --warmup \
  --dtype bfloat16 \
  --attention-backend flashinfer \
  --mm-attention-backend triton_attn \
  --cuda-graph \
  --no-torch-compile \
  --max-running-requests 16 \
  --mem-fraction-static 0.65 \
  --monitor-interval-s 0.2 \
  --launch-command 'CUDA_VISIBLE_DEVICES=0 sgl-omni serve --config examples/configs/qwen3_asr_rtx4090.yaml --model-path <locked-local-snapshot> --model-name Qwen/Qwen3-ASR-1.7B --port 8000' \
  --output asr-seedtts-zh-full.json
```

English scoring uses `whisper.normalizers.EnglishTextNormalizer`. Chinese
scoring removes Chinese and ASCII punctuation and spaces before computing CER.

### Stability

```bash
python -m benchmarks.eval.benchmark_asr_stability \
  --host 127.0.0.1 \
  --port 8000 \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --model-revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --meta /path/to/seedtts \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --duration-s 1800 \
  --concurrencies 1,4,8,16 \
  --samples-per-language 20 \
  --request-timeout-s 60 \
  --gpu-index 0 \
  --monitor-interval-s 0.2 \
  --launch-command 'CUDA_VISIBLE_DEVICES=0 sgl-omni serve --config examples/configs/qwen3_asr_rtx4090.yaml --model-path <locked-local-snapshot> --model-name Qwen/Qwen3-ASR-1.7B --port 8000' \
  --output asr-stability-1800.json
```

The stability run includes known-good English and Chinese requests,
streaming/non-streaming consistency, cancellation/reconnect, malformed and
boundary-duration audio, periodic health checks, and memory checkpoints.
After the run, stop the process group with SIGTERM and confirm that the HTTP
listener and GPU compute processes are gone.

## Quality and performance

All measured repetitions completed without a reported skip or benchmark error.
English processed 1,088/1,088 clips per repetition; Chinese processed
2,020/2,020 clips per repetition.

### SeedTTS English

| Concurrency | Samples/s | Mean latency (s) | P95 latency (s) | Mean RTF | Corpus WER |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.001 | 0.125 | 0.154 | 0.0270 | 0.01223 |
| 2 | 12.468 | 0.160 | 0.198 | 0.0347 | 0.01226 |
| 4 | 18.968 | 0.210 | 0.270 | 0.0456 | 0.01221 |
| 8 | 28.686 | 0.278 | 0.366 | 0.0602 | 0.01218 |
| 16 | 39.721 | 0.401 | 0.527 | 0.0869 | 0.01226 |
| 32 | 37.257 | 0.852 | 1.023 | 0.1858 | 0.01221 |

The observed WER range, `0.01218–0.01226`, matches the `0.0122` H100 BF16
reference reported in
[#1151](https://github.com/sgl-project/sglang-omni/issues/1151) at the
published precision.

### SeedTTS Chinese

| Concurrency | Samples/s | Mean latency (s) | P95 latency (s) | Mean RTF | Corpus CER |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.799 | 0.113 | 0.142 | 0.0245 | 0.00619 |
| 2 | 13.656 | 0.146 | 0.178 | 0.0316 | 0.00617 |
| 4 | 21.032 | 0.190 | 0.252 | 0.0410 | 0.00617 |
| 8 | 31.978 | 0.250 | 0.346 | 0.0538 | 0.00617 |
| 16 | 44.695 | 0.357 | 0.489 | 0.0770 | 0.00617 |
| 32 | 42.190 | 0.755 | 0.939 | 0.1633 | 0.00617 |

No matching Qwen3-ASR H100 Chinese reference is currently published.

## Memory and stability

- Idle before launch: 1 MiB.
- Static-allocation checkpoint: 16,368 MiB.
- Ready checkpoint: 16,516 MiB.
- Quality-sweep device peak: 17,370 MiB.
- Quality-sweep process peak: 16,880 MiB.
- At least 7,080 MiB was free at the recorded soak stage boundaries.

Resource samples were recorded every 200 ms.
The highest sampled draw was 235.694 W in the quality sweep and 233.099 W in
the stability run. The benchmark did not change the pre-existing 400 W limit,
and neither workload approached it.

The 1,800-second mixed workload completed 47,486/47,486 requests with zero
unexpected errors:

| Concurrency | Duration (s) | Requests | Requests/s | Mean latency (s) | P95 latency (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 450.02 | 3,679 | 8.175 | 0.122 | 0.206 |
| 4 | 450.18 | 9,002 | 19.997 | 0.200 | 0.312 |
| 8 | 450.24 | 14,172 | 31.477 | 0.254 | 0.407 |
| 16 | 450.19 | 20,633 | 45.831 | 0.349 | 0.572 |

All eight functional checks and all 56 injected malformed-input and
cancellation/reconnect events passed. Final health returned HTTP 200. SIGTERM
cleanup left no listener or GPU compute process and returned the device to
1 MiB.

## Limitations

- Qwen3-ASR accepts at most 30 seconds of audio per request.
- The recorded SSE reconnect path passed, but this model emitted only the
  terminal transcript event during the run.
- FA4 was installed but was not selected on SM89; this profile used
  FlashInfer/Triton.
- Resource metrics describe the recorded isolated run. They are not a general
  memory-retention threshold for unrelated workloads.
- Aggregate JSON supports the public delivery level established in #1151.
  Retaining per-request hypotheses for independent transcript-level
  recomputation is a useful future harness improvement.

## Raw JSON artifacts

https://gist.github.com/lijrjyan/43dcc9772082dabcaeeaaf356b9d3cf5
