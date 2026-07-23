# [Benchmark] Validate Qwen3-ASR 1.7B on one RTX 4090

## Motivation

This publishes a reproducible Qwen3-ASR 1.7B BF16 validation on one 24 GB
RTX 4090. The result covers full SeedTTS English and Chinese quality sweeps,
concurrency scaling, memory sampling, request lifecycle checks, a 30-minute
mixed workload, and bounded cleanup.

The claim is intentionally specific to the recorded Linux environment and
frozen source, model, and dataset revisions.

### Validation rationale

The consumer-GPU roadmap needs at least one complete ASR result on a single
RTX 4090 before broader profiles or architecture claims are made. Model startup
alone is insufficient: the complete colocated pipeline must also preserve
quality, remain within 24 GB, handle error and cancellation paths, sustain a
mixed workload, and release its resources after shutdown.

### Scope

Included:

- one colocated Qwen3-ASR 1.7B worker on one RTX 4090;
- full SeedTTS EN and ZH evaluation;
- concurrency `1/2/4/8/16/32`, with warmup and three measured repetitions;
- 200 ms resource sampling and startup/steady-state memory checkpoints;
- streaming, non-streaming, malformed-input, duration-boundary,
  cancellation/reconnect, health, and shutdown checks;
- a 1,800-second mixed EN/ZH workload.

Not included:

- multi-GPU, tensor parallel, stage placement, or router changes;
- RTX 5090 or SM120 claims;
- quantization;
- changes to serving behavior or model kernels.

This result-document change depends on the Qwen3-ASR SM89 profile and input
guards in #1154 and the benchmark/resource/stability harness in #1155. The
complete validation stack is preserved by the umbrella draft #1152. The
current base does not contain `examples/configs/qwen3_asr_rtx4090.yaml` or
`benchmarks/eval/benchmark_asr_stability.py`; this PR does not imply that its
reproduction commands run without those dependencies.

## Modifications

- Add `docs/benchmarks/qwen3_asr_rtx4090.md` with the frozen environment,
  copyable launch and benchmark commands, quality/performance tables, memory
  checkpoints, stability results, artifact placeholder, and limitations.
- Register the result page in the Benchmarks toctree.
- Keep the implementation dependency explicit instead of duplicating the
  profile or benchmark harness from #1154 and #1155.

### Environment

| Item | Value |
| --- | --- |
| Platform | Linux 6.8, glibc 2.39 |
| GPU | 1× NVIDIA GeForce RTX 4090, 24,564 MiB, SM89 |
| GPU power limit | 400 W |
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

The source checkout was clean. Model and dataset inputs were materialized from
the locked revisions and served offline.

Validated runtime settings:

- BF16;
- FlashInfer LM attention and Triton multimodal attention;
- CUDA Graph enabled and `torch.compile` disabled;
- `max_running_requests=16`;
- `mem_fraction_static=0.65`.

## Related Issues

- Roadmap: #1120
- Reference validation report: #1151
- Follow-up implementation tracking: #1152

## Accuracy Test

The full SeedTTS evaluation completed 1,088/1,088 English clips and 2,020/2,020
Chinese clips at every measured concurrency and repetition, with no reported
skip or benchmark error. English corpus WER was `0.01218–0.01226`, matching the
`0.0122` H100 BF16 reference in #1151 at the published precision. Chinese
corpus CER was `0.00617–0.00619`; no matching H100 Chinese reference is
currently published.

## Benchmark & Profiling

The result page records three measured repetitions after warmup at concurrency
`1/2/4/8/16/32`, 200 ms resource sampling, phase memory checkpoints, and a
1,800-second mixed workload. The detailed tables and reproduction commands are
in the Results and Validation sections below.

### Results

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
reference reported in #1151 at the published precision.

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

### Memory and stability

- Idle before launch: 1 MiB.
- Static-allocation checkpoint: 16,368 MiB.
- Ready checkpoint: 16,516 MiB.
- Quality-sweep device peak: 17,370 MiB.
- Quality-sweep process peak: 16,880 MiB.
- At least 7,080 MiB was free at the recorded soak stage boundaries.

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

### Validation

Fresh local verification recomputed the aggregate report from the canonical
JSON artifacts and checked:

- six concurrency levels and three measured repetitions for both languages;
- sample counts, WER/CER, throughput, mean/P95 latency, and RTF;
- 1,800-second duration and 47,486 successful requests;
- `8/8` functional checks and `56/56` chaos events;
- final HTTP 200 health;
- primary artifact and package checksums.

The public result page is
[`docs/benchmarks/qwen3_asr_rtx4090.md`](../../../../docs/benchmarks/qwen3_asr_rtx4090.md).

The local pre-publication evidence package contains 17 entries and has
SHA256:

```text
7e247330207a34c65818ba73535023625b952cf78391a44ea566007a94cc14ed
```

It retains absolute asset paths and run provenance for internal audit and must
not be uploaded as the public artifact. Raw artifacts:
`<PUBLIC_ARTIFACT_URL>`. Replace this placeholder only with a separately
sanitized package and its post-sanitization checksums.

### Limitations

- This validates one exact Linux, RTX 4090, and dependency stack; it does not
  claim support for every RTX 4090 host.
- Qwen3-ASR accepts at most 30 seconds of audio per request.
- The recorded SSE reconnect path passed, but this model emitted only the
  terminal transcript event during the run.
- FA4 was installed but was not selected on SM89; this profile used
  FlashInfer/Triton.
- Resource metrics describe the recorded isolated run. They are not a general
  memory-retention threshold for unrelated workloads.
- Aggregate JSON supports the same public delivery level as #1151. Retaining
  per-request hypotheses for independent transcript-level recomputation is a
  useful future harness improvement, not part of this result-only change.

### Test plan

- [x] Verify locked source, model, and dataset revisions.
- [x] Run full SeedTTS EN and ZH at concurrency `1/2/4/8/16/32`.
- [x] Run three measured repetitions after warmup.
- [x] Compare English WER with the published H100 BF16 reference.
- [x] Record 200 ms resource samples and phase memory checkpoints.
- [x] Exercise non-streaming, streaming, cancellation/reconnect, malformed
      input, and duration-boundary paths.
- [x] Complete a 30-minute mixed workload.
- [x] Verify final health, bounded SIGTERM cleanup, and GPU-process release.
- [x] Recompute reported aggregates from canonical JSON.
- [x] Verify raw-artifact and publication-package checksums.

## Checklist

- [x] Format your code according with pre-commit.
- [x] Add unit tests. (Not applicable: this is a documentation-only result
      change; the exercised benchmark harness is tracked in #1155.)
- [x] Update documentation / docstrings / example tutorials as needed.
- [x] Provide throughput / latency benchmark results and accuracy evaluation
      results as needed.
- [ ] For reviewers: If you haven't made any contributions to this PR and are
      only assisting with merging the main branch, please remove yourself as a
      co-author when merging the PR.

Before publication, replace `<PUBLIC_ARTIFACT_URL>` with the URL and checksum
of the separately sanitized evidence package. Maintainer review is required
before treating this recorded result as an official support claim.

## CI

This is a documentation-only result change. The focused MyST parse, toctree
registration, relative-link checks, sensitive-content scan, and
`git diff --check` pass locally. Repository pre-commit hooks are included in
the local test plan. GPU CI remains maintainer-triggered through the repository
`run-ci` label policy.
