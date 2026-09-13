# SPDX-License-Identifier: Apache-2.0
"""Generic SGLang bootstrap utilities for model-specific schedulers."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Protocol

from sglang_omni.scheduling.speculative_support import (
    DraftGraphKind,
    draft_graph_kind,
    is_omni_supported_speculative_algorithm,
    unsupported_speculative_algorithm_message,
)
from sglang_omni.utils.gpu_compat import (
    get_visible_gpu_sm_version,
    gpu_architecture_for_sm,
)

logger = logging.getLogger(__name__)


class _SGLangServerArgsForDiagnostics(Protocol):
    attention_backend: str | None
    sampling_backend: str | None

    def get_attention_backends(self) -> tuple[str | None, str | None]: ...


def _describe_sglang_runtime_configuration(
    server_args: _SGLangServerArgsForDiagnostics,
    gpu_id: int,
) -> str:
    sm_version = get_visible_gpu_sm_version(gpu_id)
    prefill_attention_backend, decode_attention_backend = (
        server_args.get_attention_backends()
    )
    return (
        f"SGLang runtime configuration: gpu_id={gpu_id}, sm={sm_version}, "
        f"architecture={gpu_architecture_for_sm(sm_version)}, "
        f"attention_backend={server_args.attention_backend}, "
        f"decode_attention_backend={decode_attention_backend}, "
        f"prefill_attention_backend={prefill_attention_backend}, "
        f"sampling_backend={server_args.sampling_backend}"
    )


def init_sglang_cuda_graphs(model_worker: Any) -> None:
    """Initialize SGLang graphs with Omni's prefill-embedding capture view."""
    from sglang.srt.utils.tensor_bridge import use_mlx

    if use_mlx():
        # note (yexiaodong): The MLX stub has no Torch graph lifecycle because
        # native MLX lazy evaluation owns graph execution.
        return
    if not model_worker.enable_prefill_input_embeds:
        # Required even when graphs are disabled: SGLang installs its eager
        # phase runner from init_cuda_graphs().
        model_worker.model_runner.init_cuda_graphs()
    else:
        model_config = model_worker.model_config
        original_is_multimodal = model_config.is_multimodal
        # note (Junnan Li): attention backends have already captured the real
        # modality; only the prefill graph runner needs this temporary view so
        # it allocates the upstream input_embeds replay buffer.
        model_config.is_multimodal = True
        try:
            model_worker.model_runner.init_cuda_graphs()
        finally:
            model_config.is_multimodal = original_is_multimodal

    spec_algorithm = getattr(model_worker.model_runner, "spec_algorithm", None)
    if spec_algorithm is not None and is_omni_supported_speculative_algorithm(
        spec_algorithm
    ):
        runner = model_worker.model_runner.decode_cuda_graph_runner
        logger.info(
            "%s speculative graph startup: target_verify=%s",
            spec_algorithm.name,
            "captured" if runner is not None else "eager",
        )


def _inner_draft_worker(draft_worker: Any) -> Any:
    """Return the per-step draft worker wrapped by a native V2 spec worker."""
    inner = getattr(draft_worker, "draft_worker", None)
    if inner is None:
        inner = getattr(draft_worker, "_draft_worker", None)
    if inner is None:
        raise AttributeError(
            f"{type(draft_worker).__name__} exposes no inner draft worker"
        )
    return inner


def speculative_draft_runner(draft_worker: Any) -> Any:
    runner = getattr(draft_worker, "draft_model_runner", None)
    if runner is not None:
        return runner
    return _inner_draft_worker(draft_worker).draft_runner


def speculative_draft_model(draft_worker: Any) -> Any:
    """Return the draft model owned by a native speculative worker."""
    return speculative_draft_runner(draft_worker).model


def init_speculative_draft_cuda_graphs(
    draft_worker: Any, *, capture_draft_decode_graph: bool, spec_algorithm: Any = None
) -> None:
    """Install the draft runner's graph state (one helper for every algorithm).

    Dispatch on the worker's ``init_cuda_graphs`` signature first: the Whisper
    encoder-decoder patch exposes explicit capture selection. A stock worker
    takes no keywords, so when draft decode capture is disabled the algorithm's
    draft graph kind decides how to keep the eager runner initialised without
    capturing: block drafts (DFLASH) initialise the inner worker with capture
    off; chain drafts (STANDALONE/EAGLE3) only capture decode for multiple
    steps, so the step count is lowered to 1 around the call and restored in
    ``finally``. Draft-extend initialisation is preserved in every branch.
    """
    init_cuda_graphs = draft_worker.init_cuda_graphs
    parameters = inspect.signature(init_cuda_graphs).parameters
    if "capture_draft_decode_graph" in parameters:
        init_cuda_graphs(
            capture_draft_decode_graph=capture_draft_decode_graph,
            capture_draft_extend_graph=False,
        )
    elif capture_draft_decode_graph:
        init_cuda_graphs()
    elif (
        draft_graph_kind(draft_worker.server_args.speculative_algorithm)
        is DraftGraphKind.VERIFY
    ):
        _inner_draft_worker(draft_worker).init_cuda_graphs(
            capture_decode_cuda_graph=False
        )
    else:
        inner = _inner_draft_worker(draft_worker)
        original_steps = inner.speculative_num_steps
        inner.speculative_num_steps = 1
        try:
            init_cuda_graphs()
        finally:
            inner.speculative_num_steps = original_steps
    inner_draft_worker = _inner_draft_worker(draft_worker)
    draft_decode_runner = getattr(inner_draft_worker, "cuda_graph_runner", None)
    draft_extend_runner = getattr(
        inner_draft_worker, "cuda_graph_runner_for_draft_extend", None
    )
    logger.info(
        "%s speculative graph startup: draft_decode=%s, draft_extend=%s",
        draft_worker.server_args.speculative_algorithm,
        "captured" if draft_decode_runner is not None else "eager",
        "captured" if draft_extend_runner is not None else "eager",
    )


def _hidden_capture_max_tokens(server_args: Any) -> int:
    """Largest token-row count a single thinker forward can produce.

    Covers chunked prefill, non-chunked prefill, decode batches, and every
    configured CUDA graph bucket maximum, so the capture buffers are large
    enough for both eager forwards and graph replay.
    """
    chunked_prefill_size = server_args.chunked_prefill_size
    candidates: list[Any] = []
    if chunked_prefill_size is not None and chunked_prefill_size > 0:
        candidates.append(chunked_prefill_size)
    else:
        candidates.append(server_args.max_prefill_tokens)
        # Note(wenyao): Without chunking, SGLang always admits the first prefill request even
        # when it exceeds the batch token budget, up to the model context bound.
        candidates.append(server_args.context_length)
    candidates.append(server_args.max_running_requests)
    candidates.append(server_args.cuda_graph_config.decode.max_bs)
    candidates.append(server_args.cuda_graph_config.prefill.max_bs)

    positive = [int(value) for value in candidates if value is not None and value > 0]
    if not positive:
        raise ValueError(
            "Cannot derive hidden capture capacity: none of chunked_prefill_size, "
            "max_prefill_tokens, context_length, max_running_requests, or the "
            "CUDA graph batch maxima is positive"
        )
    return max(positive)


def _parse_speculative_algorithm(server_args: Any) -> Any:
    """Parse and constrain Omni's native speculative algorithm."""
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    algorithm = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)
    if not algorithm.is_none() and not is_omni_supported_speculative_algorithm(
        algorithm
    ):
        raise ValueError(
            unsupported_speculative_algorithm_message(server_args.speculative_algorithm)
        )
    return algorithm


def create_sglang_infrastructure(
    server_args: Any,
    gpu_id: int,
    *,
    tp_rank: int = 0,
    nccl_port: int | None = None,
    model_arch_override: str | None = None,
    weight_prefix: str | None = None,
    capture_hidden_layers: list[int] | None = None,
    total_gpu_memory_fraction: float | None = None,
    defer_cuda_graph_capture: bool = False,
    enable_prefill_input_embeds: bool = False,
    target_tokenizer: Any = None,
):
    """Create SGLang worker, memory pools, and tree cache."""
    # ModelRunner.__init__ publishes server_args as the process-wide runtime
    # context; publishing again would silently reconfigure whatever already runs
    # here, so an engine is only built where the context is unpublished. A
    # construction that failed after publishing is therefore not retried here.
    from sglang.srt.runtime_context import get_context

    from sglang_omni.model_runner.model_worker import (
        ModelWorker,
        ModelWorkerConfig,
        SpeculativeTargetWorkerAdapter,
        register_speculative_omni_models,
    )
    from sglang_omni.scheduling.sglang_backend import create_tree_cache
    from sglang_omni.scheduling.stage_kv_budget import consume_stage_kv_cache_bytes

    if get_context().is_config_namespace_published("model"):
        raise RuntimeError(
            "this process already holds a published SGLang runtime context; "
            "an SGLang AR engine must own its OS process. Place SGLang AR "
            "stages in separate processes."
        )

    logger.info(_describe_sglang_runtime_configuration(server_args, gpu_id))

    spec_algorithm = _parse_speculative_algorithm(server_args)
    if not spec_algorithm.is_none():
        register_speculative_omni_models(model_arch_override)
        if not server_args.disable_overlap_schedule:
            raise ValueError(
                "Omni speculative decoding requires disable_overlap_schedule=True"
            )

    kv_cache_bytes = consume_stage_kv_cache_bytes()
    worker_config = ModelWorkerConfig(
        model_arch_override=model_arch_override,
        weight_prefix=weight_prefix,
        nccl_port=nccl_port,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        kv_cache_bytes=kv_cache_bytes,
        enable_prefill_input_embeds=enable_prefill_input_embeds,
    )
    from sglang.srt.utils.tensor_bridge import use_mlx

    if use_mlx():
        # Note (Jiaxin Deng): the MLX worker sizes no SGLang KV pool, so a
        # declared byte budget could only be ignored; refuse instead.
        if kv_cache_bytes is not None:
            raise ValueError(
                "engine.kv_cache_bytes is not supported on the MLX path; "
                "remove it or run this stage on CUDA"
            )
        from sglang_omni.model_runner.mlx_model_worker import create_mlx_model_worker

        model_worker = create_mlx_model_worker(
            config=worker_config,
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
        )
    else:
        model_worker = ModelWorker(
            config=worker_config,
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
        )

    target_worker_adapter = None
    draft_worker = None
    if not spec_algorithm.is_none():
        target_worker_adapter = SpeculativeTargetWorkerAdapter(
            model_worker,
            tokenizer=target_tokenizer,
        )
        draft_worker_class = spec_algorithm.create_worker(server_args)
        draft_worker = draft_worker_class(
            server_args=server_args,
            gpu_id=gpu_id,
            ps=model_worker.model_runner.ps,
            nccl_port=model_worker.model_runner.dist_port,
            target_worker=target_worker_adapter,
        )
        logger.info(
            "Initialized %s speculative workers: target_arch=%s draft_model=%s",
            getattr(server_args, "speculative_algorithm", None),
            model_arch_override,
            server_args.speculative_draft_model_path,
        )

    if capture_hidden_layers:
        from sglang_omni.model_runner._hidden_capture import (
            install_hidden_capture_hooks,
        )

        model = model_worker.model_runner.model
        install_hidden_capture_hooks(
            model,
            capture_hidden_layers,
            max_tokens=_hidden_capture_max_tokens(server_args),
        )

    # Phase order follows upstream Scheduler.init_model_worker().
    model_runner = model_worker.model_runner
    if draft_worker is not None:
        # Price the loaded draft's KV layers; encoder layers have no persistent KV.
        model_runner.spec_aux_config.eagle_draft_num_layers = speculative_draft_runner(
            draft_worker
        ).layer_info.num_effective_layers
    model_runner.alloc_memory_pool()
    if draft_worker is not None:
        req_to_token_pool, token_to_kv_pool_allocator = model_worker.get_memory_pool()
        draft_worker.alloc_memory_pool(
            memory_pool_config=model_runner.memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
    model_runner.init_attention_backends()
    if draft_worker is not None:
        draft_worker.init_attention_backends()

    if not defer_cuda_graph_capture:
        init_sglang_cuda_graphs(model_worker)
        if draft_worker is not None:
            # note (Junnan Li): with graphs disabled this is the draft
            # eager-runner hook.
            init_speculative_draft_cuda_graphs(
                draft_worker,
                capture_draft_decode_graph=False,
                spec_algorithm=spec_algorithm,
            )

    req_to_token_pool, token_to_kv_pool_allocator = model_worker.get_memory_pool()

    tree_cache = create_tree_cache(
        server_args,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        server_args.page_size,
    )

    infrastructure = (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        model_worker.model_config,
    )
    if draft_worker is not None:
        return (*infrastructure, target_worker_adapter, draft_worker)
    return infrastructure


# note (luojiaxuan): Some Omni generation stages cannot capture CUDA graphs
# immediately during infrastructure construction. At that point the shared
# request pools exist, but stage-owned decode state may not: speech tokenizers
# may still need to be attached, sampler or feedback buffers may not be
# allocated, stage-local decode helpers may not be compiled, and the
# model-specific buffer capacity may not yet have been checked against the
# serving batch policy. Capturing before that work would freeze replay around an
# incomplete decode path and can make later steady-state requests either miss the
# intended graph buckets or overrun model-side per-request buffers. The capture
# priority should be the hot path users repeatedly pay for under concurrency:
# decode batches admitted by max_running_requests, capped by cuda_graph_max_bs
# and request-token slots, with all per-request model buffers already allocated.
# One-time bootstrap work such as processor loading, cache construction, audio
# decoder/vocoder setup, and other host-side staging should stay outside CUDA
# graph coverage because graph replay will not amortize it.
def create_sglang_infrastructure_defer_cuda_graph(
    server_args: Any,
    gpu_id: int,
    **kwargs: Any,
):
    """Build shared SGLang infrastructure while deferring CUDA graph capture.

    The caller finishes stage-specific decode setup, then runs
    init_cuda_graphs() only when this returns that CUDA graphs were requested.
    """
    want_cuda_graph = not bool(server_args.disable_cuda_graph)
    infrastructure = create_sglang_infrastructure(
        server_args,
        gpu_id,
        defer_cuda_graph_capture=want_cuda_graph,
        **kwargs,
    )
    return want_cuda_graph, infrastructure
