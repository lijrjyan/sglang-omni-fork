# SPDX-License-Identifier: Apache-2.0
"""Builders for SGLang-backed autoregressive engine stages."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from numbers import Integral
from typing import Any, ClassVar

from sglang_omni.scheduling.generation_batch_policy import (
    CudaGraphBackend,
    build_generation_batch_overrides,
    get_prefill_cuda_graph_backend,
    nested_prefill_overrides,
    validate_generation_batch_policy,
)
from sglang_omni.scheduling.speculative_support import (
    OMNI_SPECULATIVE_ALGORITHMS,
    unsupported_speculative_algorithm_message,
)
from sglang_omni.utils.checkpoint import resolve_checkpoint as _resolve_checkpoint

logger = logging.getLogger(__name__)


def _operator_selected_prefill_graph_backend(
    server_args_overrides: Mapping[str, Any] | None,
) -> bool:
    if not server_args_overrides:
        return False
    if "cuda_graph_backend_prefill" in server_args_overrides:
        return True
    return "backend" in nested_prefill_overrides(server_args_overrides)


def _normalize_context_length(value: Any, *, model_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(
            f"{model_name} context length must be a positive integer, got {value!r}"
        )
    context_length = int(value)
    if context_length <= 0:
        raise ValueError(
            f"{model_name} resolved an invalid context length: {context_length}"
        )
    return context_length


_SPECULATIVE_OVERRIDE_FIELDS = (
    "speculative_draft_model_path",
    "speculative_num_steps",
    "speculative_eagle_topk",
    "speculative_num_draft_tokens",
)


def configure_speculative_overrides(
    overrides: Mapping[str, Any],
    *,
    enable_speculative: bool,
    speculative_draft_model_path: str | None,
    speculative_num_steps: int,
    speculative_num_draft_tokens: int,
    speculative_cuda_graph: bool = False,
    speculative_draft_cuda_graph: bool = False,
    speculative_algorithm: str = "STANDALONE",
    speculative_prefill_cuda_graph: bool = False,
    speculative_overlap: bool = False,
) -> dict[str, Any]:
    """Validate Omni's native speculative decoding configuration.

    STANDALONE and EAGLE3 drafts run ``speculative_num_steps`` chain steps and
    verify ``steps + 1`` tokens; DFLASH verifies one fixed block of
    ``speculative_num_draft_tokens`` and takes no step count.
    """
    configured = dict(overrides)
    if speculative_draft_cuda_graph and not speculative_cuda_graph:
        raise ValueError(
            "speculative_draft_cuda_graph=True requires speculative_cuda_graph=True"
        )
    if speculative_prefill_cuda_graph and not speculative_cuda_graph:
        raise ValueError(
            "speculative_prefill_cuda_graph=True requires speculative_cuda_graph=True"
        )
    if enable_speculative:
        first_class_algorithm = str(speculative_algorithm).upper()
        if first_class_algorithm not in OMNI_SPECULATIVE_ALGORITHMS:
            raise ValueError(
                unsupported_speculative_algorithm_message(speculative_algorithm)
            )
        requested = {
            "speculative_algorithm": first_class_algorithm,
            "speculative_draft_model_path": speculative_draft_model_path,
            "speculative_eagle_topk": 1,
            "speculative_num_draft_tokens": speculative_num_draft_tokens,
        }
        if first_class_algorithm != "DFLASH":
            requested["speculative_num_steps"] = speculative_num_steps
        for name, value in requested.items():
            existing = configured.get(name)
            conflicts = existing is not None and existing != value
            if name == "speculative_algorithm" and existing is not None:
                conflicts = str(existing).upper() != value
            if conflicts:
                raise ValueError(
                    f"Conflicting first-class {name}={value!r} and "
                    f"server_args_overrides {name}={existing!r}"
                )
            configured[name] = value

    algorithm = configured.get("speculative_algorithm")
    if algorithm is None:
        if speculative_overlap:
            raise ValueError("speculative_overlap requires native DFLASH decoding")
        configured_fields = [
            name
            for name in _SPECULATIVE_OVERRIDE_FIELDS
            if configured.get(name) is not None
        ]
        if configured_fields:
            raise ValueError(
                "speculative_algorithm='STANDALONE' is required when setting "
                + ", ".join(configured_fields)
            )
        if speculative_cuda_graph:
            raise ValueError(
                "speculative_cuda_graph=True requires native speculative decoding"
            )
        return configured

    normalized_algorithm = str(algorithm).upper()
    if normalized_algorithm not in OMNI_SPECULATIVE_ALGORITHMS:
        raise ValueError(unsupported_speculative_algorithm_message(algorithm))
    configured["speculative_algorithm"] = normalized_algorithm

    draft_path = configured.get("speculative_draft_model_path")
    if not isinstance(draft_path, str) or not draft_path.strip():
        raise ValueError(
            "speculative_draft_model_path is required for "
            f"{normalized_algorithm} speculative decoding"
        )

    topk = configured.setdefault("speculative_eagle_topk", 1)
    if topk != 1:
        raise ValueError(
            f"speculative_eagle_topk must be 1 for the synchronous lane; got {topk!r}"
        )
    if normalized_algorithm == "DFLASH":
        block_size = int(configured.setdefault("speculative_num_draft_tokens", 8))
        if block_size < 1:
            raise ValueError(
                "DFLASH speculative_num_draft_tokens is the draft block size and "
                f"must be >= 1; got {block_size}"
            )
        configured["speculative_num_draft_tokens"] = block_size
        configured.pop("speculative_num_steps", None)
    else:
        num_steps = int(configured.setdefault("speculative_num_steps", 3))
        num_draft_tokens = int(configured.setdefault("speculative_num_draft_tokens", 4))
        if num_steps < 0:
            raise ValueError(f"speculative_num_steps must be >= 0, got {num_steps}")
        if num_draft_tokens != num_steps + 1:
            raise ValueError(
                "topk=1 requires speculative_num_draft_tokens == "
                f"speculative_num_steps + 1; got {num_draft_tokens} and {num_steps}"
            )
        configured["speculative_num_steps"] = num_steps
        configured["speculative_num_draft_tokens"] = num_draft_tokens

    if speculative_overlap:
        if normalized_algorithm != "DFLASH" or int(configured.get("tp_size", 1)) != 1:
            raise ValueError("speculative_overlap currently requires DFLASH and TP1")
        if configured.get("disable_overlap_schedule") is True:
            raise ValueError(
                "speculative_overlap conflicts with disable_overlap_schedule=True"
            )
    # Worker allocation and scheduler result processing must use one contract.
    configured["disable_overlap_schedule"] = not speculative_overlap
    configured["disable_cuda_graph"] = not speculative_cuda_graph
    # SGLang 0.5.18 gives nested graph settings precedence over flat flags.
    # The speculative path uses target verify graphs; the stage's prefill graph
    # policy is kept only when a decoder-only builder opts in explicitly.
    if "cuda_graph_config" in configured:
        from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig

        graph_config = configured["cuda_graph_config"]
        graph_config = (
            graph_config.to_dict()
            if isinstance(graph_config, CudaGraphConfig)
            else dict(graph_config)
        )
        if not speculative_prefill_cuda_graph:
            graph_config["prefill"] = {"backend": "disabled"}
        graph_config["decode"] = {
            **graph_config.get("decode", {}),
            "backend": "full" if speculative_cuda_graph else "disabled",
        }
        configured["cuda_graph_config"] = graph_config
    if "cuda_graph_backend_decode" in configured:
        configured["cuda_graph_backend_decode"] = (
            "full" if speculative_cuda_graph else "disabled"
        )
    if (
        "cuda_graph_backend_prefill" in configured
        and not speculative_prefill_cuda_graph
    ):
        configured["cuda_graph_backend_prefill"] = "disabled"
        configured.pop("cuda_graph_bs_prefill", None)
        configured.pop("cuda_graph_max_bs_prefill", None)
    return configured


class SGLangGenerationEngineBuilder(ABC):
    """Build the model-neutral parts of a SGLang AR engine stage.

    Model-specific builders provide checkpoint preprocessing, model setup,
    request/result adapters, validation policy, and any stage-owned resources.
    Family-specific builders such as :class:`AsrEngineBuilder` and
    :class:`TtsEngineBuilder` define the lifecycle policy for each modality.
    """

    model_name: str
    context_length: int
    model_arch_override: str | None = None
    supports_context_length_override: ClassVar[bool] = False
    # Set True only by builders whose model has adopted the breakable prefill
    # CUDA graph contract; a deployment override cannot enable it otherwise.
    supports_breakable_prefill_cuda_graph: bool = False
    enable_speculative: bool = False
    speculative_algorithm: str = "STANDALONE"
    speculative_draft_model_path: str | None = None
    speculative_num_steps: int = 3
    speculative_num_draft_tokens: int = 4
    speculative_cuda_graph: bool = False
    speculative_draft_cuda_graph: bool = False
    speculative_prefill_cuda_graph: bool = False

    def build(
        self,
        model_path: str,
        *,
        device: str | None = None,
        gpu_id: int | None = None,
        dtype: str = "bfloat16",
        server_args_overrides: dict[str, Any] | None = None,
    ) -> Any:
        import torch

        from sglang_omni.platforms import current_platform
        from sglang_omni.scheduling import bootstrap as scheduling_bootstrap
        from sglang_omni.scheduling import sglang_backend
        from sglang_omni.utils.device import place_device_spec, resolve_device_spec

        checkpoint_dir = self.resolve_checkpoint(model_path)
        device = (
            resolve_device_spec(None, gpu_id)
            if device is None
            else place_device_spec(device, gpu_id)
        )
        gpu_id = torch.device(device).index or 0
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.gpu_id = gpu_id
        self.dtype = dtype

        self.pre_infra_setup(checkpoint_dir)

        if current_platform.is_cpu():
            # A stage default asking for a graph would otherwise fail inside
            # capture rather than at configuration time.
            server_args_overrides = dict(server_args_overrides or {})
            server_args_overrides["disable_cuda_graph"] = True

        requested_context_length = (
            server_args_overrides.get("context_length")
            if server_args_overrides is not None
            else None
        )
        if (
            requested_context_length is not None
            and self.supports_context_length_override
        ):
            context_length = requested_context_length
        else:
            context_length = self.resolve_context_length(
                checkpoint_dir,
                server_args_overrides=server_args_overrides,
            )
        self.context_length = _normalize_context_length(
            context_length,
            model_name=self.model_name,
        )

        operator_selected_prefill_backend = _operator_selected_prefill_graph_backend(
            server_args_overrides
        )
        overrides = build_generation_batch_overrides(
            server_args_overrides=server_args_overrides,
            **self.generation_defaults(dtype=dtype),
        )
        overrides = configure_speculative_overrides(
            overrides,
            enable_speculative=self.enable_speculative,
            speculative_algorithm=self.speculative_algorithm,
            speculative_draft_model_path=self.speculative_draft_model_path,
            speculative_num_steps=self.speculative_num_steps,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
            speculative_cuda_graph=self.speculative_cuda_graph,
            speculative_draft_cuda_graph=self.speculative_draft_cuda_graph,
            speculative_prefill_cuda_graph=self.speculative_prefill_cuda_graph,
        )
        self.adjust_overrides(overrides)
        if "context_length" in overrides:
            if not self.supports_context_length_override:
                raise ValueError(
                    f"{self.model_name} does not support a context_length override"
                )
            overrides.pop("context_length")
        # Note (Jiaxin Deng): user fractions were rejected upstream; what remains
        # is a builder KV-tuned default, dropped so headroom derives cleanly.
        from sglang_omni.scheduling.stage_kv_budget import peek_stage_kv_cache_bytes

        if peek_stage_kv_cache_bytes() is not None:
            builder_default_fraction = overrides.pop("mem_fraction_static", None)
            if builder_default_fraction is not None:
                logger.info(
                    f"{self.model_name}: clearing builder default "
                    f"mem_fraction_static={builder_default_fraction} because the "
                    "stage declares engine.kv_cache_bytes"
                )
        # Left unset, SGLang re-detects off a CUDA-first ladder that can contradict
        # placement. It owns the type, not the index.
        resolved_type = torch.device(device).type
        requested_type = overrides.get("device")
        if requested_type is not None and requested_type != resolved_type:
            raise ValueError(
                f"server_args_overrides set device={requested_type!r}, but this stage "
                f"resolved to {device!r}. Omni owns placement, so drop the override or "
                f"set device={resolved_type!r}."
            )
        overrides["device"] = resolved_type

        server_args = sglang_backend.build_sglang_server_args(
            checkpoint_dir,
            context_length=self.context_length,
            **overrides,
        )
        self.customize_server_args(server_args)
        if (
            overrides.get("chunked_prefill_size") is None
            and get_prefill_cuda_graph_backend(server_args) != CudaGraphBackend.DISABLED
        ):
            logger.info(
                f"{self.model_name}: chunked_prefill_size was unset, SGLang resolved "
                f"{server_args.chunked_prefill_size}, prefill CUDA graph cap "
                f"{server_args.cuda_graph_config.prefill.max_bs}"
            )
        self.validate_before_infrastructure(server_args)

        infra_kwargs = dict(self.infra_kwargs())
        if self.model_arch_override is not None:
            infra_kwargs.setdefault("model_arch_override", self.model_arch_override)
        prefill_graph_backend = get_prefill_cuda_graph_backend(server_args)
        if (
            prefill_graph_backend != CudaGraphBackend.DISABLED
            and not operator_selected_prefill_backend
        ):
            # SGLang treats every non-default source as operator-locked, and a
            # locked prefill backend skips upstream's model compatibility
            # resolution; a model-qualified stage default must stay eligible
            # for it.
            server_args._cuda_graph_config_locked.discard(("prefill", "backend"))
        if prefill_graph_backend == CudaGraphBackend.BREAKABLE:
            if not self.supports_breakable_prefill_cuda_graph:
                raise RuntimeError(
                    f"{self.model_name} has not adopted the breakable prefill "
                    "CUDA graph contract "
                    "(supports_breakable_prefill_cuda_graph=False); refusing "
                    "cuda_graph_backend_prefill='breakable'"
                )
            infra_kwargs.setdefault("enable_prefill_input_embeds", True)
        want_cuda_graph, infrastructure = (
            scheduling_bootstrap.create_sglang_infrastructure_defer_cuda_graph(
                server_args,
                gpu_id,
                **infra_kwargs,
            )
        )
        (
            model_worker,
            tree_cache,
            req_to_token_pool,
            token_to_kv_pool_allocator,
            model_config,
            *spec_workers,
        ) = infrastructure
        if spec_workers:
            target_worker_adapter, draft_worker = spec_workers
        else:
            target_worker_adapter = None
            draft_worker = None
        model = model_worker.model_runner.model

        self.setup_model(
            model_worker=model_worker,
            checkpoint_dir=checkpoint_dir,
            device=device,
            gpu_id=gpu_id,
            server_args=server_args,
        )
        if draft_worker is not None:
            self.setup_speculative_models(
                model, scheduling_bootstrap.speculative_draft_model(draft_worker)
            )

        self.validate_after_model_setup(model, server_args)

        self.compile_model(model, server_args)

        if want_cuda_graph:
            scheduling_bootstrap.init_sglang_cuda_graphs(model_worker)
            if draft_worker is not None:
                scheduling_bootstrap.init_speculative_draft_cuda_graphs(
                    draft_worker,
                    capture_draft_decode_graph=self.speculative_draft_cuda_graph,
                )
            self.post_cuda_graph_setup(model, server_args)
            if prefill_graph_backend != CudaGraphBackend.DISABLED:
                from sglang_omni.utils import cuda_graph_batch_validator

                cuda_graph_batch_validator.attest_prefill_cuda_graphs(
                    model_worker.model_runner, server_args
                )

        try:
            # Model-local encoder graphs and caches must be initialized after
            # SGLang's generation graphs to preserve the established order.
            self.setup_model_resources(
                model,
                server_args,
                generation_cuda_graph_enabled=want_cuda_graph,
            )

            output_proc = sglang_backend.SGLangOutputProcessor(
                capture_hidden=False,
                capture_hidden_layers=None,
                model=model,
            )
            self.setup_runtime_resources(model, server_args)
            speculative_kwargs = (
                dict(
                    target_worker_adapter=target_worker_adapter,
                    draft_worker=draft_worker,
                )
                if draft_worker is not None
                else {}
            )
            scheduler, model_runner = self._build_runtime(
                model_worker=model_worker,
                model=model,
                output_proc=output_proc,
                tree_cache=tree_cache,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
                server_args=server_args,
                model_config=model_config,
                **speculative_kwargs,
            )
            self.post_scheduler_setup(scheduler, model_runner)
            return scheduler
        except Exception:
            self.cleanup_build_failure()
            raise

    def resolve_checkpoint(self, model_path: str) -> str:
        # The shared builder treats checkpoint resolution as a family policy.
        # Subclasses override this when they need a resolved local snapshot.
        return model_path

    @abstractmethod
    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        del checkpoint_dir

    def resolve_context_length(
        self,
        checkpoint_dir: str,
        *,
        server_args_overrides: Mapping[str, Any] | None = None,
    ) -> int:
        del checkpoint_dir, server_args_overrides
        return self.context_length

    def validate_before_infrastructure(self, server_args: Any) -> None:
        del server_args

    def validate_after_model_setup(self, model: Any, server_args: Any) -> None:
        del model, server_args

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        del overrides

    def customize_server_args(self, server_args: Any) -> None:
        del server_args

    def infra_kwargs(self) -> dict[str, Any]:
        return {}

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del model_worker, checkpoint_dir, device, gpu_id, server_args

    def get_model_buffer_bs(self, model: Any) -> int | None:
        del model
        return None

    def setup_speculative_models(self, model: Any, draft_model: Any) -> None:
        """Link the target and speculative draft models after both are loaded."""
        del model, draft_model

    def compile_model(self, model: Any, server_args: Any) -> None:
        del model, server_args

    def post_cuda_graph_setup(self, model: Any, server_args: Any) -> None:
        del model, server_args

    def setup_model_resources(
        self,
        model: Any,
        server_args: Any,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        del model, server_args, generation_cuda_graph_enabled

    def setup_runtime_resources(self, model: Any, server_args: Any) -> None:
        del model, server_args

    @abstractmethod
    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        raise NotImplementedError

    def _build_runtime(
        self,
        *,
        model_worker: Any,
        model: Any,
        output_proc: Any,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        server_args: Any,
        model_config: Any,
        target_worker_adapter: Any = None,
        draft_worker: Any = None,
    ) -> tuple[Any, Any]:
        request_builder, result_adapter = self.make_adapters(model)
        scheduler_kwargs = self.extra_scheduler_kwargs()
        model_runner = self.make_model_runner(model_worker, output_proc)
        scheduler = self._make_scheduler(
            model_worker=model_worker,
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            server_args=server_args,
            model_config=model_config,
            model_runner=model_runner,
            request_builder=request_builder,
            result_adapter=result_adapter,
            extra_scheduler_kwargs=scheduler_kwargs,
            target_worker_adapter=target_worker_adapter,
            draft_worker=draft_worker,
        )
        return scheduler, model_runner

    def make_abort_callback(self) -> Any | None:
        return None

    def make_request_finished_callback(self) -> Any | None:
        return None

    def extra_scheduler_callbacks(self) -> dict[str, Any]:
        return {}

    def cleanup_build_failure(self) -> None:
        pass

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {}

    def _make_scheduler(
        self,
        *,
        model_worker: Any,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        server_args: Any,
        model_config: Any,
        model_runner: Any,
        request_builder: Any,
        result_adapter: Any,
        extra_scheduler_kwargs: dict[str, Any],
        target_worker_adapter: Any = None,
        draft_worker: Any = None,
    ) -> Any:
        from sglang_omni.scheduling import omni_scheduler

        scheduler_kwargs = {
            "tp_worker": model_worker,
            "tree_cache": tree_cache,
            "req_to_token_pool": req_to_token_pool,
            "token_to_kv_pool_allocator": token_to_kv_pool_allocator,
            "server_args": server_args,
            "model_config": model_config,
            "model_runner": model_runner,
            "request_builder": request_builder,
            "result_adapter": result_adapter,
            "abort_callback": self.make_abort_callback(),
            "request_finished_callback": self.make_request_finished_callback(),
            "target_worker_adapter": target_worker_adapter,
            "draft_worker": draft_worker,
        }
        scheduler_kwargs.update(self.extra_scheduler_callbacks())
        scheduler_kwargs.update(extra_scheduler_kwargs)
        return omni_scheduler.OmniScheduler(**scheduler_kwargs)

    def post_scheduler_setup(self, scheduler: Any, model_runner: Any) -> None:
        del scheduler, model_runner


class AsrEngineBuilder(SGLangGenerationEngineBuilder):
    """Shared lifecycle policy for SGLang-backed ASR stages."""

    def resolve_checkpoint(self, model_path: str) -> str:
        # ASR model loaders accept either a repo id or a local path and should
        # preserve the operator-provided value through server-args creation.
        return model_path

    def validate_before_infrastructure(self, server_args: Any) -> None:
        validate_generation_batch_policy(
            model_name=self.model_name,
            server_args=server_args,
        )

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        from sglang_omni.model_runner.base import ModelRunner

        return ModelRunner(model_worker, output_proc)


class TtsEngineBuilder(SGLangGenerationEngineBuilder):
    """Compatibility builder preserving the historical TTS contract."""

    @abstractmethod
    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        raise NotImplementedError

    def resolve_checkpoint(self, model_path: str) -> str:
        return _resolve_checkpoint(model_path)

    def validate_before_infrastructure(self, server_args: Any) -> None:
        del server_args

    def validate_after_model_setup(self, model: Any, server_args: Any) -> None:
        validate_generation_batch_policy(
            model_name=self.model_name,
            server_args=server_args,
            model_buffer_bs=self.get_model_buffer_bs(model),
        )

    def make_scheduler(
        self,
        *,
        model_worker: Any,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        server_args: Any,
        model_config: Any,
        model_runner: Any,
        request_builder: Any,
        result_adapter: Any,
    ) -> Any:
        from sglang_omni.scheduling import omni_scheduler

        return omni_scheduler.OmniScheduler(
            tp_worker=model_worker,
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            server_args=server_args,
            model_config=model_config,
            model_runner=model_runner,
            request_builder=request_builder,
            result_adapter=result_adapter,
            abort_callback=self.make_abort_callback(),
            request_finished_callback=self.make_request_finished_callback(),
            **self.extra_scheduler_kwargs(),
        )

    def _build_runtime(
        self,
        *,
        model_worker: Any,
        model: Any,
        output_proc: Any,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        server_args: Any,
        model_config: Any,
    ) -> tuple[Any, Any]:
        model_runner = self.make_model_runner(model_worker, output_proc)
        request_builder, result_adapter = self.make_adapters(model)
        scheduler = self.make_scheduler(
            model_worker=model_worker,
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            server_args=server_args,
            model_config=model_config,
            model_runner=model_runner,
            request_builder=request_builder,
            result_adapter=result_adapter,
        )
        return scheduler, model_runner
