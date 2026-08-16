# SPDX-License-Identifier: Apache-2.0
"""Builders for SGLang-backed autoregressive engine stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from sglang_omni.scheduling.generation_batch_policy import (
    CudaGraphBackend,
    build_generation_batch_overrides,
    get_prefill_cuda_graph_backend,
    nested_prefill_overrides,
    validate_generation_batch_policy,
)
from sglang_omni.utils.checkpoint import resolve_checkpoint as _resolve_checkpoint


def _operator_selected_prefill_graph_backend(
    server_args_overrides: Mapping[str, Any] | None,
) -> bool:
    if not server_args_overrides:
        return False
    if "cuda_graph_backend_prefill" in server_args_overrides:
        return True
    return "backend" in nested_prefill_overrides(server_args_overrides)


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
) -> dict[str, Any]:
    """Validate Omni's STANDALONE speculative decoding configuration."""
    configured = dict(overrides)
    if speculative_draft_cuda_graph and not speculative_cuda_graph:
        raise ValueError(
            "speculative_draft_cuda_graph=True requires speculative_cuda_graph=True"
        )
    if enable_speculative:
        requested = {
            "speculative_algorithm": "STANDALONE",
            "speculative_draft_model_path": speculative_draft_model_path,
            "speculative_num_steps": speculative_num_steps,
            "speculative_eagle_topk": 1,
            "speculative_num_draft_tokens": speculative_num_draft_tokens,
        }
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
                "speculative_cuda_graph=True requires STANDALONE speculative decoding"
            )
        return configured

    normalized_algorithm = str(algorithm).upper()
    if normalized_algorithm != "STANDALONE":
        raise ValueError(
            "sglang-omni only supports STANDALONE speculative decoding in the "
            f"synchronous lane; got speculative_algorithm={algorithm!r}"
        )
    configured["speculative_algorithm"] = normalized_algorithm

    draft_path = configured.get("speculative_draft_model_path")
    if not isinstance(draft_path, str) or not draft_path.strip():
        raise ValueError(
            "speculative_draft_model_path is required for STANDALONE "
            "speculative decoding"
        )

    topk = configured.setdefault("speculative_eagle_topk", 1)
    if topk != 1:
        raise ValueError(
            "speculative_eagle_topk must be 1 for the STANDALONE lane; " f"got {topk!r}"
        )
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

    # note (Junnan Li): target and draft graph capture follow multi-token parity.
    configured["disable_overlap_schedule"] = True
    configured["disable_cuda_graph"] = not speculative_cuda_graph
    if "cuda_graph_backend_prefill" in configured:
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
    # Set True only by builders whose model has adopted the breakable prefill
    # CUDA graph contract; a deployment override cannot enable it otherwise.
    supports_breakable_prefill_cuda_graph: bool = False
    enable_speculative: bool = False
    speculative_draft_model_path: str | None = None
    speculative_num_steps: int = 3
    speculative_num_draft_tokens: int = 4
    speculative_cuda_graph: bool = False
    speculative_draft_cuda_graph: bool = False

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
            speculative_draft_model_path=self.speculative_draft_model_path,
            speculative_num_steps=self.speculative_num_steps,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
            speculative_cuda_graph=self.speculative_cuda_graph,
            speculative_draft_cuda_graph=self.speculative_draft_cuda_graph,
        )
        self.adjust_overrides(overrides)
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
        self.validate_before_infrastructure(server_args)

        infra_kwargs = dict(self.infra_kwargs())
        if self.model_arch_override is not None:
            infra_kwargs.setdefault("model_arch_override", self.model_arch_override)
        prefill_graph_backend = get_prefill_cuda_graph_backend(server_args)
        if (
            prefill_graph_backend != CudaGraphBackend.DISABLED
            and not operator_selected_prefill_backend
        ):
            # SGLang treats every non-default source as operator-locked. A
            # model-qualified stage default should survive compatibility
            # resolution, but must remain eligible for the late free-memory
            # safety gate immediately before graph capture.
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
            prefill_mgr,
            decode_mgr,
            model_config,
            *spec_workers,
        ) = infrastructure
        if spec_workers:
            if len(spec_workers) != 2:
                raise RuntimeError(
                    "SGLang infrastructure returned an invalid speculative "
                    "worker payload"
                )
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
            scheduler, model_runner = self._build_runtime(
                model_worker=model_worker,
                model=model,
                output_proc=output_proc,
                tree_cache=tree_cache,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
                server_args=server_args,
                model_config=model_config,
                prefill_manager=prefill_mgr,
                decode_manager=decode_mgr,
                target_worker_adapter=target_worker_adapter,
                draft_worker=draft_worker,
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
        prefill_manager: Any,
        decode_manager: Any,
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
            prefill_manager=prefill_manager,
            decode_manager=decode_manager,
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
        prefill_manager: Any,
        decode_manager: Any,
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
            "prefill_manager": prefill_manager,
            "decode_manager": decode_manager,
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

    tokenizer: Any = None

    def resolve_checkpoint(self, model_path: str) -> str:
        # ASR model loaders accept either a repo id or a local path and should
        # preserve the operator-provided value through server-args creation.
        return model_path

    def validate_before_infrastructure(self, server_args: Any) -> None:
        validate_generation_batch_policy(
            model_name=self.model_name,
            server_args=server_args,
        )

    def infra_kwargs(self) -> dict[str, Any]:
        return {"target_tokenizer": self.tokenizer}

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
        prefill_manager: Any,
        decode_manager: Any,
        model_runner: Any,
        request_builder: Any,
        result_adapter: Any,
        target_worker_adapter: Any = None,
        draft_worker: Any = None,
    ) -> Any:
        from sglang_omni.scheduling import omni_scheduler

        return omni_scheduler.OmniScheduler(
            tp_worker=model_worker,
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            server_args=server_args,
            model_config=model_config,
            prefill_manager=prefill_manager,
            decode_manager=decode_manager,
            model_runner=model_runner,
            request_builder=request_builder,
            result_adapter=result_adapter,
            abort_callback=self.make_abort_callback(),
            request_finished_callback=self.make_request_finished_callback(),
            target_worker_adapter=target_worker_adapter,
            draft_worker=draft_worker,
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
        prefill_manager: Any,
        decode_manager: Any,
        target_worker_adapter: Any = None,
        draft_worker: Any = None,
    ) -> tuple[Any, Any]:
        model_runner = self.make_model_runner(model_worker, output_proc)
        request_builder, result_adapter = self.make_adapters(model)
        scheduler_kwargs = dict(
            model_worker=model_worker,
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            server_args=server_args,
            model_config=model_config,
            prefill_manager=prefill_manager,
            decode_manager=decode_manager,
            model_runner=model_runner,
            request_builder=request_builder,
            result_adapter=result_adapter,
        )
        if draft_worker is not None:
            scheduler_kwargs.update(
                target_worker_adapter=target_worker_adapter,
                draft_worker=draft_worker,
            )
        scheduler = self.make_scheduler(**scheduler_kwargs)
        return scheduler, model_runner
