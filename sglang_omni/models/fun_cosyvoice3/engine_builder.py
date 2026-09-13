# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 SGLang engine builder."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
from typing import Any

import torch

from sglang_omni.models.fun_cosyvoice3 import request_builders
from sglang_omni.models.fun_cosyvoice3.utils import (
    CosyVoice3Tokenizer,
    SpeakerEncoder,
    SpeechTokenizerV3,
)
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
from sglang_omni.utils.checkpoint import resolve_checkpoint as _resolve_checkpoint

logger = logging.getLogger(__name__)

_SUPPORTED_SPECULATIVE_ALGORITHMS = ("EAGLE3", "DSPARK")


def _draft_cache_root(kind: str = "eagle3") -> str:
    return os.path.join(
        os.environ.get("SGLANG_OMNI_CACHE_DIR")
        or os.path.join(os.path.expanduser("~"), ".cache", "sglang_omni"),
        f"cosyvoice3_{kind}_drafts",
    )


def _materialize_draft_dir(resolved: str, normalized_config: dict, kind: str) -> str:
    """Write ``normalized_config`` next to symlinked checkpoint files in the cache."""
    normalized_config = dict(normalized_config)
    normalized_config["_sglang_omni_source"] = resolved
    digest = hashlib.sha256(
        json.dumps(normalized_config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    target_dir = os.path.join(_draft_cache_root(kind), digest)
    os.makedirs(target_dir, exist_ok=True)
    for entry in os.listdir(resolved):
        if entry == "config.json":
            continue
        link = os.path.join(target_dir, entry)
        source = os.path.realpath(os.path.join(resolved, entry))
        if os.path.lexists(link):
            if os.path.realpath(link) == source:
                continue
            os.remove(link)
        os.symlink(source, link)
    with open(os.path.join(target_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(normalized_config, handle, indent=2, sort_keys=True)
    return target_dir


def materialize_cosyvoice3_dspark_draft(draft_model_path: str) -> str:
    """Return a DSpark draft directory SGLang's loader can consume.

    The published checkpoint is in speculators format (no ``model_type``,
    nested ``transformer_layer_config``, unified 158720 vocab). The normalised
    copy flattens it, points at the Omni draft class and expresses the vocab /
    mask id in the draft-owned speech embedding space.
    """
    from sglang_omni.models.fun_cosyvoice3.sglang_model import (
        DSPARK_UNIFIED_SPEECH_OFFSET,
        TOTAL_VOCAB_SIZE,
        normalize_cosyvoice3_dspark_config,
    )

    resolved = _resolve_checkpoint(draft_model_path)
    with open(os.path.join(resolved, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    if str(config.get("speculators_model_type", "")).lower() != "dspark" and (
        "DSparkDraftModel" not in list(config.get("architectures") or [])
    ):
        raise ValueError(
            "Fun-CosyVoice3 DSPARK speculative decoding expects a DSpark draft "
            f"checkpoint; got architectures={config.get('architectures')!r}"
        )
    layer_config = config.get("transformer_layer_config") or {}
    vocab = int(layer_config.get("vocab_size") or config.get("vocab_size") or 0)
    mask_token_id = config.get("mask_token_id")
    if vocab < DSPARK_UNIFIED_SPEECH_OFFSET + TOTAL_VOCAB_SIZE + 1 or (
        mask_token_id != DSPARK_UNIFIED_SPEECH_OFFSET + TOTAL_VOCAB_SIZE
    ):
        raise ValueError(
            "Fun-CosyVoice3 DSPARK draft must use the unified vocab with the "
            f"speech range at offset {DSPARK_UNIFIED_SPEECH_OFFSET} and the mask "
            f"token right after it; got vocab_size={vocab} "
            f"mask_token_id={mask_token_id!r}"
        )
    if int(config.get("markov_rank", 0) or 0) <= 0:
        raise ValueError("Fun-CosyVoice3 DSPARK draft requires markov_rank > 0")
    normalized = normalize_cosyvoice3_dspark_config(config)
    from sglang.srt.speculative.dspark_components.dspark_config import (
        parse_dspark_draft_config,
    )

    if (
        parse_dspark_draft_config(draft_hf_config=normalized).gamma
        != int(normalized["block_size"]) - 1
    ):
        raise RuntimeError(
            "CosyVoice3 DSpark requires SGLang bonus-anchor layout support; "
            "update the native DSpark runtime before enabling this checkpoint."
        )
    target_dir = _materialize_draft_dir(resolved, normalized, "dspark")
    logger.info(
        "Materialized normalized CosyVoice3 DSpark draft %s -> %s (input_block_size=%s)",
        draft_model_path,
        target_dir,
        normalized.get("block_size"),
    )
    return target_dir


def materialize_cosyvoice3_eagle3_draft(draft_model_path: str) -> str:
    """Return a draft directory whose config.json the stock EAGLE3 loader can use.

    The AngelSlim checkpoint publishes ``tie_word_embeddings=True`` and
    ``num_hidden_layers=24`` for a one-layer draft with its own head. SGLang
    sizes the draft KV pool and builds the draft from those fields, so a
    normalised copy (weights symlinked, config rewritten) is materialised in
    the Omni cache and used as ``speculative_draft_model_path``.
    """
    from sglang_omni.models.fun_cosyvoice3.sglang_model import (
        EAGLE3_DRAFT_ARCHITECTURE,
        TOTAL_VOCAB_SIZE,
        normalize_cosyvoice3_eagle3_config,
    )

    resolved = _resolve_checkpoint(draft_model_path)
    config_path = os.path.join(resolved, "config.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    architectures = list(config.get("architectures") or [])
    if EAGLE3_DRAFT_ARCHITECTURE not in architectures:
        raise ValueError(
            "Fun-CosyVoice3 speculative decoding expects an EAGLE3 draft with "
            f"architectures={[EAGLE3_DRAFT_ARCHITECTURE]!r}; got {architectures!r}"
        )
    hidden = int(config.get("hidden_size", 0))
    vocab = int(config.get("vocab_size", 0))
    draft_vocab = int(config.get("draft_vocab_size") or vocab)
    if vocab != TOTAL_VOCAB_SIZE or draft_vocab != TOTAL_VOCAB_SIZE:
        raise ValueError(
            "Fun-CosyVoice3 EAGLE3 draft vocab must equal the speech vocab "
            f"{TOTAL_VOCAB_SIZE}; got vocab_size={vocab} draft_vocab_size={draft_vocab}"
        )

    class _Namespace:
        pass

    normalized = _Namespace()
    normalized.__dict__.update(config)
    normalize_cosyvoice3_eagle3_config(normalized)
    normalized_config = dict(normalized.__dict__)
    target_dir = _materialize_draft_dir(resolved, normalized_config, "eagle3")
    logger.info(
        "Materialized normalized CosyVoice3 EAGLE3 draft %s -> %s (hidden=%d)",
        draft_model_path,
        target_dir,
        hidden,
    )
    return target_dir


def _validate_cosyvoice3_draft_config(
    target_config: Any,
    draft_dir: str,
    fields: tuple[str, ...] = (
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
    ),
) -> None:
    with open(os.path.join(draft_dir, "config.json"), encoding="utf-8") as handle:
        draft_config = json.load(handle)
    for field in fields:
        target_value = getattr(target_config, field, None)
        draft_value = draft_config.get(field)
        if target_value is not None and draft_value != target_value:
            raise ValueError(
                "Fun-CosyVoice3 target and draft configs must match for "
                f"speculative decoding: {field} target={target_value!r} "
                f"draft={draft_value!r}"
            )


class FunCosyVoice3EngineBuilder(TtsEngineBuilder):
    model_name = "Fun-CosyVoice3"
    context_length = 4096
    model_arch_override = "FunCosyVoice3SGLangModel"

    def __init__(
        self,
        *,
        enable_speculative: bool = False,
        speculative_algorithm: str = "EAGLE3",
        speculative_draft_model_path: str | None = None,
        speculative_num_steps: int = 3,
        speculative_num_draft_tokens: int | None = None,
        speculative_cuda_graph: bool = False,
        speculative_draft_cuda_graph: bool = False,
        speculative_use_rejection_sampling: bool = True,
    ) -> None:
        super().__init__()
        self._checkpoint_root: str | None = None
        self.enable_speculative = bool(enable_speculative)
        self.speculative_algorithm = str(speculative_algorithm).upper()
        if self.speculative_algorithm not in _SUPPORTED_SPECULATIVE_ALGORITHMS:
            raise ValueError(
                "Fun-CosyVoice3 speculative decoding supports "
                f"{list(_SUPPORTED_SPECULATIVE_ALGORITHMS)}; "
                f"got {speculative_algorithm!r}"
            )
        self.speculative_draft_model_path = speculative_draft_model_path
        if self.speculative_algorithm == "DSPARK":
            # DSpark drafts one block per step; the verify window is gamma + 1
            # and defaults to the checkpoint's block size.
            self.speculative_num_steps = 1
            self.speculative_num_draft_tokens = (
                None
                if speculative_num_draft_tokens is None
                else int(speculative_num_draft_tokens)
            )
        else:
            self.speculative_num_steps = int(speculative_num_steps)
            self.speculative_num_draft_tokens = int(
                4
                if speculative_num_draft_tokens is None
                else speculative_num_draft_tokens
            )
        self.speculative_cuda_graph = bool(speculative_cuda_graph)
        self.speculative_draft_cuda_graph = bool(speculative_draft_cuda_graph)
        self.speculative_use_rejection_sampling = bool(
            speculative_use_rejection_sampling
        )
        self._target_hf_config: Any = None

    def _blanken_dir(self) -> str:
        assert self._checkpoint_root is not None, "checkpoint_root not set"
        return os.path.join(self._checkpoint_root, "CosyVoice-BlankEN")

    def resolve_checkpoint(self, model_path: str) -> str:
        resolved = _resolve_checkpoint(model_path)
        self._checkpoint_root = resolved
        # SGLang needs CosyVoice-BlankEN/ which has config.json (model_type: qwen2)
        return self._blanken_dir()

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        from transformers import AutoConfig

        self._target_hf_config = AutoConfig.from_pretrained(checkpoint_dir)

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        algorithm = overrides.get("speculative_algorithm")
        if algorithm is None:
            return
        self.enable_speculative = True
        if str(algorithm).upper() == "DSPARK":
            self._adjust_dspark_overrides(overrides)
        else:
            self.speculative_num_draft_tokens = int(
                overrides["speculative_num_draft_tokens"]
            )
            draft_dir = materialize_cosyvoice3_eagle3_draft(
                overrides["speculative_draft_model_path"]
            )
            if self._target_hf_config is not None:
                _validate_cosyvoice3_draft_config(self._target_hf_config, draft_dir)
            overrides["speculative_draft_model_path"] = draft_dir
            overrides["speculative_use_rejection_sampling"] = (
                self.speculative_use_rejection_sampling
            )
        # note (Junnan Li): the sync speculative lane is eager on prefill and
        # forbids Omni's async decode lookahead; verify graphs follow
        # speculative_cuda_graph via configure_speculative_overrides.
        overrides["disable_overlap_schedule"] = True

    def _adjust_dspark_overrides(self, overrides: dict[str, Any]) -> None:
        draft_dir = materialize_cosyvoice3_dspark_draft(
            overrides["speculative_draft_model_path"]
        )
        if self._target_hf_config is not None:
            _validate_cosyvoice3_draft_config(
                self._target_hf_config, draft_dir, fields=("hidden_size",)
            )
        with open(os.path.join(draft_dir, "config.json"), encoding="utf-8") as handle:
            draft_config = json.load(handle)
        # gamma = verify window - 1; SGLang cross-checks it against the
        # checkpoint proposal count (the bonus anchor is not a proposal).
        gamma = overrides.get("speculative_dspark_block_size")
        num_draft_tokens = overrides.get("speculative_num_draft_tokens")
        if gamma is None:
            gamma = (
                int(num_draft_tokens) - 1
                if num_draft_tokens is not None
                else int(draft_config["block_size"])
                - int(draft_config.get("dspark_bonus_anchor", False))
            )
        gamma = int(gamma)
        if gamma < 1:
            raise ValueError(
                f"Fun-CosyVoice3 DSPARK block size must be >= 1; got {gamma}"
            )
        if num_draft_tokens is not None and int(num_draft_tokens) != gamma + 1:
            raise ValueError(
                "Fun-CosyVoice3 DSPARK speculative_num_draft_tokens must equal "
                f"block size + 1; got {num_draft_tokens} for block size {gamma}"
            )
        overrides["speculative_dspark_block_size"] = gamma
        overrides["speculative_num_draft_tokens"] = gamma + 1
        overrides["speculative_num_steps"] = 1
        overrides["speculative_draft_model_path"] = draft_dir
        self.speculative_num_draft_tokens = gamma + 1

    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> dict[str, Any]:
        return {
            "max_running_requests": 32,
            "cuda_graph_max_bs": 32,
            "torch_compile_max_bs": 32,
            "dtype": dtype,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "enable_torch_compile": False,
            "mem_fraction_static": 0.85,
            "max_prefill_tokens": 4096,
            "sampling_backend": "pytorch",
            "trust_remote_code": True,
        }

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del checkpoint_dir, gpu_id, server_args
        model = model_worker.model_runner.model
        root = self._checkpoint_root
        assert root is not None, "checkpoint_root not set"
        from sglang_omni.models.fun_cosyvoice3.sglang_model import TOTAL_VOCAB_SIZE

        model_worker.model_runner.model_config.vocab_size = TOTAL_VOCAB_SIZE

        llm_pt_path = os.path.join(root, "llm.pt")
        logger.info("Loading CosyVoice3 fine-tuned weights from %s", llm_pt_path)
        state_dict = torch.load(llm_pt_path, map_location="cpu", weights_only=True)
        model.load_weights(list(state_dict.items()))
        logger.info("CosyVoice3 weights loaded")

        tokenizer_path = os.path.join(root, "CosyVoice-BlankEN")
        speech_tokenizer_path = os.path.join(root, "speech_tokenizer_v3.onnx")
        campplus_path = os.path.join(root, "campplus.onnx")

        tokenizer = CosyVoice3Tokenizer(tokenizer_path)
        speech_tokenizer = SpeechTokenizerV3(speech_tokenizer_path, device=device)
        speaker_encoder = SpeakerEncoder(campplus_path, device=device)

        request_builders.set_cosyvoice3_preprocessing_context(
            model=model,
            tokenizer=tokenizer,
            speech_tokenizer=speech_tokenizer,
            speaker_encoder=speaker_encoder,
            model_revision=root,
        )

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        model_runner_mod = importlib.import_module(
            "sglang_omni.models.fun_cosyvoice3.model_runner"
        )
        return model_runner_mod.FunCosyVoice3ModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        return request_builders.make_cosyvoice3_scheduler_adapters(
            model=model, enable_speculative=self.enable_speculative
        )

    def make_abort_callback(self) -> Any | None:
        return request_builders.cleanup_prepared_cosyvoice3_request
