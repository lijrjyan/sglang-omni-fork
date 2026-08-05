# SPDX-License-Identifier: Apache-2.0
"""StagePayload <-> SGLang request adapters for Qwen3-ASR.

Unlike Whisper (encoder-decoder, features consumed inside the model forward),
Qwen3-ASR is a Qwen3 causal LM that ingests audio as multimodal embeddings:
the prompt contains an ``<|audio_pad|>`` placeholder repeated once per audio
token, and the model's ``general_mm_embed_routine`` scatters the encoder output
into those positions. So request_builder must:
  * extract mel features (WhisperFeatureExtractor) + attention mask,
  * compute how many audio tokens the encoder will emit,
  * build the chat prompt with that many ``<|audio_pad|>`` tokens,
  * hand the features over as a ``MultimodalDataItem``.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable

import torch
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.preprocessing.transcription import prepare_audio
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

from .audio_lengths import qwen3_asr_num_audio_tokens

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000

# Qwen's native wrapper keeps each clip in one model forward up to 1200 seconds.
# Longer inputs are split by that wrapper, but SGLang-Omni deliberately does not
# add a second transcript stitching policy here.
QWEN3_ASR_MAX_AUDIO_SECONDS = 1200.0
QWEN3_ASR_UPSTREAM_MAX_NEW_TOKENS = 4096
_OUTPUT_TOKENS_PER_AUDIO_SECOND = 10

_AUDIO_START = "<|audio_start|>"
_AUDIO_PAD = "<|audio_pad|>"
_AUDIO_END = "<|audio_end|>"
_ASR_TEXT = "<asr_text>"


@dataclass
class Qwen3ASRRequestData(SGLangARRequestData):
    enforce_request_limits: bool = True
    prompt_token_ids: list[int] | None = None
    output_ids: list[int] | None = None
    audio_duration_s: float = 0.0
    language: str = "en"
    engine_start_s: float = 0.0


def _decode_token_ids(
    tokenizer: Any, token_ids: list[int], skip_special_tokens: bool
) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def _find_subsequence(values: list[int], pattern: list[int]) -> int | None:
    if not pattern:
        return None
    limit = len(values) - len(pattern) + 1
    for start in range(max(limit, 0)):
        if values[start : start + len(pattern)] == pattern:
            return start
    return None


def _encode_literal(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text, add_special_tokens=False)
    if hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids
    else:
        input_ids = encoded["input_ids"]
    return list(input_ids)


def build_qwen3_asr_prompt_ids(
    tokenizer: Any, num_audio_tokens: int, language: str
) -> list[int]:
    prompt = (
        f"<|im_start|>user\n"
        f"{_AUDIO_START}{_AUDIO_PAD * num_audio_tokens}{_AUDIO_END}"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    # Qwen3-ASR needs the same forced assistant prefix as upstream qwen_asr.
    prompt += f"language {language}{_ASR_TEXT}"
    return list(tokenizer(prompt, add_special_tokens=False).input_ids)


def qwen3_asr_prompt_overhead_tokens(tokenizer: Any) -> int:
    """Tokenized non-audio portion of the native ASR prompt."""
    return len(build_qwen3_asr_prompt_ids(tokenizer, 0, "English"))


def qwen3_asr_auto_output_budget(audio_duration_s: float) -> int:
    """Keep the upstream floor while avoiding a flat long-audio ceiling."""
    if not math.isfinite(audio_duration_s) or audio_duration_s < 0:
        raise ValueError("audio duration must be a finite non-negative number")
    return max(
        QWEN3_ASR_UPSTREAM_MAX_NEW_TOKENS,
        math.ceil(audio_duration_s * _OUTPUT_TOKENS_PER_AUDIO_SECOND),
    )


def _positive_max_new_tokens(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("max_new_tokens must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    return result


def qwen3_asr_request_output_budget(
    params: dict[str, Any],
    *,
    audio_duration_s: float,
    default_max_new_tokens: int | None,
) -> int:
    """Resolve request > operator > duration-aware automatic precedence."""
    explicit = params.get("max_new_tokens")
    if explicit is not None:
        return _positive_max_new_tokens(explicit)
    if default_max_new_tokens is not None:
        return _positive_max_new_tokens(default_max_new_tokens)
    return qwen3_asr_auto_output_budget(audio_duration_s)


def make_qwen3_asr_scheduler_adapters(
    *,
    tokenizer: Any,
    max_new_tokens: int | None,
    feature_extractor: Any = None,
) -> tuple[
    Callable[[StagePayload], Qwen3ASRRequestData], Callable[[Any], StagePayload]
]:
    if feature_extractor is None:
        raise ValueError("Qwen3-ASR processor is missing a feature_extractor")

    audio_pad_token_id = int(tokenizer.convert_tokens_to_ids(_AUDIO_PAD))
    eos_token_id = int(tokenizer.eos_token_id)
    vocab_size = int(tokenizer.vocab_size)
    asr_text_token_ids = _encode_literal(tokenizer, _ASR_TEXT)

    def request_builder(payload: StagePayload) -> Qwen3ASRRequestData:
        params = payload.request.params or {}
        prepared = prepare_audio(
            payload, source_name="Qwen3-ASR", target_sample_rate=_SAMPLE_RATE
        )
        audio = prepared.waveform
        audio_duration_s = prepared.duration_s
        fingerprint = prepared.fingerprint

        # note (Jeffro Qu): unlike Whisper's default 30s window, here we pad the mel to the clip's true length.
        # WhisperFeatureExtractor defaults to padding="max_length", padding every clip to nb_max_frames=3000 (~30s),
        # so a short clip pays the full 30s of mel FFT on silence.
        # This is safe for Qwen3-ASR because its encoder is variable-length and keeps only the
        # valid frames via feature_attention_mask; vanilla Whisper's fixed-length encoder would instead break on padding="longest" (see ref: transformers#26241).
        # refs:
        #  https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/feature_extraction_whisper.py
        #  https://github.com/huggingface/transformers/issues/26241
        extracted = feature_extractor(
            audio,
            sampling_rate=_SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
            padding="longest",
            # note (Junnan Li): Qwen3-ASR's encoder accepts mel sequences beyond
            # Whisper's 30-second window, so preprocessing must not truncate them.
            truncation=False,
        )
        features = extracted.input_features
        feature_attention_mask = getattr(extracted, "attention_mask", None)
        if feature_attention_mask is None:
            # WhisperFeatureExtractor normally returns one; fall back to all-valid.
            feature_attention_mask = torch.ones(
                (features.shape[0], features.shape[-1]), dtype=torch.long
            )
        # note (Jeffro Qu): get_audio_feature uses the mask to select valid
        # frames; its no-mask branch transposes wrong, so the mask path must be taken.
        num_mel_frames = int(feature_attention_mask.sum().item())
        num_audio_tokens = int(qwen3_asr_num_audio_tokens(num_mel_frames))
        logger.debug(
            f"[qwen3-asr] mel_frames={num_mel_frames} "
            f"num_audio_tokens={num_audio_tokens} feat_shape={tuple(features.shape)}"
        )

        lang_raw = str(params.get("language") or "en").strip().lower()
        forced_language = {"zh": "Chinese", "cn": "Chinese"}.get(
            lang_raw, "Chinese" if lang_raw.startswith("zh") else "English"
        )
        input_ids = build_qwen3_asr_prompt_ids(
            tokenizer, num_audio_tokens, forced_language
        )

        audio_item = MultimodalDataItem(
            modality=Modality.AUDIO,
            hash=prepared.fingerprint_int,
            feature=features,
            model_specific_data={
                "feature_attention_mask": feature_attention_mask,
            },
        )
        # general_mm_embed_routine locates audio positions by matching each
        # item's pad_value against input_ids. The omni scheduler does not run
        # pad_input_ids for us, so compute the pad_value, replace the
        # <|audio_pad|> placeholders with it, and record the placeholder span as
        # item.offsets. SGLang treats offsets as inclusive.
        audio_item.set_pad_value()
        audio_start = input_ids.index(audio_pad_token_id)
        input_ids = [
            audio_item.pad_value if tok == audio_pad_token_id else tok
            for tok in input_ids
        ]
        audio_item.offsets = [(audio_start, audio_start + num_audio_tokens - 1)]

        mm_inputs = MultimodalInputs(
            mm_items=[audio_item],
            num_image_tokens=num_audio_tokens,
        )
        mm_inputs.audio_token_id = audio_pad_token_id
        # sglang indexes mm_input.mrope_positions[:, start:end] during prefill and
        # does not compute a default, so we must supply it. Qwen3-ASR's MRoPE is
        # degenerate for ASR (all 3 sections share the text position), i.e. plain
        # 1-D positions broadcast to shape [3, seq_len].
        seq_len = len(input_ids)
        positions = torch.arange(seq_len, dtype=torch.long)
        mm_inputs.mrope_positions = positions.unsqueeze(0).expand(3, -1).clone()
        mm_inputs.mrope_position_delta = torch.tensor([0], dtype=torch.long)

        temperature = float(params.get("temperature") or 0.0)
        if temperature == 0.0:
            # Qwen3-ASR degenerates under pure-greedy (emits only the language
            # tag then EOS); upstream uses 0.01 near-greedy.
            temperature = 0.01
        request_max_new_tokens = qwen3_asr_request_output_budget(
            params,
            audio_duration_s=audio_duration_s,
            default_max_new_tokens=max_new_tokens,
        )
        logger.debug(
            f"[qwen3-asr] sampling temp={temperature} "
            f"max_new_tokens={request_max_new_tokens} params={dict(params)}"
        )
        sampling_params = SamplingParams(
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            stop_token_ids=[eos_token_id],
        )
        sampling_params.normalize(tokenizer=None)

        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=input_ids,
            sampling_params=sampling_params,
            vocab_size=vocab_size,
            extra_key=fingerprint,
        )
        req.multimodal_inputs = mm_inputs
        req._codec_suppress_tokens = None

        return Qwen3ASRRequestData(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            req=req,
            prompt_token_ids=input_ids,
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            audio_duration_s=audio_duration_s,
            language=str(params.get("language") or "en"),
            engine_start_s=time.perf_counter(),
            stage_payload=payload,
        )

    def result_adapter(data: Qwen3ASRRequestData) -> StagePayload:
        payload = data.stage_payload
        output_ids = list(data.output_ids or [])
        # Keep the marker handling at token level. Byte-level BPE decode->encode
        # is not an identity transform for all whitespace/Unicode transcripts.
        raw = _decode_token_ids(tokenizer, output_ids, skip_special_tokens=False)
        logger.debug(
            f"[qwen3-asr] n_out={len(output_ids)} ids={output_ids[:40]} raw={raw!r}"
        )
        asr_text_idx = _find_subsequence(output_ids, asr_text_token_ids)
        transcript_ids = (
            output_ids[asr_text_idx + len(asr_text_token_ids) :]
            if asr_text_idx is not None
            else output_ids
        )
        text = _decode_token_ids(tokenizer, transcript_ids, skip_special_tokens=True)
        engine_time_s = (
            time.perf_counter() - data.engine_start_s if data.engine_start_s else 0.0
        )
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "text": text,
                "language": data.language,
                "duration_s": data.audio_duration_s,
                "asr_latency_s": engine_time_s,
                "usage": {"engine_time_s": engine_time_s},
                "modality": "text",
            },
        )

    return request_builder, result_adapter


__all__ = [
    "QWEN3_ASR_MAX_AUDIO_SECONDS",
    "QWEN3_ASR_UPSTREAM_MAX_NEW_TOKENS",
    "Qwen3ASRRequestData",
    "build_qwen3_asr_prompt_ids",
    "make_qwen3_asr_scheduler_adapters",
    "qwen3_asr_auto_output_budget",
    "qwen3_asr_prompt_overhead_tokens",
    "qwen3_asr_request_output_budget",
]
