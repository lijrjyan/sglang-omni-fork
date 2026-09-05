# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 EAGLE3 speculative decoding wiring (CPU-only)."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3 import engine_builder as engine_builder_module
from sglang_omni.models.fun_cosyvoice3.engine_builder import (
    FunCosyVoice3EngineBuilder,
    materialize_cosyvoice3_eagle3_draft,
)
from sglang_omni.models.fun_cosyvoice3.model_runner import FunCosyVoice3ModelRunner
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    CosyVoice3SGLangRequestData,
    apply_sglang_cosyvoice3_result,
    make_cosyvoice3_scheduler_adapters,
)
from sglang_omni.models.fun_cosyvoice3.sglang_model import (
    EAGLE3_DRAFT_ARCHITECTURE,
    EOS_ID,
    TOTAL_VOCAB_SIZE,
    CosyVoice3Eagle3LlamaForCausalLM,
    FunCosyVoice3SGLangModel,
    normalize_cosyvoice3_eagle3_config,
)
from sglang_omni.proto import OmniRequest, StagePayload


def _draft_config(**overrides) -> dict:
    config = {
        "architectures": [EAGLE3_DRAFT_ARCHITECTURE],
        "hidden_size": 896,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "num_hidden_layers": 24,
        "tie_word_embeddings": True,
        "vocab_size": TOTAL_VOCAB_SIZE,
        "draft_vocab_size": TOTAL_VOCAB_SIZE,
    }
    config.update(overrides)
    return config


def _write_draft(tmp_path, config: dict) -> str:
    draft_dir = tmp_path / "draft"
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text(json.dumps(config))
    (draft_dir / "model.safetensors").write_bytes(b"weights")
    return str(draft_dir)


def test_normalize_eagle3_config_untangles_published_fields() -> None:
    config = SimpleNamespace(tie_word_embeddings=True, num_hidden_layers=24)

    normalize_cosyvoice3_eagle3_config(config)

    assert config.tie_word_embeddings is False
    assert config.num_hidden_layers == 1
    assert config.draft_vocab_size == TOTAL_VOCAB_SIZE


def test_materialize_draft_rewrites_config_and_links_weights(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SGLANG_OMNI_CACHE_DIR", str(tmp_path / "cache"))
    source = _write_draft(tmp_path, _draft_config())
    monkeypatch.setattr(engine_builder_module, "_resolve_checkpoint", lambda p: p)

    target = materialize_cosyvoice3_eagle3_draft(source)

    with open(os.path.join(target, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    assert config["tie_word_embeddings"] is False
    assert config["num_hidden_layers"] == 1
    assert config["architectures"] == [EAGLE3_DRAFT_ARCHITECTURE]
    assert config["_sglang_omni_source"] == source
    weights = os.path.join(target, "model.safetensors")
    assert os.path.islink(weights)
    assert open(weights, "rb").read() == b"weights"
    # Re-materializing is idempotent and stable.
    assert materialize_cosyvoice3_eagle3_draft(source) == target


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"architectures": ["LlamaForCausalLMEagle3"]}, "expects an EAGLE3 draft"),
        ({"vocab_size": 158720, "draft_vocab_size": 8192}, "speech vocab"),
    ],
)
def test_materialize_draft_rejects_incompatible_checkpoints(
    tmp_path, monkeypatch: pytest.MonkeyPatch, overrides: dict, match: str
) -> None:
    monkeypatch.setenv("SGLANG_OMNI_CACHE_DIR", str(tmp_path / "cache"))
    source = _write_draft(tmp_path, _draft_config(**overrides))
    monkeypatch.setattr(engine_builder_module, "_resolve_checkpoint", lambda p: p)

    with pytest.raises(ValueError, match=match):
        materialize_cosyvoice3_eagle3_draft(source)


def test_engine_builder_rejects_unsupported_algorithm() -> None:
    with pytest.raises(ValueError, match="supports \\['EAGLE3'\\]"):
        FunCosyVoice3EngineBuilder(speculative_algorithm="STANDALONE")


def test_engine_builder_adjust_overrides_materializes_draft(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SGLANG_OMNI_CACHE_DIR", str(tmp_path / "cache"))
    source = _write_draft(tmp_path, _draft_config())
    monkeypatch.setattr(engine_builder_module, "_resolve_checkpoint", lambda p: p)
    builder = FunCosyVoice3EngineBuilder(
        enable_speculative=True,
        speculative_draft_model_path=source,
        speculative_use_rejection_sampling=False,
    )
    builder._target_hf_config = SimpleNamespace(
        hidden_size=896, num_attention_heads=14, num_key_value_heads=2
    )
    overrides = {
        "speculative_algorithm": "EAGLE3",
        "speculative_draft_model_path": source,
        "speculative_num_draft_tokens": 4,
    }

    builder.adjust_overrides(overrides)

    assert overrides["speculative_draft_model_path"] != source
    assert os.path.exists(
        os.path.join(overrides["speculative_draft_model_path"], "config.json")
    )
    assert overrides["speculative_use_rejection_sampling"] is False
    assert overrides["disable_overlap_schedule"] is True
    assert builder.enable_speculative is True


def test_engine_builder_adjust_overrides_rejects_hidden_size_mismatch(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SGLANG_OMNI_CACHE_DIR", str(tmp_path / "cache"))
    source = _write_draft(tmp_path, _draft_config(hidden_size=1024))
    monkeypatch.setattr(engine_builder_module, "_resolve_checkpoint", lambda p: p)
    builder = FunCosyVoice3EngineBuilder(enable_speculative=True)
    builder._target_hf_config = SimpleNamespace(
        hidden_size=896, num_attention_heads=14, num_key_value_heads=2
    )

    with pytest.raises(ValueError, match="hidden_size"):
        builder.adjust_overrides(
            {
                "speculative_algorithm": "EAGLE3",
                "speculative_draft_model_path": source,
                "speculative_num_draft_tokens": 4,
            }
        )


def test_engine_builder_without_speculative_leaves_overrides_alone() -> None:
    builder = FunCosyVoice3EngineBuilder()
    overrides = {"speculative_algorithm": None, "max_running_requests": 32}

    builder.adjust_overrides(overrides)

    assert overrides == {"speculative_algorithm": None, "max_running_requests": 32}
    assert builder.enable_speculative is False


def test_target_shares_speech_embedding_and_head() -> None:
    model = object.__new__(FunCosyVoice3SGLangModel)
    torch.nn.Module.__init__(model)
    model.speech_embedding = torch.nn.Embedding(TOTAL_VOCAB_SIZE, 4)
    model.llm_decoder = torch.nn.Linear(4, TOTAL_VOCAB_SIZE, bias=False)

    embed, head = model.get_embed_and_head()

    assert embed is model.speech_embedding.weight
    assert head is model.llm_decoder.weight
    assert model.get_input_embeddings() is model.speech_embedding


def test_target_forward_embeds_ids_and_publishes_prompt_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object.__new__(FunCosyVoice3SGLangModel)
    torch.nn.Module.__init__(model)
    model.speech_embedding = torch.nn.Embedding(TOTAL_VOCAB_SIZE, 4)
    model.llm_decoder = torch.nn.Linear(4, TOTAL_VOCAB_SIZE, bias=False)
    model.capture_aux_hidden_states = True
    seen = {}

    def fake_backbone(input_ids, positions, forward_batch, input_embeds, **kwargs):
        seen["input_embeds"] = input_embeds
        return input_embeds * 2, [input_embeds, input_embeds]

    model.model = fake_backbone
    model.logits_processor = lambda ids, hidden, head, fb, aux: (hidden, head, aux)

    class Mode:
        def __init__(self, extend: bool) -> None:
            self._extend = extend

        def is_extend(self) -> bool:
            return self._extend

    # Decode / verify rows: ids are speech tokens embedded in-model.
    ids = torch.tensor([1, 2, EOS_ID])
    decode_batch = SimpleNamespace(forward_mode=Mode(False), mm_input_embeds=None)
    hidden, head, aux = model.forward(ids, torch.arange(3), decode_batch)
    assert torch.equal(seen["input_embeds"], model.speech_embedding(ids))
    assert head is model.llm_decoder and len(aux) == 2
    assert decode_batch.mm_input_embeds is None

    # Prefill on projected rows publishes them for the draft.
    prompt = torch.randn(3, 4)
    extend_batch = SimpleNamespace(forward_mode=Mode(True), mm_input_embeds=None)
    model.forward(ids, torch.arange(3), extend_batch, input_embeds=prompt)
    assert extend_batch.mm_input_embeds is prompt
    assert seen["input_embeds"] is prompt


def _draft_stub(shift: bool) -> CosyVoice3Eagle3LlamaForCausalLM:
    draft = object.__new__(CosyVoice3Eagle3LlamaForCausalLM)
    torch.nn.Module.__init__(draft)
    draft.model = SimpleNamespace(embed_tokens=torch.nn.Embedding(TOTAL_VOCAB_SIZE, 2))
    draft.shift_prompt_embeds = shift
    return draft


def test_draft_prompt_rows_are_rolled_and_seeded_with_sampled_token() -> None:
    draft = _draft_stub(shift=True)
    prompt = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    # Two requests: rows [0, 3) and [3, 5); draft ids already rotated by SGLang.
    input_ids = torch.tensor([9, 9, 7, 9, 8])
    forward_batch = SimpleNamespace(
        extend_start_loc=torch.tensor([0, 3]), extend_seq_lens=torch.tensor([3, 2])
    )

    rows = draft.prompt_input_embeds(prompt, input_ids, forward_batch)

    embed = draft.model.embed_tokens
    assert torch.equal(rows[0], prompt[1])
    assert torch.equal(rows[1], prompt[2])
    assert torch.equal(rows[2], embed(torch.tensor(7)))
    assert torch.equal(rows[3], prompt[4])
    assert torch.equal(rows[4], embed(torch.tensor(8)))


def test_draft_prompt_rows_unshifted_variant_keeps_target_rows() -> None:
    draft = _draft_stub(shift=False)
    prompt = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    input_ids = torch.tensor([9, 9, 5])
    forward_batch = SimpleNamespace(
        extend_start_loc=torch.tensor([0]), extend_seq_lens=torch.tensor([3])
    )

    rows = draft.prompt_input_embeds(prompt, input_ids, forward_batch)

    assert torch.equal(rows[:2], prompt[:2])
    assert torch.equal(rows[2], draft.model.embed_tokens(torch.tensor(5)))
    assert torch.equal(prompt[2], torch.tensor([2.0, 2.0]))


def test_runner_speculative_prefill_rows_follow_request_order() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.model = torch.nn.Linear(3, 3, bias=False)
    requests = [
        SimpleNamespace(
            data=SimpleNamespace(
                req=SimpleNamespace(
                    extend_range=SimpleNamespace(length=1), prefix_indices=[]
                ),
                prompt_input_embeds=torch.tensor([[1.0, 1.0, 1.0]]),
            )
        ),
        SimpleNamespace(
            data=SimpleNamespace(
                req=SimpleNamespace(
                    extend_range=SimpleNamespace(length=2), prefix_indices=[0]
                ),
                prompt_input_embeds=torch.tensor(
                    [[9.0, 9.0, 9.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
                ),
            )
        ),
    ]

    rows = runner.speculative_prefill_input_embeds(object(), requests)

    assert torch.equal(
        rows, torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    )
    assert runner.speculative_prefill_input_embeds(object(), []) is None


def _payload() -> StagePayload:
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs={"input": "hello"}),
        data={"text": "hello", "seed": 0},
    )


def test_result_adapter_reads_native_output_ids_for_speculative_requests() -> None:
    req = SimpleNamespace(output_ids=[5, 6, EOS_ID])
    data = CosyVoice3SGLangRequestData(
        input_ids=torch.zeros(3, dtype=torch.long),
        max_new_tokens=8,
        temperature=0.0,
        output_ids=[5, 6, EOS_ID],
        req=req,
        speculative=True,
    )

    payload = apply_sglang_cosyvoice3_result(_payload(), data)

    codes = torch.as_tensor(payload.data["audio_codes"]).reshape(-1).tolist()
    assert codes == [5, 6]
    assert payload.data["completion_tokens"] == 2


def test_result_adapter_keeps_collected_codes_for_native_decode() -> None:
    data = CosyVoice3SGLangRequestData(
        input_ids=torch.zeros(2, dtype=torch.long),
        max_new_tokens=8,
        temperature=0.0,
        output_ids=[5, 6, EOS_ID],
        req=SimpleNamespace(output_ids=[5, 6, EOS_ID]),
        output_codes=[torch.tensor([7]), torch.tensor([8])],
    )

    payload = apply_sglang_cosyvoice3_result(_payload(), data)

    codes = torch.as_tensor(payload.data["audio_codes"]).reshape(-1).tolist()
    assert codes == [7, 8]


def test_scheduler_adapters_mark_speculative_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.fun_cosyvoice3 import request_builders

    built = CosyVoice3SGLangRequestData(
        input_ids=torch.zeros(1, dtype=torch.long),
        max_new_tokens=1,
        temperature=0.0,
        output_ids=[],
        req=SimpleNamespace(output_ids=[]),
    )
    monkeypatch.setattr(
        request_builders,
        "build_sglang_cosyvoice3_request",
        lambda payload, *, model: built,
    )
    request_builder, _ = make_cosyvoice3_scheduler_adapters(
        model=object(), enable_speculative=True
    )

    assert request_builder(_payload()).speculative is True
