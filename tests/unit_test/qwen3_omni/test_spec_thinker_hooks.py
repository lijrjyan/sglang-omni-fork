# SPDX-License-Identifier: Apache-2.0
"""Native speculative decoding hooks on the Qwen3-Omni text-only thinker."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.qwen3_omni.components.sglang_thinker import (
    Qwen3OmniThinkerForCausalLM,
)


class _Backbone:
    def __init__(self) -> None:
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.layers_to_capture: list[int] = []
        self.dflash_layers: list[int] | None = None
        self.return_aux = False

    def set_eagle3_layers_to_capture(self, layer_ids):
        self.layers_to_capture = list(layer_ids)

    def set_dflash_layers_to_capture(self, layer_ids):
        self.dflash_layers = list(layer_ids)

    def __call__(self, **kwargs):
        hidden = torch.ones(3, 4)
        if self.return_aux:
            return hidden, [torch.zeros(3, 4), torch.full((3, 4), 2.0)]
        return hidden


def _thinker(num_hidden_layers: int = 48) -> Qwen3OmniThinkerForCausalLM:
    thinker = Qwen3OmniThinkerForCausalLM.__new__(Qwen3OmniThinkerForCausalLM)
    torch.nn.Module.__init__(thinker)
    thinker.config = SimpleNamespace(num_hidden_layers=num_hidden_layers)
    thinker.model = _Backbone()
    thinker.lm_head = torch.nn.Linear(4, 8, bias=False)
    thinker._fused_rope_gate = None
    thinker.capture_aux_hidden_states = False
    calls: list[tuple] = []
    thinker.logits_processor = lambda *args: calls.append(args) or "logits"
    thinker._calls = calls
    return thinker


def test_eagle3_layer_ids_follow_upstream_shift() -> None:
    thinker = _thinker()

    thinker.set_eagle3_layers_to_capture([1, 23, 44])

    assert thinker.capture_aux_hidden_states is True
    assert thinker.model.layers_to_capture == [2, 24, 45]


def test_eagle3_default_layers_match_upstream_qwen3_moe() -> None:
    thinker = _thinker(num_hidden_layers=48)

    thinker.set_eagle3_layers_to_capture(None)

    assert thinker.model.layers_to_capture == [2, 24, 45]


def test_dflash_layer_ids_follow_upstream_shift() -> None:
    thinker = _thinker()

    thinker.set_dflash_layers_to_capture([1, 12, 23, 34, 45])

    assert thinker.capture_aux_hidden_states is True
    assert thinker.model.dflash_layers == [2, 13, 24, 35, 46]


def test_embed_and_head_are_shared_by_reference() -> None:
    thinker = _thinker()

    embed, head = thinker.get_embed_and_head()

    assert embed is thinker.model.embed_tokens.weight
    assert head is thinker.lm_head.weight
    assert thinker.get_input_embeddings() is thinker.model.embed_tokens


def test_forward_threads_aux_hidden_states_only_when_capturing() -> None:
    thinker = _thinker()
    forward_batch = SimpleNamespace(mrope_positions=None)
    input_ids = torch.arange(3)
    positions = torch.arange(3)

    assert thinker.forward(input_ids, positions, forward_batch) == "logits"
    (_, hidden, head, batch, aux) = thinker._calls[-1]
    assert aux is None and head is thinker.lm_head and batch is forward_batch

    thinker.set_eagle3_layers_to_capture([1, 23, 44])
    thinker.model.return_aux = True
    thinker.forward(input_ids, positions, forward_batch)
    (_, hidden, _, _, aux) = thinker._calls[-1]
    assert hidden.shape == (3, 4)
    assert len(aux) == 2 and float(aux[1][0, 0]) == 2.0
