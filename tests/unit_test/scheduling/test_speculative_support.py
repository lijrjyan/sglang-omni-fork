# SPDX-License-Identifier: Apache-2.0
"""Algorithm rules, the model-support defaults and the scheduler bridge seams."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.scheduling import speculative_bridge, speculative_support
from sglang_omni.scheduling.speculative_model_support import (
    SpeculativeModelSupport,
    assign_spec_role,
)


@pytest.mark.parametrize(
    ("algorithm", "rule", "graph"),
    [
        ("STANDALONE", "chain", "decode+extend"),
        ("eagle3", "chain", "decode+extend"),
        ("DFLASH", "block", "verify"),
        ("DSPARK", "block", "verify"),
    ],
)
def test_algorithm_rule_tables_cover_every_admitted_algorithm(algorithm, rule, graph):
    assert speculative_support.window_rule(algorithm).value == rule
    assert speculative_support.draft_graph_kind(algorithm).value == graph
    assert speculative_support.takes_step_count(algorithm) is (rule == "chain")


def test_rule_tables_refuse_unadmitted_algorithms() -> None:
    with pytest.raises(ValueError, match="only supports STANDALONE, EAGLE3, DFLASH"):
        speculative_support.window_rule("NGRAM")


def test_window_rules_normalise_chain_and_block_knobs() -> None:
    chain: dict = {}
    speculative_support.apply_window_rule(chain, "EAGLE3")
    assert chain == {"speculative_num_steps": 3, "speculative_num_draft_tokens": 4}

    block = {"speculative_num_steps": 3}
    speculative_support.apply_window_rule(block, "DFLASH")
    assert block == {"speculative_num_draft_tokens": 8}

    with pytest.raises(ValueError, match="speculative_num_steps \\+ 1"):
        speculative_support.apply_window_rule(
            {"speculative_num_steps": 2, "speculative_num_draft_tokens": 4},
            "STANDALONE",
        )


def test_eager_prefill_hidden_axis_is_standalone_only() -> None:
    standalone = SimpleNamespace(is_standalone=lambda: True)
    eagle = SimpleNamespace(is_standalone=lambda: False)
    assert speculative_support.eager_prefill_has_hidden(standalone) is False
    assert speculative_support.eager_prefill_has_hidden(eagle) is True


def test_model_support_defaults_are_identity_and_assign_roles() -> None:
    support = SpeculativeModelSupport()
    target, draft = SimpleNamespace(), SimpleNamespace()

    assert support.validate_speculative_draft(None, None, "EAGLE3", 4) is None
    assert support.reject_speculative_request(object()) is None
    assert support.position_reserve(4) == 0
    mm_inputs = object()
    assert support.draft_view_of_request(mm_inputs) is mm_inputs

    support.link_models(target, draft)
    assert (target.spec_role, draft.spec_role) == ("target", "draft")
    # D8: the pre-unification Whisper name stays readable.
    assert (target.encoder_share_role, draft.encoder_share_role) == ("target", "draft")

    model = SimpleNamespace()
    assign_spec_role(model, "draft")
    assert model.spec_role == model.encoder_share_role == "draft"


def test_bridge_filters_spec_rows_with_signature_compatible_keywords() -> None:
    calls = []

    class EagleLike:
        def filter_batch(self, new_indices, has_been_filtered, new_indices_cpu):
            calls.append(
                ("eagle", new_indices.numel(), has_been_filtered, new_indices_cpu)
            )

    class DFlashLike:
        def filter_batch(self, new_indices, new_indices_cpu):
            calls.append(("dflash", new_indices.numel(), new_indices_cpu))

    speculative_bridge.clear_spec_info_rows(EagleLike(), torch.device("cpu"))
    speculative_bridge.clear_spec_info_rows(DFlashLike(), torch.device("cpu"))
    assert calls == [("eagle", 0, False, []), ("dflash", 0, [])]


def test_bridge_validation_messages_are_unchanged() -> None:
    algorithm = SimpleNamespace(
        is_standalone=lambda: True, is_eagle3=lambda: False, is_dflash=lambda: False
    )
    kwargs = dict(
        requested_algorithm="STANDALONE",
        enable_overlap=False,
        enable_async_decode=False,
        target_worker_adapter=object(),
        draft_worker=object(),
    )
    speculative_bridge.validate_scheduler_speculative_config(algorithm, **kwargs)
    with pytest.raises(ValueError, match="requires enable_overlap=False"):
        speculative_bridge.validate_scheduler_speculative_config(
            algorithm, **{**kwargs, "enable_overlap": True}
        )
    with pytest.raises(ValueError, match="requires enable_async_decode=False"):
        speculative_bridge.validate_scheduler_speculative_config(
            algorithm, **{**kwargs, "enable_async_decode": True}
        )
    with pytest.raises(ValueError, match="prebuilt target and draft workers"):
        speculative_bridge.validate_scheduler_speculative_config(
            algorithm, **{**kwargs, "draft_worker": None}
        )
    assert speculative_bridge.refuses_omni_lookahead(
        SimpleNamespace(is_none=lambda: False)
    )
    assert not speculative_bridge.refuses_omni_lookahead(
        SimpleNamespace(is_none=lambda: True)
    )
