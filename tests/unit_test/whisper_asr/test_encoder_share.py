# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.whisper_asr.encoder_share import EncoderStateShare
from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder


def test_publish_then_take_returns_batch_in_request_order_and_consumes() -> None:
    share = EncoderStateShare()
    flat = torch.arange(12, dtype=torch.float32).reshape(6, 2)

    share.publish([7, 3], flat, [4, 2])
    taken = share.take([3, 7])

    assert torch.equal(taken, torch.cat([flat[4:6], flat[0:4]], dim=0))
    assert share.take([3]) is None
    assert (share.published, share.consumed) == (2, 2)


def test_take_is_all_or_nothing_and_entries_are_bounded() -> None:
    share = EncoderStateShare(max_entries=2)
    flat = torch.zeros(3, 1)

    share.publish([1, 2, 3], flat, [1, 1, 1])

    assert share.take([1, 2]) is None
    assert share.misses == 1
    assert share.take([2, 3]) is not None


def test_builder_links_target_and_draft_models_when_enabled() -> None:
    builder = WhisperASREngineBuilder(
        max_running_requests=4, max_new_tokens=32, mem_fraction_static=0.2
    )
    target = SimpleNamespace()
    draft = SimpleNamespace()

    builder.setup_speculative_models(target, draft)

    assert target.encoder_share is draft.encoder_share
    assert (target.encoder_share_role, draft.encoder_share_role) == ("target", "draft")

    off = WhisperASREngineBuilder(
        max_running_requests=4,
        max_new_tokens=32,
        mem_fraction_static=0.2,
        speculative_share_encoder=False,
    )
    plain = SimpleNamespace()
    off.setup_speculative_models(plain, SimpleNamespace())
    assert not hasattr(plain, "encoder_share")
