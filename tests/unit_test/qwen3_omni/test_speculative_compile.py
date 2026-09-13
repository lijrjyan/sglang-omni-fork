# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from sglang_omni.models.qwen3_omni.bootstrap import create_thinker_scheduler
from sglang_omni.scheduling import generation_batch_policy


@pytest.mark.parametrize("algorithm", [None, "EAGLE3", "DFLASH"])
@pytest.mark.parametrize("compile_enabled", [False, True])
def test_thinker_rejects_dflash_compile_before_loading(
    monkeypatch, algorithm, compile_enabled
):
    class StartupReached(Exception):
        pass

    def stop_before_loading(server_args):
        raise StartupReached

    monkeypatch.setattr(
        generation_batch_policy, "get_prefill_cuda_graph_backend", stop_before_loading
    )
    server_args = SimpleNamespace(
        speculative_algorithm=algorithm, enable_torch_compile=compile_enabled
    )
    if algorithm == "DFLASH" and compile_enabled:
        with pytest.raises(ValueError, match="enable_torch_compile=False"):
            create_thinker_scheduler(server_args, enable_async_decode=False)
    else:
        with pytest.raises(StartupReached):
            create_thinker_scheduler(server_args, enable_async_decode=False)
