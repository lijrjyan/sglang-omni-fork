"""Live verification penalties must survive graph replay and batch changes."""

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.model_worker import SpeculativeTargetWorkerAdapter
from sglang_omni.models.fun_cosyvoice3.sglang_model import FunCosyVoice3SGLangModel


@pytest.mark.parametrize("graph", [False, True])
def test_target_adapter_penalizes_after_forward_with_current_state(graph):
    model = SimpleNamespace()
    model.postprocess_speculative_logits = lambda logits, batch: (
        FunCosyVoice3SGLangModel.postprocess_speculative_logits(model, logits, batch)
    )
    raw = torch.tensor([[4.0, 8.0, 12.0], [4.0, 8.0, 12.0]])
    runner = SimpleNamespace(model=model)
    # A graph replay returns raw model logits without executing model.forward.
    runner.forward = lambda *args, **kwargs: SimpleNamespace(
        logits_output=SimpleNamespace(next_token_logits=raw.clone()),
        can_run_graph=graph,
        expert_distribution_metrics=None,
        routed_experts_output=None,
        indexer_topk_output=None,
    )
    worker = SimpleNamespace(
        model_runner=runner, server_args=None, model_config=None, device="cpu", gpu_id=0
    )
    adapter = SpeculativeTargetWorkerAdapter(worker, tokenizer=None)
    batch = SimpleNamespace(
        input_ids=torch.tensor([2, 1]),
        forward_mode=SimpleNamespace(is_target_verify=lambda: True),
        spec_algorithm=SimpleNamespace(is_dspark=lambda: True),
        apply_deprecated_skip_attn_backend_init=lambda value: None,
    )
    # Different requests / refreshed history must not reuse captured penalties.
    for penalty in [3.0, 1.0, 2.0]:
        batch.sampling_info = SimpleNamespace(
            acc_scaling_penalties=torch.tensor([[1.0, 2.0, 1.0]]),
            _omni_repetition_penalties=torch.tensor([[penalty]]),
        )
        result = adapter.forward_batch_generation(
            None, forward_batch=batch, is_verify=True
        )
        expected = torch.tensor(
            [[4.0, 4.0, 12.0 / penalty], [4.0, 8.0 / penalty, 12.0 / penalty]]
        )
        torch.testing.assert_close(result.logits_output.next_token_logits, expected)
        assert result.can_run_cuda_graph == graph


@pytest.mark.parametrize("required", [True, False])
def test_scheduler_publishes_current_draft_state_after_refresh(monkeypatch, required):
    import sglang_omni.models.fun_cosyvoice3.sglang_model as cosy
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    state = {"info": object(), "rep_values": torch.tensor([[9.0]])}
    monkeypatch.setattr(cosy, "_DRAFT_SAMPLING_INFO", state)
    model = object.__new__(FunCosyVoice3SGLangModel)
    scheduler = object.__new__(OmniScheduler)
    scheduler.spec_algorithm = SimpleNamespace(is_dspark=lambda: True)
    scheduler.model_worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    events = []
    scheduler._cumulate_speculative_output_tokens = lambda batch: events.append(
        "history"
    )

    # Filtering / a new request changes both row count and the info object.
    for size in [2, 1]:
        info = SimpleNamespace(
            penalizer_orchestrator=SimpleNamespace(is_required=required),
            acc_scaling_penalties=None,
            acc_additive_penalties=None,
        )

        def update():
            assert events[-1] == "history"
            info.acc_scaling_penalties = torch.full((size, 3), 1.3)
            events.append("updated")

        info.update_penalties = update
        batch = SimpleNamespace(
            sampling_info=info,
            device="cpu",
            reqs=[
                SimpleNamespace(
                    sampling_params=SimpleNamespace(
                        repetition_penalty=1.3 if required else 1.0
                    )
                )
                for _ in range(size)
            ],
        )
        scheduler._prepare_speculative_batch(batch)
        assert state["info"] is info
        if required:
            assert events[-1] == "updated"
            assert state["info"].acc_scaling_penalties.shape == (size, 3)
            torch.testing.assert_close(state["rep_values"], torch.full((size, 1), 1.3))
        else:
            assert state["rep_values"] is None
            assert not events

    # A batch without sampling info must also release the previous request.
    scheduler._prepare_speculative_batch(SimpleNamespace(sampling_info=None))
    assert state == {"info": None, "rep_values": None}
