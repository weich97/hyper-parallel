# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Verify Torch fully_shard hook state transitions with activation checkpointing."""
import copy
import os

os.environ.setdefault("HP_LOG_CONFIG", "FSDP:DEBUG")

# pylint: disable=C0413
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint

from hyper_parallel import SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.fully_shard.api import _extend_module_with_hsdp_interface
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy, OffloadPolicy
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2
from tests.torch.utils import init_dist_gloo


class _Block(nn.Module):
    """Small deterministic module managed by one fully_shard scheduler."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.arange(16, dtype=torch.float32).view(4, 4) / 16)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the block's deterministic linear transform and activation."""
        return torch.relu(inputs @ self.weight)


class _ConditionalModel(nn.Module):
    """Keep one FSDP child active only in the first accumulation microbatch."""

    def __init__(self) -> None:
        """Construct the common and conditionally executed blocks."""
        super().__init__()
        self.common = _Block()
        self.early = _Block()

    def forward(self, inputs: torch.Tensor, use_early: bool) -> torch.Tensor:
        """Run the common block and, when selected, the early block."""
        result = self.common(inputs)
        return result + self.early(inputs) if use_early else result


class _TracingHSDPScheduler(HSDPSchedulerV2):
    """Record scheduler transitions while retaining the production hook implementation."""

    def __init__(self, scenario: str, *args, **kwargs) -> None:
        self.scenario = scenario
        self.trace = []
        super().__init__(*args, **kwargs)

    @staticmethod
    def _state_name(state) -> str:
        return "NONE" if state is None else state.name

    def _record(self, event: str, before) -> None:
        before_name = self._state_name(before)
        after_name = self._state_name(self.scheduler_state)
        self.trace.append((event, before_name, after_name))
        print(
            f"FSDP_STATE_TRACE rank={dist.get_rank()} scenario={self.scenario} "
            f"event={event} before={before_name} after={after_name} "
            f"root_bp_state={self.scheduler_ctx.root_bp_state}",
            flush=True,
        )

    def _forward_pre_hook(self, cell, args, kwargs):
        before = self.scheduler_state
        result = super()._forward_pre_hook(cell, args, kwargs)
        self._record("forward_pre", before)
        return result

    def _forward_hook(self, cell, inputs, outputs):
        before = self.scheduler_state
        result = super()._forward_hook(cell, inputs, outputs)
        self._record("forward", before)
        return result

    def _backward_pre_hook(self, grad):
        before = self.scheduler_state
        result = super()._backward_pre_hook(grad)
        self._record("backward_pre", before)
        return result

    def _backward_hook(self):
        before = self.scheduler_state
        super()._backward_hook()
        self._record("backward", before)

    def _root_backward_hook(self):
        before = self.scheduler_state
        self._record("root_backward_enter", before)
        super()._root_backward_hook()
        self._record("root_backward_exit", before)


def _build_model(scenario: str, mesh) -> tuple[nn.Module, _TracingHSDPScheduler]:
    """Build a traced scheduler model for one hook-state scenario."""
    model = _Block()
    _extend_module_with_hsdp_interface(model)
    scheduler = _TracingHSDPScheduler(
        scenario,
        model,
        mesh,
        True,
        None,
        MixedPrecisionPolicy(),
        OffloadPolicy(),
        set(),
        set(),
        torch.device("cpu"),
        False,
    )
    model.hsdp_scheduler = scheduler
    return model, scheduler


def _run_scenario(scenario: str, mesh) -> list[tuple[str, str, str]]:
    """Run one hook-state scenario and return its scheduler transition trace."""
    model, scheduler = _build_model(scenario, mesh)
    inputs = torch.arange(8, dtype=torch.float32).view(2, 4) / 8

    if scenario == "normal":
        outputs = model(inputs)
    elif scenario == "non_reentrant":
        outputs = checkpoint(model, inputs, use_reentrant=False)
    elif scenario == "reentrant":
        dummy = torch.ones((), requires_grad=True)

        def run_with_dummy(tensor, unused_dummy):
            del unused_dummy
            return model(tensor)

        outputs = checkpoint(run_with_dummy, inputs, dummy, use_reentrant=True)
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    outputs.sum().backward()
    assert model.weight.grad is not None, (
        f"Expected a reduced gradient after {scenario} backward, got model.weight.grad={model.weight.grad}"
    )
    return scheduler.trace


def _accumulation_inputs(rank: int, micro: int, requires_grad: bool) -> torch.Tensor:
    """Create distinct, deterministic inputs for each rank and microbatch."""
    return torch.tensor(
        [[rank + 1.0, micro + 1.0, rank + micro + 1.0, 2.0]],
        dtype=torch.float32,
        requires_grad=requires_grad,
    )


def _assert_accumulation_gradients(model: nn.Module, reference: nn.Module, main_grad: bool) -> None:
    """Compare every reduced gradient shard to an ordinary Torch reference."""
    expected_params = dict(reference.named_parameters())
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    for name, param in model.named_parameters():
        grad = getattr(param, "main_grad", None) if main_grad else param.grad
        assert grad is not None, f"Expected a reduced gradient for {name}, got {grad}"
        expected = expected_params[name].grad.chunk(world_size, dim=0)[rank]
        torch.testing.assert_close(grad.to_local(), expected, rtol=0, atol=0)


def _assert_backward_boundary(model: nn.Module, expected_callbacks: int) -> None:
    """Verify per-backward state reset and exactly-once callback execution."""
    context = model.hsdp_scheduler.scheduler_ctx
    states = [scheduler.scheduler_state for scheduler in context.all_hsdp_schedulers]
    assert all(state is None for state in states), f"Expected idle scheduler states, got {states}"
    assert not context.post_backward_final_callback_queued, (
        f"Expected no queued root callback, got {context.post_backward_final_callback_queued}"
    )
    root_exits = [event for event in model.hsdp_scheduler.trace if event[0] == "root_backward_exit"]
    assert len(root_exits) == expected_callbacks, (
        f"Expected {expected_callbacks} root callbacks, got {len(root_exits)}"
    )


def _run_multiple_forward_backward(model: nn.Module, reference: nn.Module, input_grad: bool, main_grad: bool) -> None:
    """Check separate and combined backwards after multiple forwards."""
    rank = dist.get_rank()
    for source_rank in range(dist.get_world_size()):
        for micro in (3, 4):
            reference(_accumulation_inputs(source_rank, micro, input_grad), True).sum().backward()
    losses = [model(_accumulation_inputs(rank, micro, input_grad), True).sum() for micro in (3, 4)]
    for index, loss in enumerate(losses):
        model.set_requires_gradient_sync(index == 1)
        loss.backward()
        _assert_backward_boundary(model, index + 4)
    _assert_accumulation_gradients(model, reference, main_grad)

    for source_rank in range(dist.get_world_size()):
        for micro in (5, 6):
            reference(_accumulation_inputs(source_rank, micro, input_grad), True).sum().backward()
    losses = [model(_accumulation_inputs(rank, micro, input_grad), True).sum() for micro in (5, 6)]
    sum(losses).backward()
    _assert_backward_boundary(model, 6)
    _assert_accumulation_gradients(model, reference, main_grad)


def _run_nested_accumulation(mesh, comm_fusion: bool, input_grad: bool, main_grad: bool) -> None:
    """Compare accumulated and child-only gradients with an ordinary Torch model."""
    reference = _ConditionalModel()
    model = copy.deepcopy(reference)
    policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        cast_forward_inputs=False,
        apply_grad_on_fp32_main_grad=main_grad,
    )
    for unit in (model.common, model.early, model):
        _extend_module_with_hsdp_interface(unit)
        unit.hsdp_scheduler = _TracingHSDPScheduler(
            "nested_accumulation", unit, mesh, True, None, policy, OffloadPolicy(),
            set(), set(), torch.device("cpu"), comm_fusion,
        )
    model.set_reduce_op_type("sum")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    for source_rank in range(world_size):
        for micro in range(2):
            reference(_accumulation_inputs(source_rank, micro, input_grad), micro == 0).sum().backward()

    with SkipDTensorDispatch():
        for micro in range(2):
            model.set_requires_gradient_sync(micro == 1)
            model(_accumulation_inputs(rank, micro, input_grad), micro == 0).sum().backward()
            _assert_backward_boundary(model, micro + 1)
            early_param = model.early.hsdp_scheduler.hsdp_state.hsdp_params[0]
            if micro == 0:
                assert early_param.unsharded_accumulated_grad is not None, (
                    f"Expected no-sync gradient for early, got {early_param.unsharded_accumulated_grad}"
                )

        _assert_accumulation_gradients(model, reference, main_grad)
        assert early_param.unsharded_accumulated_grad is None, (
            f"Expected early's accumulated gradient to be reduced, got {early_param.unsharded_accumulated_grad}"
        )
        for source_rank in range(world_size):
            reference.early(_accumulation_inputs(source_rank, 2, input_grad)).sum().backward()
        model.early(_accumulation_inputs(rank, 2, input_grad)).sum().backward()
        _assert_backward_boundary(model, 3)
        _assert_accumulation_gradients(model, reference, main_grad)

        _run_multiple_forward_backward(model, reference, input_grad, main_grad)
        for scheduler in model.hsdp_scheduler.scheduler_ctx.all_hsdp_schedulers:
            for param in scheduler.hsdp_state.hsdp_params:
                assert param.unsharded_accumulated_grad is None, (
                    f"Expected no pending gradient for {param}, got {param.unsharded_accumulated_grad}"
                )


def test_torch_fully_shard_hook_state_machine() -> None:
    """Verify normal, non-reentrant, and reentrant hook state transitions with real autograd."""
    init_dist_gloo()
    mesh = init_device_mesh(
        device_type="cpu",
        mesh_shape=(dist.get_world_size(),),
        mesh_dim_names=("fsdp",),
    )
    expected_traces = {
        "normal": [
            ("forward_pre", "NONE", "PRE_FORWARD"),
            ("forward", "PRE_FORWARD", "FORWARD"),
            ("backward_pre", "FORWARD", "PRE_BACKWARD"),
            ("root_backward_enter", "PRE_BACKWARD", "PRE_BACKWARD"),
            ("backward", "PRE_BACKWARD", "BACKWARD"),
            ("root_backward_exit", "PRE_BACKWARD", "NONE"),
        ],
        "non_reentrant": [
            ("forward_pre", "NONE", "PRE_FORWARD"),
            ("forward", "PRE_FORWARD", "FORWARD"),
            ("backward_pre", "FORWARD", "PRE_BACKWARD"),
            ("forward_pre", "PRE_BACKWARD", "PRE_BACKWARD"),
            ("root_backward_enter", "PRE_BACKWARD", "PRE_BACKWARD"),
            ("backward", "PRE_BACKWARD", "BACKWARD"),
            ("root_backward_exit", "PRE_BACKWARD", "NONE"),
        ],
        "reentrant": [
            ("forward_pre", "NONE", "PRE_FORWARD"),
            ("forward", "PRE_FORWARD", "FORWARD"),
            ("forward_pre", "FORWARD", "PRE_FORWARD"),
            ("forward", "PRE_FORWARD", "FORWARD"),
            ("backward_pre", "FORWARD", "PRE_BACKWARD"),
            ("root_backward_enter", "PRE_BACKWARD", "PRE_BACKWARD"),
            ("backward", "PRE_BACKWARD", "BACKWARD"),
            ("root_backward_exit", "PRE_BACKWARD", "NONE"),
        ],
    }

    for scenario, expected_trace in expected_traces.items():
        actual_trace = _run_scenario(scenario, mesh)
        assert actual_trace == expected_trace, (
            f"Unexpected hook state trace for scenario={scenario}: "
            f"expected={expected_trace}, got={actual_trace}"
        )

    for comm_fusion in (False, True):
        for input_grad in (False, True):
            for main_grad in (False, True):
                _run_nested_accumulation(mesh, comm_fusion, input_grad, main_grad)
