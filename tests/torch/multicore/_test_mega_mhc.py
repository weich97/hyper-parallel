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
"""NPU worker for cached-plan isolation across concurrent mHC invocations."""

import os

import torch
import torch_npu  # noqa: F401  # pylint: disable=unused-import

from hyper_parallel.core.multicore.modules.mega_mhc.function import _hyper_mega_mhc_forward
from hyper_parallel.core.multicore.modules.mega_mhc_grad.function import hyper_mega_mhc_grad


def _make_case(seed: int) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Make two reproducible, distinct cases that share an LRU plan key."""
    torch.manual_seed(seed)
    device = torch.device("npu", torch.npu.current_device())
    tokens, hidden = 256, 512
    inputs = (
        torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16),
        torch.randn(tokens, 4, hidden, device=device, dtype=torch.bfloat16),
        torch.sigmoid(torch.randn(tokens, 4, device=device)),
        2 * torch.sigmoid(torch.randn(tokens, 4, device=device)),
        torch.softmax(torch.randn(tokens, 4, 4, device=device), dim=-1),
        torch.randn(24, 4 * hidden, device=device) / (4 * hidden) ** 0.5,
        torch.ones(3, device=device),
        torch.zeros(24, device=device),
        torch.ones(hidden, device=device, dtype=torch.bfloat16),
    )
    gradients = tuple(torch.randn_like(inputs[index]) * 0.1 for index in (1, 2, 3, 4, 0))
    return inputs, gradients


def _run_case(inputs: tuple[torch.Tensor, ...], gradients: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """Run paired forward/backward calls without a host or device synchronize."""
    outputs, cache = _hyper_mega_mhc_forward(*inputs, need_backward=True)
    grad_outputs = hyper_mega_mhc_grad(*gradients, *inputs, cache)
    return (*outputs, *grad_outputs)


def test_same_shape_multistream_forward_backward() -> None:
    """Compare concurrent invocations to serial native executions, not a performance golden."""
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    with torch.no_grad():
        cases = (_make_case(17), _make_case(23))
        # Warm both LRU plans on the default stream before testing counter isolation.
        expected = tuple(_run_case(*case) for case in cases)
        torch.npu.synchronize()
        default_stream = torch.npu.current_stream()
        streams = (torch.npu.Stream(), torch.npu.Stream())
        for stream in streams:
            stream.wait_stream(default_stream)
        pending = []
        for _ in range(4):
            for index, stream in enumerate(streams):
                with torch.npu.stream(stream):
                    pending.append((index, _run_case(*cases[index])))
        for stream in streams:
            stream.synchronize()
        for index, outputs in pending:
            for actual, serial in zip(outputs, expected[index]):
                torch.testing.assert_close(actual, serial, rtol=3e-3, atol=3e-3)
