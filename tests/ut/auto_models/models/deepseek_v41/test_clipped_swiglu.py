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
"""CPU contract tests for the bounded packed SwiGLU autograd adapter."""

import unittest
from unittest.mock import patch

import torch
from torch.nn import functional

from hyper_parallel.models.deepseek_v41.adapter import clipped_swiglu
from tests.common.mark_utils import arg_mark


class TestPackedSwiGLU(unittest.TestCase):
    """Check clipping derivatives, stored state, chunk boundaries, and validation."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_forward_and_gradient_match_reference(self) -> None:
        """
        Feature: Bounded packed SwiGLU.
        Description: Exercise saturation, exact boundaries, disabled limits, and partial chunks.
        Expectation: Outputs are exact and gradients agree within FP32 rounding or exactly in BF16.
        """
        for dtype in (torch.float32, torch.bfloat16):
            for limit in (-1.0, 0.0, 1.5, 10.0):
                for rows in (0, 1, 11):
                    with self.subTest(dtype=dtype, limit=limit, rows=rows):
                        torch.manual_seed(71)
                        inputs = (torch.randn(rows, 14) * 13).to(dtype).requires_grad_()
                        with torch.no_grad():
                            if rows:
                                inputs[0] = torch.tensor(
                                    [-20, -limit, -1, 0, 1, limit, 20] * 2, dtype=dtype,
                                )
                        reference = inputs.detach().clone().requires_grad_()
                        gate, up = reference.float().chunk(2, dim=-1)
                        if limit > 0:
                            gate = gate.clamp(max=limit)
                            up = up.clamp(min=-limit, max=limit)
                        expected = (functional.silu(gate) * up).to(dtype)
                        actual = clipped_swiglu.deepseek_v41_packed_swiglu(inputs, limit, 4)
                        # Strided gradients must be accepted by both backends.
                        gradient = torch.randn(7, rows, dtype=dtype).transpose(0, 1)
                        expected.backward(gradient)
                        actual.backward(gradient)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(
                            inputs.grad, reference.grad,
                            rtol=3e-7 if dtype == torch.float32 else 0, atol=1e-30,
                        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_saved_state_and_backward_working_set(self) -> None:
        """
        Feature: Bounded activation memory.
        Description: Inspect saved tensor storage and derivative tile sizes.
        Expectation: Only the original packed tensor is retained and every tile respects the bound.
        """
        inputs = torch.randn(11, 14, dtype=torch.bfloat16, requires_grad=True)
        saved = []

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            """Record saved storage without cloning or detaching it."""
            saved.append(tensor)
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = clipped_swiglu.deepseek_v41_packed_swiglu(inputs, 10.0, 4)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].data_ptr(), inputs.data_ptr())
        with patch.object(clipped_swiglu, "_backward_chunk", wraps=clipped_swiglu._backward_chunk) as backward:
            output.backward(torch.ones_like(output))
        self.assertEqual([call.args[1].shape[0] for call in backward.call_args_list], [4, 4, 3])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_reject_invalid_contract(self) -> None:
        """
        Feature: Packed activation boundary validation.
        Description: Supply malformed layouts, dtypes, limits, and chunk sizes.
        Expectation: Invalid contracts fail before any native kernel dispatch.
        """
        for tensor in (torch.zeros(3, 5), torch.zeros(2, 0), torch.zeros(2, 3, 4),
                       torch.zeros(2, 4, dtype=torch.float16), torch.zeros(4, 2).t()):
            with self.subTest(shape=tensor.shape, dtype=tensor.dtype):
                with self.assertRaises(ValueError):
                    clipped_swiglu.deepseek_v41_packed_swiglu(tensor, 10.0)
        for limit in (True, float("inf"), float("nan"), "10"):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                clipped_swiglu.deepseek_v41_packed_swiglu(torch.zeros(2, 4), limit)
        for chunk in (True, 0, -1, 1.5):
            with self.subTest(chunk=chunk), self.assertRaises(ValueError):
                clipped_swiglu.deepseek_v41_packed_swiglu(torch.zeros(2, 4), 10, chunk)
