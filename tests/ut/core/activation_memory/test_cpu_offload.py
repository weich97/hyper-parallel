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
"""Autograd and tensor contracts for pageable activation offload."""

import unittest

import torch
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.activation_memory.cpu_offload import PageableSaveOnCpu, copy_to_pageable_cpu


class TestPageableSaveOnCpu(unittest.TestCase):
    """Check semantic contracts without requiring accelerator hardware."""

    def test_cpu_views_remain_detached_aliases(self):
        """Preserve values and strides without creating an autograd cycle."""
        source = torch.arange(24.0).reshape(4, 6).t().requires_grad_()
        saved = copy_to_pageable_cpu(source)
        torch.testing.assert_close(saved, source, rtol=0, atol=0)
        self.assertEqual(saved.stride(), source.stride())
        self.assertEqual(saved.data_ptr(), source.data_ptr())
        self.assertFalse(saved.requires_grad)
        self.assertIsNone(saved.grad_fn)

    def test_empty_integer_and_boolean_inputs(self):
        """Checkpoint metadata and empty sentinels keep their dtype and shape."""
        for source in (torch.empty(0), torch.tensor([1, 3]), torch.tensor([True, False])):
            with self.subTest(dtype=source.dtype, shape=source.shape):
                saved = copy_to_pageable_cpu(source)
                torch.testing.assert_close(saved, source, rtol=0, atol=0)

    def test_checkpoint_forward_and_gradients(self):
        """Nested hooks preserve checkpoint outputs and first/second derivatives."""
        value = torch.linspace(-1.0, 1.0, 24, dtype=torch.float64).reshape(4, 6).t().requires_grad_()
        expected = value.sin().square().sum()
        reference = torch.autograd.grad(expected, value, create_graph=True)[0]
        reference_second = torch.autograd.grad(reference.sum(), value)[0]
        with PageableSaveOnCpu(), PageableSaveOnCpu():
            actual = checkpoint(lambda item: item.sin().square().sum(), value, use_reentrant=False)
        first = torch.autograd.grad(actual, value, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), value)[0]
        for left, right in ((actual, expected), (first, reference), (second, reference_second)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_sparse_layout_is_explicitly_rejected(self):
        """Avoid silently applying dense-storage assumptions to sparse tensors."""
        with self.assertRaisesRegex(ValueError, "strided tensor"):
            copy_to_pageable_cpu(torch.eye(3).to_sparse())
