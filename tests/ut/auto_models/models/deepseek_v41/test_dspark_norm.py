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
"""Exercise DSpark normalization through the real module replacement engine."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel

from hyper_parallel.models.deepseek_v41.adapter.dspark_norm import DSparkRMSNorm
from hyper_parallel.models.replacement import apply_module_replacements, compile_module_replacements
from hyper_parallel.trainer.config.parallelism import entries_to_module_replacements
from hyper_parallel.trainer.config.resolver import resolve_config


def _cpu_norm(input, weight, eps):
    return nn.functional.rms_norm(input, (weight.numel(),), weight, eps)


class TestDSparkRMSNorm(unittest.TestCase):
    """Check call compatibility, live parameter identity and meta initialization."""

    def test_declared_replacement_preserves_keyword_and_gradients(self):
        """The engine must accept x= and preserve every differentiable operand."""
        owner = nn.Module()
        owner.norm = nn.RMSNorm(16, eps=1e-6).eval()
        source = owner.norm
        with torch.no_grad():
            source.weight.copy_(torch.linspace(0.7, 1.3, 16))
        config = resolve_config({
            "model": {"_target_": "torch.nn.Identity"},
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.01},
            "plan_overrides": [{"match": "norm", "replace": {
                "kind": "module", "implementation": {
                    "_target_": "hyper_parallel.models.deepseek_v41.adapter.dspark_norm.DSparkRMSNorm"}}}],
        })
        plan = compile_module_replacements(owner, entries_to_module_replacements(config.plan_overrides))
        owner = apply_module_replacements(owner, plan, context={})[0]
        self.assertIs(owner.norm.weight, source.weight)
        self.assertEqual(owner.norm.eps, source.eps)
        self.assertFalse(owner.norm.training)
        self.assertEqual(set(owner.state_dict()), {"norm.weight"})
        inputs = torch.linspace(-2, 2, 7 * 16).reshape(1, 7, 16).requires_grad_()
        expected = source(x=inputs)
        with patch("hyper_parallel.models.deepseek_v41.adapter.dspark_norm._native_rms_norm", _cpu_norm):
            actual = owner.norm(x=inputs)
            torch.testing.assert_close(owner.norm(inputs), expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for left, right in zip(torch.autograd.grad(actual.square().sum(), (inputs, source.weight)),
                               torch.autograd.grad(expected.square().sum(), (inputs, source.weight))):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_meta_materialization_uses_hf_norm_initialization(self):
        """The replacement must retain HF's identity-scale initialization path."""
        with torch.device("meta"):
            replacement = DSparkRMSNorm(module=nn.RMSNorm(16, eps=1e-6))
        replacement.to_empty(device="cpu")
        with torch.no_grad():
            replacement.weight.fill_(float("nan"))
        PreTrainedModel._init_weights(SimpleNamespace(config=SimpleNamespace(initializer_range=0.02)), replacement)
        torch.testing.assert_close(replacement.weight, torch.ones(16), rtol=0, atol=0)

    def test_rejects_unsupported_normalization_contracts(self):
        """Do not silently reinterpret missing weights, epsilon or multiple axes."""
        for source in (nn.LayerNorm(16), nn.RMSNorm((4, 4), eps=1e-6)):
            with self.subTest(source=source), self.assertRaisesRegex(TypeError, "one-dimensional"):
                DSparkRMSNorm(module=source)
        for source in (nn.RMSNorm(16), nn.RMSNorm(16, eps=1e-6, elementwise_affine=False)):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "affine weight"):
                DSparkRMSNorm(module=source)
