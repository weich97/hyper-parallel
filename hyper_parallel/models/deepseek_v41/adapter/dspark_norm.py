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
"""Preserve the Torch DSpark normalization contract with the existing NPU kernel."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from hyper_parallel.models.replacement import module_replacement


def _native_rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Load the optional NPU primitive only when the selected module executes."""
    from hyper_parallel.components.functional.rms_norm import rms_norm  # pylint: disable=import-outside-toplevel

    return rms_norm(input, weight, eps)


@module_replacement
class DSparkRMSNorm(nn.Module):
    """Replace the affine, one-dimensional Torch RMSNorm used by DSpark."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """Keep the initialized parameter, epsilon and training state.

        Args:
            module: Source Torch RMSNorm with an explicit epsilon.
            module_fqn: Source path supplied by the replacement engine.
            context: Parallel context supplied by the replacement engine.
        """
        super().__init__()
        del module_fqn, context
        if not isinstance(module, nn.RMSNorm) or len(module.normalized_shape) != 1:
            raise TypeError("DSparkRMSNorm requires a one-dimensional torch.nn.RMSNorm")
        if not isinstance(module.weight, nn.Parameter) or module.eps is None:
            raise ValueError("DSparkRMSNorm requires an affine weight and an explicit epsilon")
        self.weight = module.weight
        self.eps = float(module.eps)
        self.train(module.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Preserve Torch's positional and ``x=`` keyword call forms."""
        return _native_rms_norm(x, self.weight, self.eps)
