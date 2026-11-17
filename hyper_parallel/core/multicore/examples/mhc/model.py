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

"""Dense Transformer with four-stream shifted mHC and no routed experts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hyper_parallel.components.modules import RMSNorm
from hyper_parallel.core.multicore.examples._transformer import (
    DenseFeedForward, TransformerAttention, TransformerConfig, language_model_output,
)
from hyper_parallel.core.multicore.examples.mhc.mhc import MhcBoundary, _reference_residual


def _initialize_state(hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Initialize this model's four residual streams and identity mappings."""
    leading = hidden_states.shape[:-1]
    residual = hidden_states.unsqueeze(-2).expand(*leading, 4, hidden_states.shape[-1]).contiguous()
    pre = torch.zeros((*leading, 4), device=hidden_states.device, dtype=torch.float32)
    pre[..., 0] = 1.0
    identity = torch.eye(4, device=hidden_states.device, dtype=torch.float32)
    mix = identity.expand(*leading, 4, 4).contiguous()
    return torch.zeros_like(hidden_states), residual, pre, torch.zeros_like(pre), mix


def _finish_state(state: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Apply the final pending block output before collapsing the residual streams."""
    residual = _reference_residual(state)
    return (residual.float() * state[2].float().unsqueeze(-1)).sum(dim=-2).to(residual.dtype)


@dataclass(frozen=True)
class MhcTransformerConfig(TransformerConfig):
    """Single-device dense network dimensions independent of expert parallelism."""

    vocab_size: int = 1024
    hidden_size: int = 512
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    max_seq_len: int = 128
    intermediate_size: int = 2048

    def __post_init__(self) -> None:
        """Validate the ordinary dense dimensions before creating mHC boundaries."""
        super().__post_init__()
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")


class MhcTransformerDecoderLayer(nn.Module):
    """Shift explicit mHC state across attention and an ordinary dense FFN."""

    def __init__(self, config: MhcTransformerConfig, *, use_mega_mhc: bool) -> None:
        """Create two boundaries with identical dense blocks for either backend."""
        super().__init__()
        self.attention_mhc = MhcBoundary(config.hidden_size, config.rms_norm_eps, use_mega_mhc=use_mega_mhc)
        self.ffn_mhc = MhcBoundary(config.hidden_size, config.rms_norm_eps, use_mega_mhc=use_mega_mhc)
        self.self_attn = TransformerAttention(config)
        self.mlp = DenseFeedForward(config.hidden_size, config.intermediate_size)

    def forward(self, state: tuple[torch.Tensor, ...], position_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Carry the pending output and mappings across two dense computation blocks.

        Args:
            state: Previous block output, residual streams and pending mappings.
            position_ids: Token positions used by rotary attention.
        """
        residual, pre, post, mix, hidden = self.attention_mhc(*state)
        hidden = self.self_attn(hidden, position_ids)
        residual, pre, post, mix, hidden = self.ffn_mhc(hidden, residual, pre, post, mix)
        return self.mlp(hidden), residual, pre, post, mix


class MhcTransformerModel(nn.Module):
    """Single-device dense model selecting only the mHC boundary implementation."""

    def __init__(self, config: MhcTransformerConfig, *, use_mega_mhc: bool = True) -> None:
        """Construct dense attention/FFN and explicit four-stream state boundaries."""
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(MhcTransformerDecoderLayer(config, use_mega_mhc=use_mega_mhc)
                                    for _ in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        """Return logits and optional next-token loss while flushing the final mHC state.

        Args:
            input_ids: Token IDs shaped [batch, sequence].
            labels: Optional labels with matching leading dimensions.
        """
        hidden_states = self.embed_tokens(input_ids)
        positions = torch.arange(hidden_states.shape[1], device=hidden_states.device, dtype=torch.int64)
        state = _initialize_state(hidden_states)
        for layer in self.layers:
            state = layer(state, positions)
        return language_model_output(_finish_state(state), labels, self.norm, self.lm_head)
