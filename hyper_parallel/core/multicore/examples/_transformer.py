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

"""Attention, dense FFN and output math shared by independent example architectures."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.components.functional.rotary_embedding import apply_rotary_pos_emb


@dataclass(frozen=True)
class TransformerConfig:
    """Text model dimensions independent of routing and residual architecture."""

    vocab_size: int = 32000
    hidden_size: int = 5120
    num_layers: int = 2
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    max_seq_len: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000_000.0

    def __post_init__(self) -> None:
        """Reject invalid shared model dimensions before allocating parameters."""
        for name in ("vocab_size", "hidden_size", "num_layers", "num_attention_heads",
                     "num_key_value_heads", "max_seq_len"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("attention head dimension must be even for rotary embeddings")


class TransformerAttention(nn.Module):
    """Simplified grouped-query attention for the integration example."""

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize QKV, output, and rotary projections."""
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_causal = True
        inverse_frequency = config.rope_theta ** (
            -torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
        )
        angles = torch.outer(torch.arange(config.max_seq_len, dtype=torch.float32), inverse_frequency)
        angles = torch.cat((angles, angles), dim=-1)
        self.register_buffer("rotary_cos", angles.cos(), persistent=False)
        self.register_buffer("rotary_sin", angles.sin(), persistent=False)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run fused causal grouped-query attention.

        Args:
            hidden_states: Input activations preserving the batch, sequence and hidden dimensions.
            position_ids: Position indices used by the rotary embedding.
        """
        batch_size, sequence_length, _ = hidden_states.shape
        query = (
            self.q_proj(hidden_states)
            .view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        key = (
            self.k_proj(hidden_states)
            .view(
                batch_size,
                sequence_length,
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        value = (
            self.v_proj(hidden_states)
            .view(
                batch_size,
                sequence_length,
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        cos = self.rotary_cos[position_ids].to(hidden_states.dtype)
        sin = self.rotary_sin[position_ids].to(hidden_states.dtype)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if hidden_states.device.type == "npu":
            # Import the optional Ascend backend only when executing on NPU.
            from hyper_parallel.components.functional.npu_fusion_attention import (  # pylint: disable=C0415
                npu_fusion_attention_forward,
            )
            output, _ = npu_fusion_attention_forward(
                self, query, key, value, attention_mask=None, dropout=0.0, scaling=self.scaling,
            )
        else:
            repeats = self.num_heads // self.num_key_value_heads
            output = F.scaled_dot_product_attention(  # pylint: disable=not-callable
                query, key.repeat_interleave(repeats, dim=1), value.repeat_interleave(repeats, dim=1),
                is_causal=True, scale=self.scaling,
            ).transpose(1, 2).contiguous()
        return self.o_proj(output.reshape(batch_size, sequence_length, -1))


class DenseFeedForward(nn.Module):
    """Ordinary dense SwiGLU without a Router or expert parallelism."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        """Construct one dense gate/up projection and one down projection."""
        super().__init__()
        self.gate_up = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Evaluate a dense SwiGLU block on the current device.

        Args:
            hidden_states: Token activations shaped [..., hidden_size].
        """
        gate, up = self.gate_up(hidden_states).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


def language_model_output(
    hidden_states: torch.Tensor, labels: torch.Tensor | None, norm: nn.Module, head: nn.Module,
) -> dict[str, torch.Tensor | None]:
    """Project token states and compute optional next-token cross entropy.

    Args:
        hidden_states: Final token states shaped [batch, sequence, hidden].
        labels: Optional next-token labels with matching leading dimensions.
        norm: Model's final normalization.
        head: Vocabulary projection.
    """
    logits = head(norm(hidden_states))
    loss = None
    if labels is not None:
        shifted = F.pad(labels, (0, 1), value=-100)[..., 1:]
        loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), shifted.reshape(-1), ignore_index=-100)
    return {"loss": loss, "logits": logits}
