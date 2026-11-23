"""Stage 62 (32-node / 40L, ZERO precision compromise):
  opt58 stack
  + fast sequence_load_balancing_loss counts via F.one_hot().float().sum(1) + dense_segment_sum (eliminates 79x ScatterElementsV2, ~75ms/step)
  + fast build_sliding_window_indices via clamp_min(-1) (eliminates INT64 MaskedFillAiCpu, ~19ms/step).
"""
from functools import wraps
from pathlib import Path
import runpy

import torch
import torch.distributed as dist
import torch.nn.functional as F

import hyper_parallel.components.modules.shared_compressed_dsa_attention as scda_mod
import hyper_parallel.models.deepseek_v41.adapter.aux_loss as aux_loss_mod
import hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 as modeling_mod

OPT58_PATH = Path("/xw-tree/full-v41/opt58-opt54-stack.py")


def fast_build_sliding_window_indices(
    batch_size: int,
    sequence_length: int,
    window_size: int,
    device: torch.device,
    *,
    query_offset: int = 0,
    key_length: int | None = None,
) -> torch.Tensor:
    """Build fixed-width causal sliding-window indices without AICPU MaskedFill."""
    if key_length is None:
        key_length = sequence_length
    width = min(key_length, window_size)
    query = torch.arange(
        query_offset,
        query_offset + sequence_length,
        device=device,
    ).unsqueeze(-1)
    offsets = torch.arange(width - 1, -1, -1, device=device)
    indices = (query - offsets).clamp_min(-1)
    return indices.unsqueeze(0).expand(batch_size, -1, -1)


def _dense_segment_sum(values: torch.Tensor, sequence_ids: torch.Tensor, num_sequences: int) -> torch.Tensor:
    if num_sequences > 8:
        return values.new_zeros(num_sequences, values.shape[-1]).index_add(0, sequence_ids, values)
    return torch.stack([
        (values * (sequence_ids == index).unsqueeze(-1)).sum(dim=0)
        for index in range(num_sequences)
    ])


def install():
    runpy.run_path(str(OPT58_PATH))["install"]()

    scda_mod.build_sliding_window_indices = fast_build_sliding_window_indices

    installed_seq_lb = aux_loss_mod.sequence_load_balancing_loss
    observer_counts = None
    for cell in getattr(installed_seq_lb, "__closure__", None) or ():
        val = cell.cell_contents
        if isinstance(val, dict) and "dense_calls" in val and "fallback_calls" in val:
            observer_counts = val
            break

    @wraps(installed_seq_lb)
    def fast_sequence_load_balancing_loss(
        scores: torch.Tensor,
        selected_experts: torch.Tensor,
        sequence_ids: torch.Tensor,
        num_sequences: int,
        token_mask: torch.Tensor | None = None,
        sequence_partition_groups: tuple = (),
    ) -> torch.Tensor:
        if not scores.is_npu or scores.shape[1] > 512:
            return installed_seq_lb(
                scores, selected_experts, sequence_ids, num_sequences,
                token_mask=token_mask, sequence_partition_groups=sequence_partition_groups,
            )
        if observer_counts is not None:
            observer_counts["dense_calls" if 1 < num_sequences <= 8 else "fallback_calls"] += 1
        if scores.ndim != 2 or selected_experts.ndim != 2 or scores.shape[0] != selected_experts.shape[0]:
            raise ValueError("scores and selected_experts must have shapes [T, E] and [T, K]")
        num_tokens, num_experts = scores.shape
        if num_sequences <= 0 or sequence_ids.shape != (num_tokens,) or sequence_ids.dtype != torch.long:
            raise ValueError("sequence_ids must be int64 [T] and num_sequences must be positive")
        top_k = selected_experts.shape[1]
        if not 0 < top_k <= num_experts:
            raise ValueError("top_k must be positive and no greater than num_experts")
        if token_mask is not None and (token_mask.shape != (num_tokens,) or token_mask.dtype != torch.bool):
            raise ValueError("token_mask must be a boolean tensor with shape [T]")

        scores = scores.float()
        probabilities = scores / scores.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
        oh = F.one_hot(selected_experts, num_classes=num_experts).to(torch.float32).sum(dim=1)
        if token_mask is not None:
            valid = token_mask.to(dtype=torch.float32).unsqueeze(-1)
            oh = oh * valid
            weighted_probabilities = probabilities * valid
        else:
            weighted_probabilities = probabilities

        if num_sequences == 1:
            counts = oh.sum(dim=0, keepdim=True)
            probability_sum = weighted_probabilities.sum(dim=0, keepdim=True)
        else:
            counts = _dense_segment_sum(oh, sequence_ids, num_sequences)
            probability_sum = _dense_segment_sum(weighted_probabilities, sequence_ids, num_sequences)

        for group in sequence_partition_groups:
            dist.all_reduce(counts, group=group)
            probability_sum = aux_loss_mod.differentiable_all_reduce(probability_sum, "sum", group)
        assignments = counts.sum(dim=-1, keepdim=True)
        token_count = (assignments / top_k).clamp_min(1.0)
        losses = num_experts * ((counts / assignments.clamp_min(1.0)) * (probability_sum / token_count)).sum(-1)
        return losses.sum() / (assignments > 0).sum().clamp_min(1)

    aux_loss_mod.sequence_load_balancing_loss = fast_sequence_load_balancing_loss
    modeling_mod.sequence_load_balancing_loss = fast_sequence_load_balancing_loss

    if not dist.is_initialized() or dist.get_rank() == 0:
        print("OPT62_AUX_MASK_INSTALLED=1", flush=True)
