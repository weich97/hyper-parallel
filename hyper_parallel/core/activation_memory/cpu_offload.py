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
"""Blocking activation offload to initialized pageable CPU storage."""

import torch


def copy_to_pageable_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Copy a dense device tensor to CPU storage that is safe for DMA writes.

    Args:
        tensor: Strided tensor to preserve for backward or checkpoint replay.

    Returns:
        Detached CPU tensor with the same values, shape and dtype. CPU inputs
        share their original storage; device inputs use independent storage.

    Raises:
        ValueError: The input does not have a strided layout.

    Note:
        The transfer completes before this function returns. Initializing the
        destination on CPU is required even when the copy overwrites it all:
        an allocator may return pages previously marked MADV_FREE. DMA does
        not necessarily clear that mark, allowing later reclaim to erase data.
        See https://github.com/pytorch/pytorch/issues/200905.
    """
    if tensor.layout != torch.strided:
        raise ValueError("Pageable activation offload requires a strided tensor")
    if tensor.device.type == "cpu":
        return tensor.detach()
    host = torch.empty_like(tensor, device="cpu", pin_memory=False, memory_format=torch.preserve_format)
    # CPU stores clear lazy-free state before the accelerator DMA writes data.
    host.zero_()
    host.copy_(tensor.detach(), non_blocking=False)
    return host


def _pack_to_cpu(tensor: torch.Tensor) -> tuple[torch.device, torch.Tensor]:
    return tensor.device, copy_to_pageable_cpu(tensor)


def _unpack_from_cpu(saved: tuple[torch.device, torch.Tensor]) -> torch.Tensor:
    device, tensor = saved
    return tensor.to(device=device, non_blocking=False)


class PageableSaveOnCpu(torch.autograd.graph.saved_tensors_hooks):
    """Save dense autograd tensors in initialized pageable CPU memory.

    Both directions use blocking copies. This context does not change the
    deterministic execution policy or use the pinned-memory allocator.
    It can wrap non-reentrant checkpointing to offload checkpoint inputs.
    """

    def __init__(self) -> None:
        """Install pack/unpack hooks with a synchronous transfer lifetime."""
        super().__init__(_pack_to_cpu, _unpack_from_cpu)
