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
"""NPU regressions for pageable DMA storage and checkpoint update parity."""

from contextlib import nullcontext
import ctypes
import os
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.activation_memory import cpu_offload


def _advise_owned_pages(tensor, advice):
    """Apply advice only to complete pages inside this owned allocation."""
    page = os.sysconf("SC_PAGE_SIZE")
    start = (tensor.data_ptr() + page - 1) // page * page
    end = (tensor.data_ptr() + tensor.numel() * tensor.element_size()) // page * page
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.madvise.restype = ctypes.c_int
    if libc.madvise(start, end - start, advice):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def test_dma_destination_survives_reclaim():
    """Recreate allocator lazy-free state before the real implementation copies."""
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.use_deterministic_algorithms(False)
    original_empty = torch.empty_like

    def lazy_allocation(*args, **kwargs):
        host = original_empty(*args, **kwargs)
        host.fill_(0.25)
        _advise_owned_pages(host, 15)  # MADV_NOHUGEPAGE for these test pages only.
        _advise_owned_pages(host, 8)  # MADV_FREE before the copy, never after it.
        return host

    source = torch.full((1024, 32768), 0.75, dtype=torch.bfloat16, device="npu")
    for _ in range(4):
        with patch.object(cpu_offload.torch, "empty_like", side_effect=lazy_allocation):
            host = cpu_offload.copy_to_pageable_cpu(source)
        _advise_owned_pages(host, 21)  # MADV_PAGEOUT on owned test storage.
        torch.testing.assert_close(host, torch.full_like(host, 0.75), rtol=0, atol=0)
        torch.testing.assert_close(host.to(source.device), source, rtol=0, atol=0)


def test_checkpoint_parameter_update_parity():
    """Check the storage contract through backward and an optimizer update."""
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.use_deterministic_algorithms(False)
    source = torch.linspace(-0.7, 0.7, 65536, device="npu").reshape(256, 256).t()
    reference = None
    for offload in (False, True):
        value = source.detach().clone().requires_grad_()
        weight = torch.nn.Parameter(torch.full_like(value, 0.125))
        optimizer = torch.optim.AdamW([weight], lr=0.01, foreach=False)
        context = cpu_offload.PageableSaveOnCpu() if offload else nullcontext()
        with context:
            output = checkpoint(lambda x, w: (x * w).sin().square(), value, weight, use_reentrant=False)
        output.sum().backward()
        tensors = [output.detach(), value.grad, weight.grad.clone()]
        optimizer.step()
        tensors.extend([weight.detach(), optimizer.state[weight]["exp_avg"],
                        optimizer.state[weight]["exp_avg_sq"]])
        if reference is None:
            reference = [item.clone() for item in tensors]
        else:
            for expected, actual in zip(reference, tensors):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
