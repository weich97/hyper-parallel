"""Stage 64 (32-node / 40L, ZERO precision compromise):
  opt63 stack
  + fast _NativeGroupedOutputProjection.backward:
    permute(1, 0, 2).contiguous() once + preallocated out=dw[g] 2D torch.mm loop
    (single-NPU 13.751 -> 10.891 ms, -2.86 ms/call x 40 layers = ~-114 ms/step).
"""
from pathlib import Path
import runpy

import torch
import torch.distributed as dist

import hyper_parallel.components.modules.shared_compressed_dsa_attention as scda_mod

OPT63_PATH = Path("/xw-tree/full-v41/opt63-opt62-muon2d.py")


def _fast_grouped_backward(ctx, gradient: torch.Tensor):
    grouped, weight = ctx.saved_tensors
    input_gradient = weight_gradient = None
    if not (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]):
        return None, None
    gg = gradient.permute(1, 0, 2).contiguous()
    if ctx.needs_input_grad[0]:
        input_gradient = torch.bmm(gg, weight).transpose(0, 1)
    if ctx.needs_input_grad[1]:
        gr = grouped.permute(1, 0, 2).contiguous()
        G, H_out = gg.shape[0], gg.shape[2]
        H_in = gr.shape[2]
        weight_gradient = torch.empty((G, H_out, H_in), dtype=gradient.dtype, device=gradient.device)
        for g in range(G):
            torch.mm(gg[g].T, gr[g], out=weight_gradient[g])
    return input_gradient, weight_gradient


def install():
    runpy.run_path(str(OPT63_PATH))["install"]()

    scda_mod._NativeGroupedOutputProjection.backward = staticmethod(_fast_grouped_backward)

    if not dist.is_initialized() or dist.get_rank() == 0:
        print("OPT64_GROUPED_PROJ_BWD_INSTALLED=1", flush=True)
