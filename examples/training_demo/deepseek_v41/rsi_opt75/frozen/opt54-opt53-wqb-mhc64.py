"""Stage 54 (32-node / 40L, ZERO precision compromise): opt53 + wq_b contiguous-T grad_weight (-85ms/step) + HyperMegaMhc token_tile=64."""
from pathlib import Path
import runpy

import torch
import torch.nn as nn

import hyper_parallel.models.deepseek_v41.adapter.distributed.mega_mhc as v41_mhc_mod

OPT53_PATH = Path("/xw-tree/full-v41/opt53-opt52-megamoe-noclone.py")


class _FastWqBLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return torch.matmul(x, weight.t())

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        x_2d = x.reshape(-1, x.shape[-1])
        go_2d = grad_output.reshape(-1, grad_output.shape[-1])
        grad_x = grad_w = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.matmul(go_2d, weight).reshape_as(x)
        if ctx.needs_input_grad[1]:
            # [1280, 16384] contiguous @ [16384, 32768] -> .t() avoids mac=0.39 transpose_a=True kernel
            grad_w = torch.matmul(x_2d.t().contiguous(), go_2d).t()
        return grad_x, grad_w


def install():
    runpy.run_path(str(OPT53_PATH))["install"]()

    # 1. Patch nn.Linear.forward for [16384, 1280] -> [16384, 32768] (q_b_proj)
    orig_linear_forward = nn.Linear.forward

    def fast_linear_forward(self, input):
        if (
            self.bias is None
            and input.dtype == torch.bfloat16
            and input.shape[-1] == 1280
            and input.numel() == 16384 * 1280
            and self.weight.shape == (32768, 1280)
            and type(self.weight) is torch.Tensor
            and torch.is_grad_enabled()
        ):
            return _FastWqBLinearFn.apply(input, self.weight)
        return orig_linear_forward(self, input)

    nn.Linear.forward = fast_linear_forward

    # 2. Bump DeepseekV41HyperMegaMhc default token_tile 32 -> 64
    orig_mhc_init = v41_mhc_mod.DeepseekV41HyperMegaMhc.__init__

    def fast_mhc_init(self, *args, token_tile=64, **kwargs):
        if token_tile == 32:
            token_tile = 64
        return orig_mhc_init(self, *args, token_tile=token_tile, **kwargs)

    v41_mhc_mod.DeepseekV41HyperMegaMhc.__init__ = fast_mhc_init
