"""Stage 63 (32-node / 40L, ZERO precision compromise):
  opt62 stack
  + unbatch large 3D Muon Newton-Schulz inputs (B<=4, min(M,N)>=4096) into 2D MatMulV3 calls
    (single-NPU [2,4608,5120] 21.794 -> 20.196 ms, ~32 ms/step across 20 groups).
"""
from functools import wraps
from pathlib import Path
import runpy

import torch
import torch.distributed as dist

import hyper_parallel.core.optimizer.muon as muon_mod

OPT62_PATH = Path("/xw-tree/full-v41/opt62-opt58-aux-mask.py")


def install():
    runpy.run_path(str(OPT62_PATH))["install"]()

    orig_zp = muon_mod.zeropower_via_newtonschulz5

    @wraps(orig_zp)
    def fast_zeropower_via_newtonschulz5(
        ns_inputs: torch.Tensor,
        steps: int,
        ns_variant: str = "asym5",
        epsilon: float = 1e-10,
        ns_coefficients=None,
    ) -> torch.Tensor:
        if (
            ns_inputs.is_npu
            and ns_inputs.ndim == 3
            and 1 < ns_inputs.shape[0] <= 4
            and min(ns_inputs.shape[-2], ns_inputs.shape[-1]) >= 4096
        ):
            outs = [
                orig_zp(
                    ns_inputs[i].contiguous(),
                    steps,
                    ns_variant=ns_variant,
                    epsilon=epsilon,
                    ns_coefficients=ns_coefficients,
                )
                for i in range(ns_inputs.shape[0])
            ]
            return torch.stack(outs, dim=0)
        return orig_zp(
            ns_inputs,
            steps,
            ns_variant=ns_variant,
            epsilon=epsilon,
            ns_coefficients=ns_coefficients,
        )

    muon_mod.zeropower_via_newtonschulz5 = fast_zeropower_via_newtonschulz5

    if not dist.is_initialized() or dist.get_rank() == 0:
        print("OPT63_MUON2D_INSTALLED=1", flush=True)
