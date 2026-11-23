"""Preserve the validated native reduction schedule without gate observers."""

from functools import wraps

from hyper_parallel.core.multicore.modules.mega_mhc import function as forward_module
from hyper_parallel.core.multicore.modules.mega_mhc_grad import function as backward_module


def install():
    """Select the same single-tile native backward used by the paired gate."""
    original = backward_module.hyper_mega_mhc_grad
    if forward_module.hyper_mega_mhc_grad is not original:
        raise ValueError("Unexpected native mHC backward binding")

    @wraps(original)
    def ordered(*args, **kwargs):
        residual = args[6] if len(args) > 6 else kwargs["residual"]
        kwargs["token_tile"] = residual.numel() // (4 * residual.shape[-1])
        return original(*args, **kwargs)

    backward_module.hyper_mega_mhc_grad = ordered
    forward_module.hyper_mega_mhc_grad = ordered
    print("FULL_V41_MHC_REDUCTION_MODE=single_native_scheduler_tile gate_observers=0", flush=True)
