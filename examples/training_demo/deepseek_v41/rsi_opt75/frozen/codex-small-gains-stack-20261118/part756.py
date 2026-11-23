"""Qualified one-digit-ms layout candidates for the stack."""
from hashlib import sha256
import importlib
from pathlib import Path
from types import CodeType

import torch

ROOT = Path(__file__).resolve().parent


def single_nested_code(code, name):
    for item in code.co_consts:
        if isinstance(item, CodeType) and item.co_name == name:
            return item
    raise RuntimeError(f"Missing code block: {name}")


def install():
    kl = importlib.import_module("hyper_parallel.components.functional.sparse_lightning_indexer_kl_loss_grad")
    teacher = kl.compressed_attention_teacher
    old = teacher.__code__
    if old.co_name != "fast_compressed_attention_teacher" or not old.co_filename.endswith("/opt26-teacher-chunk1k.py"):
        raise RuntimeError(f"Unexpected teacher entry: {old.co_name}, {old.co_filename}")
    original = Path(old.co_filename)
    if sha256(original.read_bytes()).hexdigest() != sha256((ROOT / "teacher-parent.py").read_bytes()).hexdigest():
        raise RuntimeError("Teacher source changed during stack review")
    path = ROOT / "teacher-left-final.py"
    updated_source = path.read_bytes()
    if sha256(updated_source).hexdigest() != "3f74c09edc11a4aa3fda3e76ff73b085056d27edc0e76c1210f6d2fe6ff448b7":
        raise RuntimeError("Teacher candidate checksum changed")
    updated = single_nested_code(single_nested_code(compile(updated_source, str(path), "exec"), "install"),
                                 old.co_name)
    if (updated.co_freevars, updated.co_argcount) != (old.co_freevars, old.co_argcount):
        raise RuntimeError("Teacher closure changed")
    # Retain the existing function object since causal teacher captures it in a closure.
    teacher.__code__ = updated

    fn = importlib.import_module("hyper_parallel.core.multicore.modules.mega_moe.function")
    previous = fn._write_weight_grad

    reported = False

    def write(output, inputs, gradient, output_first):
        nonlocal reported
        if (not output_first and output.is_npu and output.dtype == inputs.dtype == gradient.dtype == torch.bfloat16
                and output.shape == (2304, 5120) and inputs.shape == (32768, 2304)
                and gradient.shape == (32768, 5120)
                and output.is_contiguous() and inputs.is_contiguous() and gradient.is_contiguous()
                and not torch.is_grad_enabled()):
            torch.mm(inputs.T, gradient, out=output)
            if not reported:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    print("STACK_W2_OUT_ACTIVE=1", flush=True)
                reported = True
        else:
            previous(output, inputs, gradient, output_first)

    fn._write_weight_grad = write
    print("STACK_TINY_LAYOUTS_INSTALLED=1", flush=True)
