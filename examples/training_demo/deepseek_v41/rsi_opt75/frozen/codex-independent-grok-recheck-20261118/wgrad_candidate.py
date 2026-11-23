"""Benchmark-only patch for the tested EIO expert w1 shapes."""
import torch


def install(source):
    original = source._write_weight_grad

    def write_weight_grad(output, inputs, gradient, output_first):
        if (
            not output_first and output.device.type == "npu" and output.dtype == torch.bfloat16
            and inputs.dtype == torch.bfloat16 and gradient.dtype == torch.bfloat16
            and output.shape == (5120, 4608)
            and inputs.shape[0] in (16384, 32768, 49152)
            and inputs.is_contiguous() and gradient.is_contiguous() and output.is_contiguous()
        ):
            output.copy_(torch.mm(gradient.T, inputs).T)
        else:
            original(output, inputs, gradient, output_first)

    source._write_weight_grad = write_weight_grad

    def restore():
        source._write_weight_grad = original

    return restore
