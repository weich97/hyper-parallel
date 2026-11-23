"""Isolated NPU candidate for the existing first-order chunked LM loss."""
import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F


def install(source, *, skip_empty=False):
    """Install into a benchmark process and return a function to undo the patch."""
    original = source._PrecomputedChunkLoss

    class ExplicitChunkLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, weight, targets, chunk_size, ignore_index):
            if not (
                hidden.device.type == "npu" and hidden.dtype == torch.bfloat16
                and hidden.size(0) == 1 and hidden.is_contiguous() and weight.is_contiguous()
                and chunk_size == 1024
            ):
                return original.forward(ctx, hidden, weight, targets, chunk_size, ignore_index)

            grad_hidden = torch.empty_like(hidden)
            grad_weight = None
            loss_sum = torch.zeros((), dtype=torch.float32, device=hidden.device)
            hidden_chunks = hidden[0].split(chunk_size)
            target_chunks = targets[0].split(chunk_size)
            grad_chunks = grad_hidden[0].split(chunk_size)
            if skip_empty:
                # One D2H sync for the whole sequence. Conditional experiment only.
                if targets.size(1) % chunk_size == 0:
                    active = targets.reshape(-1, chunk_size).ne(ignore_index).any(dim=1).tolist()
                else:
                    active = torch.stack([t.ne(ignore_index).any() for t in target_chunks]).tolist()
            else:
                active = [True] * len(hidden_chunks)

            for x, labels, dx, keep in zip(hidden_chunks, target_chunks, grad_chunks, active, strict=True):
                if not keep:
                    dx.zero_()
                    continue
                projected = F.linear(x, weight)
                with torch.enable_grad():
                    logits = projected.float().detach().requires_grad_(True)
                    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction="sum")
                    (dlogits,) = torch.autograd.grad(loss, logits)
                dy = dlogits.to(hidden.dtype)
                dx.copy_(dy @ weight)
                dw = dy.t() @ x
                if grad_weight is None:
                    grad_weight = dw
                else:
                    grad_weight.add_(dw)
                loss_sum.add_(loss.detach())
                del projected, logits, loss, dlogits, dy, dw
            if grad_weight is None:
                grad_weight = torch.zeros_like(weight)
            ctx.save_for_backward(grad_hidden, grad_weight)
            return loss_sum

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_loss):
            if grad_loss is None:
                return None, None, None, None, None
            dh, dw = ctx.saved_tensors
            return dh * grad_loss, dw * grad_loss, None, None, None

    source._PrecomputedChunkLoss = ExplicitChunkLoss

    def restore():
        source._PrecomputedChunkLoss = original

    return restore
