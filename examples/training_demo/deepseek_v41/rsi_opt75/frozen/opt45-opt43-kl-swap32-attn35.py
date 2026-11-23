"""Stage 45 (32-node / 40L, ZERO precision compromise): opt43 + Triton fused KL loss + swap_times=32 + MuonSwap buffer_cache reuse + layer 35 self_attn no-reshard."""
from functools import wraps
from pathlib import Path
import runpy

import torch
import triton
import triton.language as tl

import hyper_parallel.components.functional.sparse_lightning_indexer_kl_loss_grad as kl_mod
import hyper_parallel.core.optimizer.swap_muon as swap_muon_mod
import hyper_parallel.core.optimizer.swap_optimizer_base as swap_base_mod
import hyper_parallel.trainer.base as trainer_base_mod

OPT43_PATH = Path("/xw-tree/full-v41/opt43-opt39-tail2d-optprof.py")


@triton.jit
def _fused_kl_loss_kernel(
    pred_ptr, idx_ptr, scores_ptr, tgt_ptr, out_ptr,
    Q, K: tl.constexpr, tiny_val,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    ks = tl.arange(0, K)
    acc = tl.zeros((), dtype=tl.float32)
    for q_idx in range(pid, Q, num_progs):
        off = q_idx * K + ks
        valid = tl.load(idx_ptr + off) >= 0
        pred = tl.load(pred_ptr + off)
        tgt = tl.load(tgt_ptr + off)
        bad = tl.max(tl.where(valid & (pred <= 0.0), 1, 0), axis=0) > 0
        if bad:
            sc = tl.where(valid, tl.load(scores_ptr + off), -1.0e9)
            m = tl.max(sc, axis=0)
            lse = m + tl.log(tl.sum(tl.exp(sc - m), axis=0))
            log_p = tl.where(valid, sc - lse, 0.0)
        else:
            log_p = tl.log(tl.where(valid, pred, 1.0))
        log_t = tl.log(tl.maximum(tgt, tiny_val))
        acc += tl.sum(tgt * (log_t - log_p), axis=0)
    tl.store(out_ptr + pid, acc)


def install():
    runpy.run_path(str(OPT43_PATH))["install"]()

    # 1. Triton fused KL loss (17 kernels -> 1 kernel)
    kl_cls = kl_mod._SparseLightningIndexerKLLoss
    orig_kl_fwd = kl_cls.forward

    @staticmethod
    def fast_kl_fwd(
        ctx, query, key, weights, attention_query, main_key, indices, sinks,
        scale, coefficient, chunk_size, ratio, residual, selected_scores=None,
    ):
        if selected_scores is None or indices.shape[-1] != 512:
            return orig_kl_fwd(
                ctx, query, key, weights, attention_query, main_key, indices, sinks,
                scale, coefficient, chunk_size, ratio, residual, selected_scores=selected_scores,
            )
        batch, length, heads, dim = query.shape
        kl_grad, _ = kl_mod._load_kl_ops()
        with torch.no_grad(), torch.autocast(device_type=query.device.type, enabled=False):
            target = kl_mod.causal_attention_teacher(
                attention_query, main_key, indices, sinks, scale, ratio, residual, chunk_size,
            )
            options, metadata = kl_mod._kl_metadata(
                query.device, batch, length, heads, dim, key.shape[1], indices.shape[-1], ratio, residual,
            )
            native_key = key.contiguous().view(batch, key.shape[1], 1, dim)
            dq, dk, dw, prediction = kl_grad(
                query.contiguous(), native_key, weights.float().contiguous(),
                indices.unsqueeze(2).int().contiguous(), target.unsqueeze(2), metadata=metadata, **options,
            )
            prediction = prediction[:, :, 0]
            Q = prediction.numel() // 512
            partial_out = torch.empty((48,), device=prediction.device, dtype=torch.float32)
            _fused_kl_loss_kernel[(48,)](
                prediction.contiguous(), indices.contiguous(), selected_scores.contiguous(), target.contiguous(),
                partial_out, Q, 512, float(torch.finfo(torch.float32).tiny),
            )
            loss = partial_out.sum()
            ctx.save_for_backward(dq, dk.squeeze(2), dw)
            ctx.factor = coefficient / (batch * length)
        return loss * ctx.factor

    kl_cls.forward = fast_kl_fwd

    # 2. SwapOptimizerConfig.swap_times = 32 (cuts exposed 1/K H2D+D2H in half)
    orig_swap_init = swap_base_mod.SwapOptimizer.__init__
    logged = {"done": False}

    @wraps(orig_swap_init)
    def fast_swap_init(self, optimizer, config, *args, **kwargs):
        object.__setattr__(config, "swap_times", 32)
        if not logged["done"]:
            logged["done"] = True
            print("OPT45_SWAP_TIMES=32", flush=True)
        return orig_swap_init(self, optimizer, config, *args, **kwargs)

    swap_base_mod.SwapOptimizer.__init__ = fast_swap_init

    # 3. Reuse buffer_cache across units within MuonSwapAdapter.step_batch
    def fast_step_batch(self, batch, step_context):
        broadcaster = step_context["broadcaster"]
        buf_cache = step_context.setdefault("buffer_cache", {})
        for unit in batch:
            kind = unit.kind
            group = self.optimizer.param_groups[unit.group_index]
            ns_inputs = self.optimizer._update_muon_momentum(group, unit.params)
            if kind == swap_muon_mod._NO_COMM:
                self._apply_unshard_update(group, ns_inputs)
            else:
                hsdp_assign = unit.hsdp_assign
                if hsdp_assign.is_shard:
                    self.optimizer._process_shard_params(
                        group, ns_inputs, [hsdp_assign], unit.group_index,
                        buffer_cache=buf_cache,
                    )
                else:
                    self._apply_unshard_update(group, ns_inputs)
            if self.optimizer.post_update_fn is not None and unit.post_update_params:
                self.optimizer._run_post_update_fn(group, unit.post_update_params)
            if unit.flushes_broadcast:
                broadcaster.flush_group(unit.hsdp_assign)

    swap_muon_mod.MuonSwapAdapter.step_batch = fast_step_batch

    # 4. layer 35.self_attn no-reshard (tiny HBM, saves AllGather in backward)
    orig_fwd_bwd = trainer_base_mod.BaseTrainer.forward_backward_step
    nr_state = {"done": False}

    @wraps(orig_fwd_bwd)
    def nr35_attn_fwd_bwd(self, *args, **kwargs):
        if not nr_state["done"]:
            nr_state["done"] = True
            inner = getattr(self.model, "model", self.model)
            layers = getattr(inner, "layers", None)
            if layers is not None:
                attn = getattr(layers[35], "self_attn", None)
                if attn is not None and hasattr(attn, "set_reshard_after_forward"):
                    attn.set_reshard_after_forward(False, recurse=True)
                    if torch.distributed.get_rank() == 0:
                        print("OPT45_LAYER35_ATTN_NO_RESHARD=1", flush=True)
        return orig_fwd_bwd(self, *args, **kwargs)

    trainer_base_mod.BaseTrainer.forward_backward_step = nr35_attn_fwd_bwd
