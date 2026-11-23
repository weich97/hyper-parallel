"""Stage 71 (32-node / 40L, ZERO precision compromise):
  opt70 stack (19.5565s / 26.0178% MFU, peak 19.5081s / 26.0823% MFU, max_alloc=50.0677 GB)
  + 1. Eliminate the 10 Rank-0 torch.npu.synchronize() pipeline stalls on Step 4 in opt20 & opt43
       (which slowed Step 4 by ~72-92 ms: 19.6003s vs 19.5287s/19.5081s). Replace with non-blocking
       torch.npu.Event(enable_timing=True) recording across fwd_bwd, prepare_clip, and optimizer,
       synchronizing once at the end of Step 4.
  + 2. Eliminate the last remaining _checkpoint(_chunk_objective) in DeepseekV41DSpark.forward via
       _FastDSparkVocabCEFn (computes logits, chunk_hits, FP32 cross_entropy, and exact BF16
       d_normed / d_markov_embed / d_markov_head_w in a single pass, freeing [N, 129280] logits
       immediately instead of recomputing markov_head + norm + linear + cross_entropy + confidence_head
       in backward; single-NPU 129.7 ms -> ~91 ms, loss_diff=0.0).
"""
from functools import wraps
from pathlib import Path
import runpy
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

import hyper_parallel.models.deepseek_v41.dspark as dspark_mod
from hyper_parallel.models.deepseek_v41.dspark import (
    DeepseekV41DSpark,
    IGNORE_INDEX,
    _collapse_streams,
    _document_geometry,
)
import hyper_parallel.trainer.base as trainer_base_mod

OPT70_PATH = Path("/xw-tree/full-v41/opt70-opt69-dspark-nockpt.py")


class _FastDSparkVocabCEFn(torch.autograd.Function):
    """Single-pass DSpark vocabulary linear + FP32 cross-entropy without saving [N, V] logits."""

    @staticmethod
    def forward(
        ctx,
        normed_flat: torch.Tensor,
        markov_embed_flat: torch.Tensor,
        head_weight: torch.Tensor,
        markov_head_weight: torch.Tensor,
        target_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = F.linear(normed_flat, head_weight) + F.linear(markov_embed_flat, markov_head_weight)
        chunk_mask = target_flat != IGNORE_INDEX
        chunk_hits = (logits.argmax(dim=-1) == target_flat) & chunk_mask
        logits_fp32 = logits.float()
        del logits
        with torch.enable_grad():
            logits_req = logits_fp32.detach().requires_grad_(True)
            del logits_fp32
            ce_sum = F.cross_entropy(
                logits_req,
                target_flat,
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            (d_logits_fp32,) = torch.autograd.grad(ce_sum, logits_req)
            del logits_req
        d_logits = d_logits_fp32.to(head_weight.dtype)
        del d_logits_fp32
        d_normed = d_logits @ head_weight
        d_me = d_logits @ markov_head_weight
        d_mhw = d_logits.t() @ markov_embed_flat
        del d_logits
        ctx.save_for_backward(d_normed, d_me, d_mhw)
        return ce_sum.detach(), chunk_hits

    @staticmethod
    def backward(ctx, grad_ce: torch.Tensor, _grad_hits: torch.Tensor):
        d_normed, d_me, d_mhw = ctx.saved_tensors
        g = grad_ce.to(d_normed.dtype)
        return d_normed * g, d_me * g, None, d_mhw * g, None


def install():
    runpy.run_path(str(OPT70_PATH))["install"]()

    # 1. Neutralize Step-4 Rank-0 mid-step torch.npu.synchronize() fences in opt20 & opt43
    #    without unwrapping any BaseTrainer/Muon/AdamW/observer hooks, and record non-blocking
    #    NPU Events across fwd_bwd, prepare_clip, and optimizer on Step 4.
    orig_fwd_bwd = trainer_base_mod.BaseTrainer.forward_backward_step
    orig_prep_clip = trainer_base_mod.BaseTrainer.prepare_optimizer_step
    orig_opt_step = trainer_base_mod.BaseTrainer.step_optimizers_and_schedulers
    orig_npu_sync = torch.npu.synchronize
    _skip_sync_files = (
        "opt20-observe-vec-rope-skip1-clip-muon.py",
        "opt43-opt39-tail2d-optprof.py",
    )

    def _guarded_sync(*args, **kwargs):
        caller_file = sys._getframe(1).f_code.co_filename
        if caller_file.endswith(_skip_sync_files):
            return None
        return orig_npu_sync(*args, **kwargs)

    step4_prof = {"step": 0, "ev0": None, "ev1": None, "ev2": None}

    @wraps(orig_fwd_bwd)
    def async_fwd_bwd(self, *args, **kwargs):
        step4_prof["step"] += 1
        if step4_prof["step"] == 4 and dist.is_initialized() and dist.get_rank() == 0:
            torch.npu.synchronize = _guarded_sync
            ev0 = torch.npu.Event(enable_timing=True)
            ev1 = torch.npu.Event(enable_timing=True)
            step4_prof["ev0"] = ev0
            step4_prof["ev1"] = ev1
            ev0.record()
            res = orig_fwd_bwd(self, *args, **kwargs)
            ev1.record()
            return res
        return orig_fwd_bwd(self, *args, **kwargs)

    @wraps(orig_prep_clip)
    def async_prep_clip(self, *args, **kwargs):
        if step4_prof["step"] == 4 and dist.is_initialized() and dist.get_rank() == 0:
            res = orig_prep_clip(self, *args, **kwargs)
            ev2 = torch.npu.Event(enable_timing=True)
            step4_prof["ev2"] = ev2
            ev2.record()
            return res
        return orig_prep_clip(self, *args, **kwargs)

    @wraps(orig_opt_step)
    def async_opt_step(self, *args, **kwargs):
        if step4_prof["step"] == 4 and dist.is_initialized() and dist.get_rank() == 0:
            try:
                res = orig_opt_step(self, *args, **kwargs)
                ev3 = torch.npu.Event(enable_timing=True)
                ev3.record()
                orig_npu_sync()
                t_fwd = step4_prof["ev0"].elapsed_time(step4_prof["ev1"]) / 1000.0
                t_clip = step4_prof["ev1"].elapsed_time(step4_prof["ev2"]) / 1000.0
                t_opt = step4_prof["ev2"].elapsed_time(ev3) / 1000.0
                print(
                    f"OPT71_STEP4_PHASE_BREAKDOWN fwd_bwd={t_fwd:.4f}s prepare_clip={t_clip:.4f}s "
                    f"optimizer={t_opt:.4f}s sum={t_fwd + t_clip + t_opt:.4f}s",
                    flush=True,
                )
                return res
            finally:
                torch.npu.synchronize = orig_npu_sync
        return orig_opt_step(self, *args, **kwargs)

    trainer_base_mod.BaseTrainer.forward_backward_step = async_fwd_bwd
    trainer_base_mod.BaseTrainer.prepare_optimizer_step = async_prep_clip
    trainer_base_mod.BaseTrainer.step_optimizers_and_schedulers = async_opt_step

    # 2. Single-pass DSpark chunk objective without _checkpoint(_chunk_objective) recomputation
    #    Inject inside 30d1b883's `sampled` wrapper closure so `30d1b883` still records anchor sampling.
    sampled_dspark_forward = DeepseekV41DSpark.forward
    orig_dspark_forward = getattr(sampled_dspark_forward, "__wrapped__", sampled_dspark_forward)

    @wraps(orig_dspark_forward)
    def fast_dspark_forward(
        self,
        target_hiddens,
        input_ids,
        labels,
        inputs_embeds,
        lm_head,
        *,
        noise_embedding,
        token_mask=None,
        sequence_boundaries=None,
    ):
        if not (torch.is_grad_enabled() and self.training):
            return orig_dspark_forward(
                self,
                target_hiddens,
                input_ids,
                labels,
                inputs_embeds,
                lm_head,
                noise_embedding=noise_embedding,
                token_mask=token_mask,
                sequence_boundaries=sequence_boundaries,
            )

        self._validate_backbone_inputs(target_hiddens, input_ids, inputs_embeds, noise_embedding)
        self._prepare_megamoe_runtime()
        batch, seq = input_ids.shape
        block = self.block_size
        device = input_ids.device
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("DSpark labels must match input_ids")
        if token_mask is not None and (token_mask.shape != input_ids.shape or token_mask.dtype != torch.bool):
            raise ValueError("DSpark token_mask must be a boolean mask matching input_ids")
        _, document_ends = _document_geometry(seq, sequence_boundaries)
        markov_inputs, targets, full_supervised, selection, anchor_indices = self._select_anchor_targets(
            input_ids, labels, token_mask, sequence_boundaries
        )
        anchor_positions = None if selection is None else selection.positions
        anchors = anchor_indices.numel()
        main_states = self.main_norm(self.main_proj(torch.cat(target_hiddens, dim=-1)))

        token_embeddings = inputs_embeds
        noise = noise_embedding.to(token_embeddings.dtype)
        draft = torch.empty(
            batch, anchors, block, token_embeddings.shape[-1],
            dtype=token_embeddings.dtype, device=device,
        )
        first_embeddings = torch.cat((token_embeddings[:, 1:], noise.expand(batch, 1, -1)), dim=1)
        if selection is not None:
            first_embeddings = first_embeddings.index_select(1, anchor_indices)
        next_is_in_document = (
            anchor_indices + 1 < torch.tensor(document_ends, device=device).index_select(0, anchor_indices)
        )
        draft[:, :, 0] = torch.where(next_is_in_document[None, :, None], first_embeddings, noise)
        draft[:, :, 1:] = noise
        streams = draft.unsqueeze(3).expand(-1, -1, -1, self.hc_mult, -1).contiguous()
        del draft, first_embeddings
        pre_mix = streams.new_zeros(batch, anchors * block, self.hc_mult, dtype=torch.float32)
        pre_mix[:, :, 0] = 1.0
        rope_cos, rope_sin = self._rope_tables(seq, device, token_embeddings.dtype)
        draft_ids = input_ids.new_full((batch, anchors, block), self.noise_token_id)
        first_ids = F.pad(input_ids[:, 1:], (0, 1), value=self.noise_token_id)
        if selection is not None:
            first_ids = first_ids.index_select(1, anchor_indices)
        draft_ids[:, :, 0] = torch.where(next_is_in_document, first_ids, self.noise_token_id)
        draft_ids = draft_ids.reshape(batch, anchors * block)

        streams, pre_mix = self._run_stages(
            streams, pre_mix, main_states, (rope_cos, rope_sin),
            draft_ids, sequence_boundaries, anchor_positions,
        )
        hidden_out = _collapse_streams(streams, pre_mix)

        raw_head_weight = lm_head.weight
        if hasattr(raw_head_weight, "full_tensor"):
            raw_head_weight = raw_head_weight.full_tensor()
        head_weight = raw_head_weight.detach()
        markov_head_weight = self.markov_head.head.weight
        if hasattr(markov_head_weight, "full_tensor"):
            markov_head_weight = markov_head_weight.full_tensor()

        chunk = self.loss_chunk_size
        ce_total = hidden_out.new_zeros((), dtype=torch.float32)
        bce_total = hidden_out.new_zeros((), dtype=torch.float32)
        hit_total = hidden_out.new_zeros((), dtype=torch.float32)
        hidden_dim = hidden_out.shape[-1]
        markov_rank = markov_head_weight.shape[1]

        for start in range(0, anchors, chunk):
            stop = min(start + chunk, anchors)
            hidden_chunk = hidden_out[:, start:stop]
            markov_chunk = markov_inputs[:, start:stop]
            target_chunk = targets[:, start:stop]

            markov_embed = self.markov_head.embed(markov_chunk)
            normed_hidden = self.norm(hidden_chunk)
            ce_sum, chunk_hits_flat = _FastDSparkVocabCEFn.apply(
                normed_hidden.reshape(-1, hidden_dim),
                markov_embed.reshape(-1, markov_rank),
                head_weight,
                markov_head_weight,
                target_chunk.reshape(-1),
            )
            chunk_mask = target_chunk != IGNORE_INDEX
            chunk_hits = chunk_hits_flat.view_as(target_chunk)
            confidence = self.confidence_head(hidden_chunk, markov_embed)
            bce = F.binary_cross_entropy_with_logits(
                confidence, chunk_hits.float(), reduction="none",
            )
            ce_total = ce_total + ce_sum
            bce_total = bce_total + (bce * chunk_mask).sum()
            hit_total = hit_total + chunk_hits.sum().float()

        supervised = (targets != IGNORE_INDEX).sum()
        denominator = full_supervised.clamp(min=1).float()
        weight = 1.0 if selection is None else selection.weight
        draft_ce = ce_total * weight / denominator
        confidence_bce = bce_total * weight / denominator
        loss = draft_ce + self.confidence_coeff * confidence_bce
        accuracy = (hit_total / supervised.clamp(min=1)).detach()
        metrics = {
            "dspark_ce": draft_ce.detach(),
            "dspark_confidence_bce": confidence_bce.detach(),
            "dspark_draft_accuracy": accuracy,
            "dspark_supervised_tokens": supervised.detach(),
        }
        if selection is not None:
            metrics["dspark_full_supervised_tokens"] = full_supervised.detach()
            metrics["dspark_sampled_anchors"] = torch.tensor(selection.valid_count, device=device)
            metrics["dspark_population_anchors"] = torch.tensor(selection.population, device=device)
        return loss, metrics

    replaced_in_closure = False
    if hasattr(sampled_dspark_forward, "__wrapped__") and sampled_dspark_forward.__closure__:
        for cell in sampled_dspark_forward.__closure__:
            if cell.cell_contents is orig_dspark_forward:
                cell.cell_contents = fast_dspark_forward
                replaced_in_closure = True
                break
    if not replaced_in_closure:
        DeepseekV41DSpark.forward = fast_dspark_forward

    if not dist.is_initialized() or dist.get_rank() == 0:
        print("OPT71_DSPARK_CE_STEP4ASYNC_INSTALLED=1", flush=True)
