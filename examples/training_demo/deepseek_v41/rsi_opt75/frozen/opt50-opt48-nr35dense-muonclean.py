"""Stage 50 (32-node / 40L, ZERO precision compromise): opt48 + layer 35 dense-only no-reshard (recurse=False) + skip muon shard_pgs_rank debug loop."""
from functools import wraps
from pathlib import Path
import runpy

import torch

import hyper_parallel.core.optimizer.muon as muon_mod
import hyper_parallel.trainer.base as trainer_base_mod

OPT48_PATH = Path("/xw-tree/full-v41/opt48-opt47-vlm-bgprefetch.py")


def install():
    runpy.run_path(str(OPT48_PATH))["install"]()

    # 1. Disable reshard_after_forward on dense-only HSDP module for layer 35 (recurse=False)
    orig_fwd_bwd = trainer_base_mod.BaseTrainer.forward_backward_step
    nr_state = {"done": False}

    @wraps(orig_fwd_bwd)
    def nr35_dense_fwd_bwd(self, *args, **kwargs):
        if not nr_state["done"]:
            nr_state["done"] = True
            inner = getattr(self.model, "model", self.model)
            layers = getattr(inner, "layers", None)
            if layers is not None and hasattr(layers[35], "set_reshard_after_forward"):
                layers[35].set_reshard_after_forward(False, recurse=False)
                if torch.distributed.get_rank() == 0:
                    print("OPT50_DENSE_NO_RESHARD_35=1", flush=True)
        return orig_fwd_bwd(self, *args, **kwargs)

    trainer_base_mod.BaseTrainer.forward_backward_step = nr35_dense_fwd_bwd

    # 2. Skip unused shard_pgs_rank debug loop in Muon._process_shard_params
    def fast_process_shard_params(
        self, group, param_to_ns_input, hsdp_assignments, group_idx, buffer_cache=None,
    ):
        shard_compute_coord = self._shard_compute_coord.get(group_idx, {})
        for hsdp_assign in hsdp_assignments:
            owned_params = hsdp_assign.owned_params
            if not owned_params:
                continue
            shard_sizes, _, shard_pgs, total_shard_size = self._get_shard_info(hsdp_assign)
            safe_batches = self._split_into_memory_safe_batches(
                owned_params, shard_size=total_shard_size, min_batch_size=total_shard_size
            )
            for sub_batch in safe_batches:
                my_updates, param_compute_coord = self._gather_and_compute_shard_updates(
                    sub_batch, param_to_ns_input, hsdp_assign,
                    shard_compute_coord, group, buffer_cache,
                )
                self._fused_broadcast_and_apply(
                    sub_batch, my_updates, param_compute_coord,
                    group, hsdp_assign,
                )

    muon_mod.Muon._process_shard_params = fast_process_shard_params
