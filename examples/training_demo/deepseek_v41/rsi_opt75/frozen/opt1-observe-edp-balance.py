"""Stage 1: Alternating 2+1 / 1+2 EDP expert sharding to balance Muon NS across superpods."""
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import runpy

import torch
import torch.distributed as dist
import torch.nn as nn

from hyper_parallel.core.fully_shard import hsdp_param as hsdp_param_mod
from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2, ShardedState
from hyper_parallel.distributed._builder import fsdp_adapter as fsdp_adapter_mod


def _is_flipped_edp_param(self) -> bool:
    return bool(getattr(self, '_dsv41_flipped_edp', False))


def install_edp_balance_hooks():
    if getattr(HSDPParamV2, '_dsv41_edp_balance_installed', False):
        return
    HSDPParamV2._dsv41_edp_balance_installed = True

    expert_unit_counter = [0]
    orig_fully_shard = fsdp_adapter_mod.fully_shard

    @wraps(orig_fully_shard)
    def wrapped_fully_shard(module, *args, **kwargs):
        mesh = kwargs.get('mesh')
        dim_names = getattr(mesh, 'mesh_dim_names', None) or ()
        if 'edp_shard' in dim_names:
            idx = expert_unit_counter[0]
            expert_unit_counter[0] = idx + 1
            if idx % 2 == 1:
                setattr(module, '_dsv41_flip_edp_shard', True)
        return orig_fully_shard(module, *args, **kwargs)

    fsdp_adapter_mod.fully_shard = wrapped_fully_shard

    orig_init_sharded_param = HSDPParamV2._init_sharded_param
    orig_get_unsharded_param_data = HSDPParamV2._get_unsharded_param_data
    orig_init_unsharded_param = HSDPParamV2.init_unsharded_param
    orig_reduce_scatter_grad = HSDPParamV2.reduce_scatter_grad

    @wraps(orig_init_sharded_param)
    @torch.no_grad()
    def wrapped_init_sharded_param(self, param, shard_placement_fn):
        hsdp_placement = self._resolve_hsdp_placement(param, shard_placement_fn)
        param_data = param.to_local() if self._orig_param_is_dtensor else param
        shard_dim = hsdp_placement.dim
        shard_world_size = self.mesh_info.shard_mesh_size if isinstance(self.mesh_info, hsdp_param_mod.FSDPMeshInfo) else 1
        should_flip = (
            shard_world_size == 2
            and shard_dim == 0
            and param_data.ndim > 0
            and param_data.size(0) % 2 == 1
            and getattr(self._module_info.module, '_dsv41_flip_edp_shard', False)
        )
        if not should_flip:
            self._dsv41_flipped_edp = False
            return orig_init_sharded_param(self, param, shard_placement_fn)

        self._dsv41_flipped_edp = True
        self.hsdp_placement = hsdp_placement
        base_placements = list(self._get_base_spmd_placements())
        self._orig_size = param_data.size()
        self._contiguous_orig_stride = hsdp_param_mod.make_contiguous_strides_for(self._orig_size)
        self.shard_rank = self.mesh_info.shard_mesh_rank
        self.shard_world_size = 2
        if isinstance(self.mesh_info, hsdp_param_mod.DDPMeshInfo):
            self.replicate_world_size = self.mesh_info.replicate_mesh_size
        else:
            self.replicate_world_size = 1
        self.is_replicate_param = (
            isinstance(self.mesh_info, hsdp_param_mod.DDPMeshInfo)
            and not isinstance(self.mesh_info, HSDPMeshInfo)
            if hasattr(hsdp_param_mod, 'HSDPMeshInfo')
            else False
        )
        self._init_shard_placements(param_data, shard_dim, base_placements)

        split1 = (param_data.size(0) + 1) // 2
        split0 = param_data.size(0) - split1
        chunks = (param_data.narrow(0, 0, split0), param_data.narrow(0, split0, split1))
        local_shard = chunks[self.shard_rank]
        sharded_param = local_shard.clone().contiguous()
        self.sharded_size = sharded_param.size()
        self.contiguous_sharded_stride = hsdp_param_mod.make_contiguous_strides_for(self.sharded_size)
        self.padded_sharded_param_size = chunks[1].size()
        if self.offload_to_cpu and not sharded_param.is_meta:
            sharded_param = sharded_param.cpu()
            if self.pin_memory:
                sharded_param = sharded_param.pin_memory()

        if self.sharded_size == self.padded_sharded_param_size:
            self._sharded_param_data = sharded_param.view(-1)
        else:
            padded_sharded_param = sharded_param.new_zeros(self.padded_sharded_param_size)
            if self.pin_memory and not padded_sharded_param.is_meta:
                padded_sharded_param = padded_sharded_param.pin_memory()
            if sharded_param.numel() > 0:
                padded_sharded_param.narrow(shard_dim, 0, self.sharded_size[shard_dim]).copy_(sharded_param)
            self._sharded_param_data = padded_sharded_param.view(-1)
            sharded_param = padded_sharded_param.narrow(shard_dim, 0, self.sharded_size[shard_dim])

        self._sharding_spec = self._build_sharding_spec(param, param_data)
        self.sharded_param = nn.Parameter(self.to_sharded_dtensor(sharded_param))
        self.sharded_param._layout = self._sharding_spec
        self.sharded_param._placements = tuple(self._sharding_spec.placements)
        self.sharded_param.requires_grad_(param.requires_grad)
        self._setattr_on_modules(self.sharded_param)
        self.sharded_param._hsdp_param_initialized = True
        self.sharded_param._dsv41_flipped_edp_shard_rank = self.shard_rank
        self.sharded_state = ShardedState.SHARDED
        self.param_dtype = None
        self.reduce_dtype = None

    @wraps(orig_get_unsharded_param_data)
    def wrapped_get_unsharded_param_data(self, async_op: bool = False):
        if not _is_flipped_edp_param(self):
            return orig_get_unsharded_param_data(self, async_op=async_op)
        all_gather_input = self.all_gather_inputs[0]
        self.init_unsharded_param_buffers(
            all_gather_input_numels=[all_gather_input.numel()],
            all_gather_input_dtypes=[all_gather_input.dtype],
            world_size=self.shard_world_size,
            device=self.device,
        )
        self.alloc_unsharded_param_buffers()
        self.allgather_comm_ctx.allgather_output = torch.empty_like(self.unsharded_param_buffers[0])
        with torch.autograd._unsafe_preserve_version_counter(self.allgather_comm_ctx.allgather_output):
            self.allgather_comm_ctx.allgather_handle = dist.all_gather_into_tensor(
                self.allgather_comm_ctx.allgather_output,
                all_gather_input,
                group=self.mesh_info.shard_process_group,
                async_op=async_op,
            )

    @wraps(orig_init_unsharded_param)
    def wrapped_init_unsharded_param(self):
        if _is_flipped_edp_param(self) and self.allgather_comm_ctx.allgather_output is not None:
            split1 = self.padded_sharded_param_size[0]
            split0 = self._orig_size[0] - split1
            view_out = self.allgather_comm_ctx.allgather_output.view(2, split1, *self._orig_size[1:])
            chunks = (view_out[0].narrow(0, 0, split0), view_out[1])
            with torch.autograd._unsafe_preserve_version_counter(self.unsharded_param_buffers[0]):
                torch.cat(
                    chunks,
                    dim=0,
                    out=self.unsharded_param_buffers[0].narrow(0, 0, self._orig_size.numel()).view(self._orig_size),
                )
            self.allgather_comm_ctx.allgather_output.untyped_storage().resize_(0)
            self.allgather_comm_ctx.allgather_output = None
        return orig_init_unsharded_param(self)

    @wraps(orig_reduce_scatter_grad)
    def wrapped_reduce_scatter_grad(self, async_op: bool = True, reduce_op=dist.ReduceOp.AVG, output_buffer=None):
        if not _is_flipped_edp_param(self):
            return orig_reduce_scatter_grad(self, async_op=async_op, reduce_op=reduce_op, output_buffer=output_buffer)
        if self.unsharded_accumulated_grad is not None:
            grad = self.unsharded_accumulated_grad_data
            self.unsharded_accumulated_grad = None
        else:
            grad = self.unsharded_grad_data
            self.unsharded_param.grad = None
        grad = grad.to(self.reduce_comm_dtype(grad))
        split1 = self.padded_sharded_param_size[0]
        split0 = self._orig_size[0] - split1
        padded_grad = grad.new_zeros((2 * split1, *grad.shape[1:]))
        if split0 > 0:
            padded_grad.narrow(0, 0, split0).copy_(grad.narrow(0, 0, split0))
        padded_grad.narrow(0, split1, split1).copy_(grad.narrow(0, split0, split1))
        self._grad = padded_grad.view(-1)
        hsdp_param_mod.apply_gradient_scaling_factor(self._grad, self.gradient_scaling_factor)

        shard_process_group = self.mesh_info.shard_process_group
        output_numel = self._grad.numel() // 2
        if output_buffer is not None:
            self.reduce_scatter_comm_ctx.reduce_scatter_output = output_buffer
        else:
            self.reduce_scatter_comm_ctx.reduce_scatter_output = torch.empty(
                output_numel, dtype=self._grad.dtype, device=self._grad.device
            )
        self.reduce_scatter_comm_ctx.reduce_scatter_handle = dist.reduce_scatter_tensor(
            self.reduce_scatter_comm_ctx.reduce_scatter_output,
            self._grad,
            op=reduce_op,
            group=shard_process_group,
            async_op=async_op,
        )
        if not async_op:
            self._grad = None

    HSDPParamV2._init_sharded_param = wrapped_init_sharded_param
    HSDPParamV2._get_unsharded_param_data = wrapped_get_unsharded_param_data
    HSDPParamV2.init_unsharded_param = wrapped_init_unsharded_param
    HSDPParamV2.reduce_scatter_grad = wrapped_reduce_scatter_grad

    from hyper_parallel.core.utils import clip_grad as clip_grad_mod
    orig_norm_coverage = clip_grad_mod._norm_coverage

    @wraps(orig_norm_coverage)
    def wrapped_norm_coverage(param):
        try:
            return orig_norm_coverage(param)
        except ValueError as exc:
            if 'exact local parameter coverage' not in str(exc):
                raise
            from hyper_parallel.core.dtensor.layout import infer_slice_area_by_layout
            global_shape = tuple(param.shape)
            layout = getattr(param, 'layout', None)
            if layout is None:
                raise
            rank = dist.get_rank() if dist.is_initialized() else 0
            area = list(infer_slice_area_by_layout(layout, layout.rank_list.index(rank), global_shape))
            local = param._local_tensor if hasattr(param, '_local_tensor') else param
            local_len = int(local.shape[0])
            s0, e0 = int(area[0][0]), int(area[0][1])
            unflipped_len = e0 - s0
            if abs(local_len - unflipped_len) != 1:
                raise
            if local_len < unflipped_len:
                area[0] = (s0, s0 + local_len)
            else:
                area[0] = (e0 - local_len, e0)
            if tuple(end - start for start, end in area) != tuple(local.shape):
                raise
            return global_shape, tuple((int(start), int(end)) for start, end in area)

    clip_grad_mod._norm_coverage = wrapped_norm_coverage



def install():
    from hyper_parallel.trainer.vlm_trainer import VLMTrainer
    parent = Path('/xw-tree/full-v41/14050b13-observe-repaired-best.py')
    if hashlib.sha256(parent.read_bytes()).hexdigest() != '14050b13bcfdb062d617a3884195711982f34588a5f3e18806a69fe7b5a04d95':
        raise ValueError('Accepted observer changed')
    runpy.run_path(str(parent))['install']()
    install_edp_balance_hooks()

    begin = VLMTrainer.on_train_begin

    @wraps(begin)
    def started(trainer):
        result = begin(trainer)
        expert_dim0_sum = 0
        expert_param_count = 0
        for name, param in trainer.base.model.named_parameters():
            if 'experts' in name and param.ndim >= 3:
                local_t = param.to_local() if hasattr(param, 'to_local') else param
                expert_dim0_sum += int(local_t.size(0))
                expert_param_count += 1
        rank = int(os.environ.get('RANK', '0'))
        if rank in (0, 128, 255):
            print(
                f'EDP_BALANCE_CHECK rank={rank} expert_params={expert_param_count} '
                f'local_expert_slices_sum={expert_dim0_sum}',
                flush=True,
            )
        root = Path(os.environ['DSV41_FULL_RUN_DIR']) / 'edp-balance'
        root.mkdir(exist_ok=True)
        (root / f'rank-{rank:03d}.json').write_text(
            json.dumps(dict(rank=rank, expert_params=expert_param_count, local_expert_slices_sum=expert_dim0_sum)) + '\n'
        )
        return result

    VLMTrainer.on_train_begin = started
