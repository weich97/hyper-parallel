"""Stage 53 (32-node / 40L, ZERO precision compromise): opt52 + skip dead execution.combine.clone() in MegaMoe forward when not any(ctx.needs_input_grad) and ctx.has_unpermute."""
from pathlib import Path
import runpy

import hyper_parallel.core.multicore.modules.mega_moe.function as mega_moe_fn_mod

OPT52_PATH = Path("/xw-tree/full-v41/opt52-opt50-swap96.py")


def install():
    runpy.run_path(str(OPT52_PATH))["install"]()

    _prepare_forward_execution = mega_moe_fn_mod._prepare_forward_execution
    _launch_forward_kernel = mega_moe_fn_mod._launch_forward_kernel
    _save_forward_state = mega_moe_fn_mod._save_forward_state
    load_torch_npu = mega_moe_fn_mod.load_torch_npu

    @staticmethod
    def fast_megamoe_forward_noclone(
        ctx, routed_tokens, weight1, weight2, route, plan, workspace, permutation, topk_weights, workspace_claimed
    ):
        spec = plan.spec
        ctx.weight_grad_offsets = None
        if spec.use_matmul_weight_grad:
            if route.expert_end_offsets is None or len(route.expert_end_offsets) != spec.local_experts:
                raise ValueError("Separate weight gradients require exact Host expert end offsets")
            ctx.weight_grad_offsets = route.expert_end_offsets
        if topk_weights is not None and (spec.dispatch_mode != "pull" or permutation is None):
            raise ValueError("Integrated output unpermutation requires pull dispatch and an input permutation.")
        metadata = route
        permutation_inputs = ()
        if permutation is not None:
            routed_tokens, _, unpermute_mapping = permutation
            if ctx.needs_input_grad[0] or topk_weights is not None:
                permutation_inputs = (unpermute_mapping,)
        ctx.source_rows = routed_tokens.shape[0]
        ctx.token_rows = ctx.source_rows // spec.top_k
        ctx.has_permutation = permutation is not None
        ctx.has_unpermute = topk_weights is not None
        if workspace_claimed:
            if not workspace.in_use:
                raise RuntimeError("MegaMoe forward requires the caller-held workspace lease.")
        else:
            if not spec.auto_tokens:
                workspace.ensure(spec, routed_tokens.dtype, routed_tokens.device)
            workspace.claim()
        try:
            capacity = metadata.expert_capacity
            execution = _prepare_forward_execution(workspace, plan, routed_tokens, capacity)
            _launch_forward_kernel(plan, metadata, execution.source, weight1, weight2, execution)
            need_grad = any(ctx.needs_input_grad)
            combine_slice = execution.combine[:ctx.source_rows]
            # When not need_grad and ctx.has_unpermute, npu_moe_token_unpermute consumes combine_slice
            # before workspace.release() and returns a newly allocated tensor, so .clone() is dead!
            output = combine_slice.clone() if (need_grad or not ctx.has_unpermute) else combine_slice
            if spec.dispatch_mode == "pull":
                saved_dispatch = execution.dispatch
            else:
                if need_grad:
                    execution.intermediates.down_proj.copy_(execution.dispatch[:capacity])
                saved_dispatch = execution.intermediates.down_proj
            if ctx.has_unpermute:
                probs = topk_weights.float().contiguous()
                ctx.topk_dtype = topk_weights.dtype
                if need_grad:
                    permutation_inputs += (output, probs)
                output = (
                    load_torch_npu().npu_moe_token_unpermute(output, unpermute_mapping, probs=probs)
                    if ctx.source_rows
                    else output[:0]
                )
            if need_grad:
                _save_forward_state(
                    ctx,
                    plan,
                    workspace,
                    saved_dispatch,
                    execution.intermediates.up_proj,
                    execution.intermediates.activation,
                    weight1,
                    weight2,
                    metadata,
                    permutation_inputs,
                )
            return output
        finally:
            if not workspace_claimed:
                workspace.release()

    mega_moe_fn_mod._MegaMoeFunction.forward = fast_megamoe_forward_noclone
