# DeepSeek-V4.1-Flash 763B MFU 优化实验记录

- **目标**: 在 32 节点 × 8 die (256 die, Ascend 910C, GBS=256, MBS=1, SeqLen=16K) 下，将 PaLM MFU 从基线 `22.6152%` 提升至 `≥25%`。
- **计算口径**: Updates 9–30 (共 22 个 update) 平均 step time；`MFU = 22.4989 * 22.6152 / mean_step_time`。

## 基线 (Baseline)
- **Run ID**: `huangxs-gbs256-baseline-repeat30-v2`
- **配置**: `14050b13-observe-repaired-best.py` (32/32 nodes exit=0, 256/256 receipts)
- **Update 9–30 均值**: `22.4989s/update`
- **PaLM MFU**: `22.6152%`
- **峰值显存 (NPU:0)**: `50.45 GiB` allocated / `52.89 GiB` reserved

## Stage 1: 交替 `2+1 / 1+2` EDP 专家分片均衡
- **Run ID**: `huangxs-opt1-edp-balance-repeat30-v4`
- **实现**: `/mnt/xufan_400T/huangxs/hp-v41/full-v41/opt1-observe-edp-balance.py` (基于 `14050b13-observe-repaired-best.py`)
  - 将奇数编号的专家 FSDP unit (`edp_shard=2`) 从默认 `2+1` 翻转为 `1+2` 分片（同步适配 `HSDPParamV2` 的 `all_gather`/`reduce_scatter` 以及 `clip_grad._norm_coverage`）。
  - 消除 Superpod A/B 间 Muon Newton-Schulz 负载不均（local expert slices 从 `166 vs 80` 均衡至 `124 vs 122`）。
- **验证状态**: 32/32 节点 `exit_code=0` (`COMPLETED`)，256/256 `mhc-cache` 与 `edp-balance` 回执齐全，2-NPU 单元测试与基线 `0.000e+00` 完全一致。
- **Update 9–30 均值**: **`22.0949s/update`** (`-0.4040s/update`)
- **PaLM MFU**: **`23.0287%`** (**`+0.4135%`**)
- **收敛与显存**: `step=30` `loss=1.91423`, `grad_norm=8.08925`, `max_alloc=50.5312 GiB`, `max_reserved=54.998 GiB`

## Stage 2: 叠加 Vision FSDP `comm_fusion=True` (`4714ab8e`)
- **Run ID**: `huangxs-opt2-edp-vision-repeat30-v1`
- **实现**: `/mnt/xufan_400T/huangxs/hp-v41/full-v41/opt2-observe-vision-comm-fusion.py` (基于 `opt1-observe-edp-balance.py`)
  - 对 `model.vision.*` 单元启用 `comm_fusion=True, comm_fusion_zero_copy=False`，将逐参数集合通信合并为按 dtype/domain 分桶。
- **验证状态**: `total_steps=30` 完成，256/256 `mhc-cache`、`vision-comm-fusion`、`edp-balance` 回执齐全。
- **Update 9–30 均值**: **`21.9461s/update`** (`-0.1488s` vs Stage 1，累计 `-0.5528s` vs Baseline)
- **PaLM MFU**: **`23.1849%`** (`+0.1562%` vs Stage 1，累计 **`+0.5697%`** vs Baseline)
- **收敛与显存**: `step=30` `loss=1.91109`, `grad_norm=8.00222`, `max_alloc=50.5312 GiB`, `max_reserved=54.2559 GiB`

## Stage 3: 扩展 `comm_fusion=True` 至全部非专家 FSDP 单元 (已否决 / Rejected)
- **Run ID**: `huangxs-opt3-edp-dense-fusion-repeat30-v1`
- **实现**: `/mnt/xufan_400T/huangxs/hp-v41/full-v41/opt3-observe-dense-comm-fusion.py` (基于 Stage 1，对所有非 `edp_shard` 单元开启 `comm_fusion=True, comm_fusion_zero_copy=False`)
- **验证状态**: 32/32 节点 `exit_code=0` (`COMPLETED`)
- **Update 9–30 均值**: `22.1639s/update` (**慢于 Stage 2 的 `21.9461s`**, `+0.2178s`)
- **PaLM MFU**: `22.9570%` (**低于 Stage 2 的 `23.1849%`**)
- **峰值显存**: `max_alloc=51.0489 GiB` (`+0.52 GiB`)
- **结论**: 对 `model.layers.*` 开启 `comm_fusion` 会合并整层参数桶、破坏逐参数通信与计算的流式重叠，得不偿失；已回退，继续以 **Stage 2 (`23.1849%`)** 为当前最优基线。

## Stage 4: `clip_grad.scale_grad_refs_` `foreach` 批量化（消除 1,262 次 `.item()` 同步）
- **Run ID**: `huangxs-opt4-sync-foreach-repeat30-v2`
- **实现**: `/mnt/xufan_400T/huangxs/hp-v41/full-v41/opt4-observe-sync-and-foreach.py` (基于 Stage 2)
  - 将 `scale_grad_refs_` 从逐张量 `grad.mul_(0D_tensor)` 循环改为单次 `.item()` + `torch._foreach_mul_`（`coef==1.0` 时直接跳过）。
- **验证状态**: 32/32 节点 `exit_code=0` (`COMPLETED`)
- **Update 9–30 均值**: **`21.9304s/update`** (`-0.0157s` vs Stage 2，累计 `-0.5685s` vs Baseline)
- **PaLM MFU**: **`23.2015%`** (`+0.0166%` vs Stage 2，累计 **`+0.5863%`** vs Baseline)
- **收敛与显存**: `step=30` `loss=1.90725`, `grad_norm=7.97096`, `max_alloc=50.5312 GiB`, `max_reserved=54.4336 GiB`

## Stage 5: 补齐 `host_table.host_lookup` 跳过每步 Engram `stage_check`
- **Run ID**: `huangxs-opt5-engram-fast-repeat30-v1`
- **实现**: `/mnt/xufan_400T/huangxs/hp-v41/full-v41/opt5-observe-engram-fast.py` (基于 Stage 4)
  - 补齐 `host_table.host_lookup` 引用替换，仅前 4 次调用执行 `min/max` + 256 卡 `stage_check` (`all_reduce`)。
- **验证状态**: 32/32 节点 `exit_code=0` (`COMPLETED`)
- **Update 9–30 均值**: **`21.9090s/update`** (`-0.0214s` vs Stage 4，累计 `-0.5899s` vs Baseline)
- **PaLM MFU**: **`23.2241%`** (`+0.0226%` vs Stage 4，累计 **`+0.6089%`** vs Baseline)
- **收敛与显存**: `step=30` `loss=1.91438`, `grad_norm=8.00236`, `max_alloc=50.5312 GiB`, `max_reserved=54.4316 GiB`

## Stage 6: `TASK_QUEUE_ENABLE=2` (已否决 / Rejected，快速筛查提前终止)
- **Run ID**: `huangxs-opt6-taskqueue2-repeat30-v1` (于 step 4 提前终止并清理)
- **实测结果**: `step=2` `28.156s`, `step=3` `23.795s`, `step=4` `26.122s`（显著慢于 Stage 5 的 ~`21.75s`）
- **结论**: `TASK_QUEUE_ENABLE=2` 在当前算子/MegaKernel 混合负载下引入额外排队/调度开销，已否决并回退至 **Stage 5 (`23.2241%`)**。

## Stage 7: `HCCL_OP_EXPANSION_MODE=AIV` (已否决 / Rejected，step 1 卡死)
- **Run ID**: `huangxs-opt7-hccl-aiv-repeat30-v2` (于第 1 步超时 6 分钟无输出后终止并清理)
- **结论**: 在当前 MegaKernel + HCCL 混合负载下切至 `AIV` 展开导致首步卡住，已否决并回退至 **Stage 5 (`23.2241%`)**。

## Stage 8: 选择性重计算跳过末尾层 (已否决 / Rejected，HBM OOM)
- **Run ID**: `huangxs-opt8-selective-recompute3-repeat30-v1` (跳过 3 层 `37..39`)、`huangxs-opt8-selective-recompute2-repeat30-v2` (跳过 2 层 `38..39`)
- **实测结果**: 跳过 3 层在 `56.24 GiB` OOM，跳过 2 层在 `55.90 GiB`（reserved `56.45 GiB` / 可用上限 `61.28 GiB`）OOM。
- **结论**: 每层激活约占 `2.2 GiB`，61.28 GiB 可用显存扣除 CANN/HCCL/SHMEM 运行时后仅余 ~`2 GiB` 安全余量，无法通过整层跳过重计算换取算力。已否决并回退至 **Stage 5 (`23.2241%`)**。








## Stage 9: Host Engram 前向→重计算查表缓存
- **Run ID**: `huangxs-opt9-engram-cache-repeat30-v1` (32/32 节点 `exit_code=0`, `COMPLETED`)
- **实现**: `/mnt/xufan_400T/huangxs/hp-v41/full-v41/opt9-observe-engram-recompute-cache.py` (基于 Stage 5)
  - 在 `no_grad` 前向阶段缓存 `inverse`、`send_counts/recv_counts`、`owned_local_ids` 与 `transfer_values`，反向重计算阶段直接复用，跳过重计算中的 `torch.unique`、`all_gather`、`ep_all_to_all(unique_ids)` 与 CPU 查表。
- **Update 9–30 均值**: **`21.8985s/update`** (`-0.0105s` vs Stage 5，累计 `-0.6004s` vs Baseline)
- **PaLM MFU**: **`23.2353%`** (`+0.0112%` vs Stage 5，累计 **`+0.6201%`** vs Baseline)
- **收敛与显存**: `step=30` `loss=1.91878`, `grad_norm=8.09579`, `max_alloc=50.5312 GiB`, `max_reserved=55.3320 GiB`

## Stage 10: 选择性跳过最后 1 层 (`layer 39`) 重计算
- **Run ID**: `huangxs-opt10-skip1-repeat30-v1` (32/32 节点 `exit_code=0`, `COMPLETED`)
- **实现**: `/mnt/xufan_400T/huangxs/hp-v41/full-v41/opt10-observe-skip1.py` (基于 Stage 9)
  - 仅对 `layer_idx >= 39` 跳过 `activation_checkpoint`（`layer 39` 在反向最先释放，不抬高全局显存峰值）。
- **Update 9–30 均值**: **`21.8370s/update`** (`-0.0615s` vs Stage 9，累计 `-0.6619s` vs Baseline)
- **PaLM MFU**: **`23.3007%`** (`+0.0654%` vs Stage 9，累计 **`+0.6855%`** vs Baseline)
- **收敛与显存**: `step=30` `loss=1.92153`, `grad_norm=8.12574`, `max_alloc=50.5312 GiB`, `max_reserved=55.8008 GiB`

## Stage 11: `prefetch_depth=3` + `dspark` 非专家 `comm_fusion` (已否决 / Rejected)
- **Run ID**: `huangxs-opt11-prefetch3-dspark-repeat30-v2` (于 step 28 提前终止并清理)
- **实测结果**: `updates 9–28` 均值 `23.9996s` (`21.2011% MFU`)，显存 `max_reserved` 升至 `56.2285 GiB` 引发分配器碎片整理抖动（单步慢至 `26.8s`）。
- **结论**: `prefetch_depth=3` 占用过多并发预取缓冲导致显存逼近上限，已否决并回退至 **Stage 10 (`23.3007%`)**。

## Stage 12: 关闭训练步内非同步 Python GC (`gc.disable()`)
- **Run ID**: `huangxs-opt12-gc-sync-repeat30-v1` (30/30 steps 完成)
- **实现**: `/mnt/xufan_400T/huangxs/hp-v41/full-v41/opt12-observe-gc-sync.py` (基于 Stage 10)
  - `on_train_begin` 后执行一次 `gc.collect()` 并 `gc.disable()`，消除 256 个独立 Python 进程在不同步时触发 Gen-2 GC 导致的全网同步毛刺（`on_train_end` 恢复）。
- **Update 9–30 均值**: **`21.6656s/update`** (`-0.1714s` vs Stage 10，累计 `-0.8333s` vs Baseline)
- **PaLM MFU**: **`23.4850%`** (`+0.1843%` vs Stage 10，累计 **`+0.8698%`** vs Baseline)
- **收敛与显存**: `step=30` `loss=1.92202`, `grad_norm=8.17003`, `max_alloc=50.5312 GiB`, `max_reserved=55.8008 GiB`

## Stage 13: `dspark` / `embed` / `lm_head` 开启 `comm_fusion=True` (已否决 / Rejected)
- **Run ID**: `huangxs-opt13-dspark-fusion-repeat30-v1` (按 `9–15` 步快测于 step 15 终止并清理)
- **实测结果**: `steps 9–15` 均值 `21.8528s` (`23.2838% MFU`)，慢于 Stage 12 的 `21.6656s` (`23.4850%`)。
- **结论**: `dspark` 阶段开启 `comm_fusion` 同样引入分桶拷贝与同步开销，仅保留 `model.vision.*` 的 `comm_fusion` 最优；回退至 **Stage 12 (`23.4850%`)**。

## Stage 14: 算子融合第一波 (`FusedClampedSwiGLU` 6.08x + `npu_rms_norm` 5.40x + RoPE `cos/sin` 缓存)
- **Run ID**: `huangxs-opt14-fused-ops-repeat30-v1` (按 `steps 3–7` 稳态快测验证成功)
- **实现**: `/mnt/xufan_400T/huangxs/hp-v41/full-v41/opt14-observe-fused-swiglu-rmsnorm.py` (基于 Stage 12)
  1. `FusedClampedSwiGLU` Triton 融合核替换 `deepseek_v41_swiglu` 朴素 7 算子（单算子 `5.261ms -> 0.865ms`, `6.08x`），省去重计算 5 份 `(1,16384,2304)` FP32 中间张量；
  2. `torch_npu.npu_rms_norm` 替换 `DeepseekV4RMSNorm` 朴素 8 算子（单算子 `1.789ms -> 0.331ms`, `5.40x`），消除每层 `q_norm`/`kv_norm` 的 `.item()` 同步；
  3. `_V41PartialRope` 缓存 `cos/sin` 的 `repeat_interleave(2, dim=-1)`。
- **稳态 `steps 3–7` 均值**: **`21.2915s/update`** (`21.3137s, 21.2672s, 21.3024s, 21.2686s, 21.3054s`；相比 Stage 12 `21.6656s` **再缩短 `0.3741s`**，累计 **`-1.2074s`** vs Baseline)
- **PaLM MFU**: **`23.8977%`** (最快单步 **`23.9250%`**；相比 Stage 12 提升 **`+0.4127%`**，相比 Baseline 累计提升 **`+1.2825%`**)
- **显存下降**: `max_alloc=50.398 GiB` (`-0.13 GiB`), `max_reserved=55.031 GiB` (`-0.77 GiB`)

## Stage 15: 深度算子融合 + 消除主机同步与冗余拷贝 (`opt15-observe-fused-v2.py`)
- **Run ID**: `huangxs-opt15-fused-v2-repeat30-v6` (`steps 1–24` 实测后手动停止并清理，直接进入下一轮快速迭代)
- **实现**: `/mnt/xufan_400T/huangxs/hp-v41/full-v41/opt15-observe-fused-v2.py` (基于 Stage 14)
  - `EngramModule._fuse` 门控正反向替换为融合 Triton 核（1 卡实测 `8.42x` 加速，消除 `(1, 16384, 4, 5120)` 瞬态 FP32 缓冲区）；
  - `DSpark _StreamCollapse` 正反向替换为融合 Triton 核（1 卡实测 `5.70x` 加速）；
  - `_PackedSwiGLU.backward` 替换为单算子 Triton 反向核（1 卡实测 `4.66x` 加速），并将 Stage 14 Triton `ClampedSwiGLU` 同步绑定到 `moe_engram_expert_parallel`；
  - `_V41PartialRope` 在 `not ctx.needs_input_grad[0]` 前向与非连续反向时启用原地末尾 64 通道旋转核（免除 87.5% 未旋转通道搬运）；
  - `MegaMoE` 缓存 `* spec.hidden_size` 偏移张量、跳过 `not any(ctx.needs_input_grad)` 前向中的冗余 `down_proj.copy_(dispatch)`、并将 `_write_weight_grad` 改为 `torch.mm(..., out=output)`；
  - `_resolve_router` 缓存固定路由索引 `balanced`、`accumulate_router_load` 在 `no_grad` 前向直接跳过、`BaseTrainer.loss_normalize` 改为 `torch._foreach_div_`、Muon Newton-Schulz 合并 `mul_().add_()`。
- **Steps 3–24 均值**: **`21.0666s/update`** (`-0.2249s` vs Stage 14，累计 `-1.4323s` vs Baseline；`steps 9–24` 均值 `21.0735s`, 最快单步 `21.0210s`)
- **PaLM MFU**: **`24.1528%`** (`+0.2551%` vs Stage 14，累计 **`+1.5376%`** vs Baseline；最快单步 `24.2052%`)
- **显存大幅下降**: `max_alloc=49.085 GiB` (`-1.313 GiB` vs Stage 14), `max_reserved=53.410 GiB` (`-1.621 GiB` vs Stage 14)

## Stage 16: 尝试额外跳过 `layer 38` 重计算 (已否决 / Rejected，HBM OOM)
- **Run ID**: `huangxs-opt16-skip2-repeat30-v1` (基于 Stage 15，跳过 `layers >= 38` 的 `activation_checkpoint`)
- **实测结果**: 在 `chunked_cross_entropy` 处触发 OOM（`54.99 GiB` allocated / `55.53 GiB` reserved，尝试再分配 `1.23 GiB`，上限 `61.27 GiB`）。整层不重算额外增加约 `~5.9 GiB` 激活驻留，超过当前余量。
- **结论**: 已清理并回退至 **Stage 15 (`24.1528% MFU`, `21.0666s/update`)**。

## Stage 17: `prefetch_depth=3` (不含 `dspark` 融合) + DSpark `rstd` 融合 + Muon `norm` 同步消除 (已否决 / Rejected)
- **Run ID**: `huangxs-opt17-prefetch3-v2-repeat30-v2` (于 step 4 提前终止并清理)
- **实测结果**: `step 3=21.1528s`, `step 4=21.1363s` (`24.05%–24.07% MFU`)，`max_alloc=49.372 GiB`, `max_reserved=54.135 GiB`（`+0.725 GiB`）。`prefetch_depth=3` 提前排队过多 AllGather 导致通信/分配器竞争，反而慢约 `+0.08s/step`。
- **结论**: 已否决 `prefetch_depth=3`，保留 Stage 15 为当前最优基线（后续保留 DSpark `rstd` 与 Muon `norm` 优化）。

## Stage 18: 阶段耗时拆解 (`STEP4_PHASE_BREAKDOWN`) + DSpark `rstd` & Muon `norm` 单独验证 (已否决 / Rejected)
- **实现**: `/mnt/xufan_400T/huangxs/hp-v41/full-v41/opt18-observe-triton-rstd-and-profile.py` (基于 Stage 15，不含 `prefetch_depth=3`)
- **实测结果**: `step 3=21.0933s`, `step 4=21.0868s` (`24.13% MFU`，略慢于 Stage 15 的 `21.0666s`，不保留 DSpark `rstd` 与 Muon `norm` 替换)。
- **关键阶段耗时拆解 (Rank 0, Step 4)**:
  - `fwd_bwd = 19.0996s` (`90.8%`)
  - `prepare_clip = 0.3325s` (`1.6%`)
  - `optimizer = 1.6077s` (`7.6%`)
  - `sum = 21.0398s`
- **结论**: 所有 32 节点已清理；后续重点聚焦 `fwd_bwd` 内的 `MegaMoE` (`4.27s`)、`SparseFlashMla` (`2.55s`)、`indexer_kl` / `RoPE` / 重计算策略，以及 `optimizer` (`1.61s`) 与 `prepare_clip` (`0.33s`)。

## Stage 19: 跳过 `layer 39` 重计算 + `_expand_query` 原地 RoPE + `foreach` `clip_grad` (已否决 / Rejected)
- **Run ID**: `huangxs-opt19-skip1-rope-clip-repeat30-v1` (于 step 4 报错退出并清理全部 32 节点)
- **实测结果**: `step 3 = 22.5495s`（慢于 Stage 15 的 `21.0666s`），`max_alloc = 49.0523 GiB`，`max_reserved = 52.2559 GiB`（`-1.15 GiB`）；step 4 因性能打点代码 `self.model_fn` 属性名笔误触发 `AttributeError` 退出。
- **原因分析**: `_owned_partial_rope_inplace_kernel` 按单 head (`1 row`) 迭代且仅启 128 program，向量宽度比原生 `v41_partial_rope.py`（一次处理 64 heads）小 64 倍，在 `_expand_query` 80 次调用上引入约 `+1.5s` 退化；但 `layer 39` 跳过重计算与 `foreach` `clip_grad` 显存完全安全（峰值甚至低于 Stage 15）。
- **结论**: 已清理全部 32 节点；下一轮改用原生 64-head 向量化 `_partial_rope_kernel` 实现原地旋转，并叠加 `Muon` 批大小调优与 `FSDP` 尾部层免重分片。

## Stage 20: 64-head 向量化原地 RoPE + 跳过 `layer 39` 重计算 + 尾部 4 层免重分片 + `foreach` `clip_grad` + `Muon` 4x 批上限
- **Run ID**: `huangxs-opt20-vec-rope-skip1-clip-muon-repeat30-v1` (于 step 4 稳态验证后停止并清理)
- **实现**: `/mnt/xufan_400T/huangxs/hp-v41/full-v41/opt20-observe-vec-rope-skip1-clip-muon.py` (基于 Stage 15)
  1. 用 64-head 向量化 `_partial_rope_inplace_vec_kernel[(48,)]` 替换 Stage 15 单 head 核，并覆盖 `_expand_query` 80 次调用；
  2. 跳过 `layer 39` `activation_checkpoint`，并对 `layers 36..39` 关闭 `reshard_after_forward` (`OPT20_NO_RESHARD_APPLIED=4`)；
  3. `clip_grad_norm_` 启用 BF16 `_foreach_norm` 与解包 `DTensor` 后 `_foreach_mul_`；
  4. `Muon._split_into_memory_safe_batches` 批元素上限提升 4 倍 (`2048M // shard_size`)。
- **实测结果 (`steps 3–4`)**: `step 3 = 20.8784s` (`24.3705% MFU`), `step 4 = 20.8557s` (**`24.3970% MFU`**)，均值 **`20.8671s/update` (`24.3838% MFU`)**，相比 Stage 15 (`21.0666s / 24.1528%`) **再提速 `0.200s` (`+0.231% MFU`)**，相比 Baseline 累计提速 **`1.6318s` (`+1.7686% MFU`)**。
- **Step 4 阶段耗时拆解**: `fwd_bwd = 18.9514s` (`-0.1482s`), `prepare_clip = 0.3173s` (`-0.0152s`), `optimizer = 1.5413s` (`-0.0664s`), `sum = 20.8099s`。
- **显存**: `max_alloc = 51.3631 GiB`, `max_reserved = 53.8574 GiB`（仍有约 `~3.6 GiB` 显存余量）。

## Stage 21: `HCCL_OP_EXPANSION_MODE=AIV` + `TASK_QUEUE_ENABLE=2` (已否决 / Rejected)
- **Run ID**: `huangxs-opt21-noreshard8-aiv-tq2-repeat30-v1` (step 1 卡死超 3 分钟，已终止并清理全部 32 节点)
- **结论**: `HCCL_OP_EXPANSION_MODE=AIV` 与 `MegaMoE` AIV 算子/通信核冲突导致 step 1 阻塞；回退至 **Stage 20 (`20.8557s / 24.3970% MFU`)**，不修改 `HCCL_OP_EXPANSION_MODE`。

## Stage 22: 尾部 8 层免重分片 + `layers 37..38` 跳过 `MegaMoE` 重计算 (已否决 / Rejected，HBM OOM)
- **Run ID**: `huangxs-opt22-noreshard8-attnckpt2-repeat30-v1` (OOM 于 `56.00 GiB` allocated + `962 MiB`，已清理全部 32 节点)
- **原因分析**: 单层 `MegaMoE` 保存 `up_proj + activation + down_proj` 约需 `~2.4 GiB`，2 层不重计算 + 8 层 `no-reshard` 超出 `56.0 GiB` 上限；回退至 **Stage 20 (`20.8557s / 24.3970% MFU`)**。

## Stage 23: 尾部 6 层免重分片 + `layer 38` 仅重计算 `self_attn` (已否决 / Rejected，HBM OOM)
- **Run ID**: `huangxs-opt23-noreshard6-attnckpt1-repeat30-v1` (OOM 于 `55.02 GiB` allocated + `1.23 GiB`，已清理全部 32 节点)
- **结论**: 即使仅保留 1 层 `MegaMoE` 状态 (`layer 38`) + 6 层 `no-reshard` 仍略超 `56.25 GiB` 峰值；后续不再减少 `MegaMoE` 重计算，专注纯算子/通信/优化器/尾部免重分片优化。

## Stage 24: 尾部 8 层免重分片 + 跳过 DSpark `_collapse` 重计算 + `TASK_QUEUE_ENABLE=2` (已否决 / Rejected，NPUWorkspaceAllocator OOM)
- **Run ID**: `huangxs-opt24-noreshard8-tq2-repeat30-v1` (`OPT24_NO_RESHARD_APPLIED=8`，触发 `NPUWorkspaceAllocator` OOM，已清理全部 32 节点)
- **结论**: 8 层免重分片 + `TASK_QUEUE_ENABLE=2` 工作区队列膨胀超限；保持 Stage 20 (`4` 层免重分片，`51.36 GiB` alloc / `53.86 GiB` res) 与默认 `TASK_QUEUE_ENABLE`。

## Stage 25: 尾部 7 层免重分片 + 跳过 DSpark `_collapse` 重计算 + `Muon` 4G 批 (已否决 / Rejected，`node 16` 43.84 GiB OOM)
- **Run ID**: `huangxs-opt25-noreshard7-muon4g-repeat30-v1` (`OPT25_NO_RESHARD_APPLIED=7`，`run-16` 于 `43.84 GiB` allocated 触发 OOM，已清理全部 32 节点)
- **结论**: `node 16` 可用显存上限低于 `node 0`，维持 Stage 20 的 4 层免重分片 (`51.36 GiB` alloc)，后续在显存零增长方向深挖。
