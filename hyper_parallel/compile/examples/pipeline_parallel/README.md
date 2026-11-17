# VPP Example (Graph-Pass Interleaved 1F1B)

Virtual pipeline-parallel (VPP) training with HyperParallel graph mode. The
joint forward+backward FX graph is sliced into per-rank *virtual* stages by
`PpPass` — **at graph level**, in contrast to torchtitan, which splits the
model itself (deepcopy + layer deletion) before wrapping chunks in
`PipelineStage`.

The shipped config runs **2 physical ranks × 2 chunks = 4 virtual stages**
with `ScheduleInterleaved1F1B`: rank `r` owns virtual stages
`r, r + pp_degree, ...`, and the schedule interleaves forwards/backwards
across those chunks to shrink the pipeline bubble.

## Quick start

```bash
cd hyper_parallel/compile/examples/pipeline_parallel

# 2 physical ranks, 2 virtual stages each (4 virtual stages total)
bash run_2cards.sh

# or manually:
torchrun --nproc_per_node=2 train.py --config config.yaml
```

On an NPU node the process group uses hccl automatically; other platforms
fall back to gloo.

## How it works

```text
joint graph (fwd+bwd, params as static placeholders)
        │  FSDPPass (off here) / PpPass
        ▼
per-virtual-stage slices: fwd_gm + bwd_gm  ── boundary act/grad via isend/irecv
        │
        ▼
ScheduleInterleaved1F1B installed as a call_module node in the graph stub
        │
        ▼
GraphTrainer runs graph_module(*state, *inputs) unchanged; user inputs are
routed by dataflow to every stage that consumes them (no arity constraint)
```

1. **Node → virtual-stage attribution** — every FX node is mapped to a
   virtual stage via its `nn_module_stack` metadata; parameters (static
   placeholders) are attributed by FQN ancestor walk against the stage plan.
2. **Boundary detection** — the activation crossing virtual stage k → k+1
   (and the backward gradient crossing back) is located.
3. **Graph surgery** — each virtual stage's nodes are sliced into a forward
   subgraph and a backward subgraph; forward outputs (boundary activation +
   every intermediate the backward needs) become backward placeholders.
4. **Live-model pruning** — foreign-stage submodules are removed in place,
   so `model.parameters()` / the optimizer only see this rank's virtual
   stages.
5. **Schedule install** — `ScheduleInterleaved1F1B` (eager
   `dist.isend`/`irecv`, microbatched, grads averaged over microbatches) is
   installed as a `call_module` node, surviving recompiles; the trainer needs
   zero PP awareness.

## Stage split options

**Automatic (default)** — the first top-level `nn.ModuleList` with at
least `pp_degree * pp_chunks` entries is the layer container; it is
distributed evenly across the virtual stages. Modules before it go to virtual
stage 0, modules after it to the last virtual stage:

```yaml
pp: {}   # omit stages -> auto even split across virtual stages
```

For this example the automatic plan is:

```text
virtual 0 (rank 0) -> [tok_embeddings, layers.0]
virtual 1 (rank 1) -> [layers.1]
virtual 2 (rank 0) -> [layers.2]
virtual 3 (rank 1) -> [layers.3, norm, lm_head]
```

**Manual** — exact module FQNs per virtual stage (no wildcards; one list per
virtual stage):

```yaml
pp:
  stages:
    - [tok_embeddings, layers.0]
    - [layers.1]
    - [layers.2]
    - [layers.3, norm, lm_head]
```

Both come from `GraphParallelPlan.pp_stage(stage_idx, fqns)` /
`create_plan_from_yaml`.

## Configuration reference (`parallel:` section)

| Key | Default | Meaning |
|-----|---------|---------|
| `pp_enabled` | `false` | Opt in to the PP pass |
| `pp_degree` | `null` | `null` → `world_size` (physical pipeline ranks) |
| `pp_chunks` | `1` | Virtual stages per rank; this example uses `2` |
| `pp_microbatch_size` | `1` | Samples per microbatch; batch must be divisible |
| `pp_schedule` | `"gpipe"` | `"gpipe"`, `"1f1b"`, or `"vpp"` / `"interleaved_1f1b"` (example uses `vpp`) |
| `enable_overlap` | `true` | No-op under PP (the schedule overlaps at Python level) |

## Current constraints

- **Pure PP by default**: `fsdp_enabled=False`. The PP+FSDP / PP+DDP hybrid
  is supported through the compiler's 2-D `(pp, DP)` mesh: set
  `fsdp_enabled=True` plus `pp_degree` and `fsdp_degree` /
  `dp_replicate_degree` with `pp_degree * dp_degree == world_size` — the
  intra-stage DP group and the cross-stage `pp` group are registered
  automatically (PP-outer rank layout: each stage's DP ranks are
  contiguous). PP+HSDP and PP composed with an external `MeshContext` are
  not supported yet.
- **Microbatch-shaped compile**: the trainer must `compile()` once with a
  microbatch-shaped sample (see `train.py`) — the traced graph bakes view
  sizes for one microbatch, and the schedule slices every full batch into
  matching microbatches at runtime.
- **Multi-value stage cuts**: every value crossing a cut ships over P2P
  (tensors as-is, int scalars packed as 0-d int64). Gradients flow back
  during the backward sweep.
- **Schedule choice**: this example uses `pp_schedule: vpp` (alias
  `interleaved_1f1b`) with `pp_chunks: 2`. The model is split into
  `pp_degree * pp_chunks` virtual stages and the classic
  interleaved-1F1B warmup / steady / cooldown order pipelines them. A manual
  plan must declare one FQN list per virtual stage.
  To reproduce the previous non-VPP example, set `pp_chunks: 1` and
  `pp_schedule: 1f1b` (or `gpipe`); the virtual stage count then collapses
  back to `pp_degree`.
- Loss is only real on the last virtual stage; other stages return a zero
  scalar (see `Last-stage loss:` in the run output for convergence).
