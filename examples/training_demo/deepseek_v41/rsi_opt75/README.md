# DeepSeek V4.1 Flash 763B · opt75

The archived 16K, global-batch-256 workload reached **26.3587% model FLOPs utilization** and **838.13 effective supervised tokens/s/die** with 32 nodes, 256 Ascend dies, fixed expert routing, and a nominal 353 TFLOP/s per die. Three independent 30-update runs contribute updates 9–30. Each run contains all 256 rank trajectories with finite loss and gradient norms. [Measurement](benchmark.json)

The training checkout starts at `eec395c4d5b1456fa0338c3beac8cbf222b41fca`. The archived opt75 launch activates the versioned Python sources in `frozen/` together with the checkout. Its execution policy includes HostEngram gradient handling, DSpark checkpoint and loss scheduling, equivalent attention and expert projection layouts, and shape-specific Muon and vocabulary-head paths. The launcher selects parts `73,616,756` and uses the recorded 16K recipe.

## Run the archived workload

The recorded execution used CANN 9.2.1, Torch 2.10.0 CPU base, TorchNPU 2.10.0.post7, the native HyperParallel Multicore and Omni OPP packages, and the CSA2 package built from `0b08075141913daeaf1e856faf7480ed393c8949`. Mount the prepared training assets, runtime, and operator packages at the paths in the frozen recipe and runtime manifest. The Sparse Flash MLA extension is supplied by the dual-backward runtime overlay in `smla-dual-0100acf2`. Its source hashes are recorded alongside the recipe. Model metadata and tokenizer, the Engram assets, the input corpus and images, and the native packages are external requirements. [Frozen recipe and input digests](frozen/full-model/S1/dsv41-vlm-256die-16k-contained-recompute-20c1e5cd-repeat30-v1/)

Run from the repository root. Stage the matching sources once into the shared directory mounted as `/xw-tree/full-v41` by the per-node training containers.

```bash
python examples/training_demo/deepseek_v41/rsi_opt75/prepare_runtime.py \
  --runtime-root /path/to/your/runtime/full-v41 --stage
```

Activate the compiled native payload in `DSV41_FULL_REPO/build/native`, using the original base identity and native-payload manifest consumed by the frozen launcher. Provide `NNODES=32`, `NPU_PER_NODE=8`, the local `NODE_RANK`, and the allocation's `MASTER_ADDR`. Run the same command on each participating node in its training container. `DSV41_FULL_REPO` defaults to this checkout and can point to a prepared copy of it.

```bash
bash examples/training_demo/deepseek_v41/rsi_opt75/run_opt75.sh v41-opt75-repeat
```

The source bundle preserves the observed install order and verifies content before launch. The reported MFU comes from the archived hardware runs. Release verification covers source checksums, recipe resolution, and syntax checks.
