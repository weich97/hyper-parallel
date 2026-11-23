#!/usr/bin/env bash
# Run the archived opt75 training recipe on a provisioned 32-node allocation.
set -eo pipefail

bundle=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
checkout=$(cd -- "$bundle/../../../.." && pwd)
root=/xw-tree/full-v41
run_name=${1:?provide a unique benchmark run name}
[[ "$run_name" =~ ^[A-Za-z0-9_-]+$ ]]
[[ "${NNODES:?}" == 32 && "${NPU_PER_NODE:?}" == 8 ]]
: "${NODE_RANK:?}" "${MASTER_ADDR:?}"

export DSV41_FULL_REPO="${DSV41_FULL_REPO:-$checkout}"
python "$bundle/prepare_runtime.py" --runtime-root "$root"
(cd "$DSV41_FULL_REPO" && sha256sum --check --strict "$bundle/hp-source.sha256")
source "$root/runtime-cann921/install/runtime-cann921-env-b15c7b60.sh"

export DSV41_FULL_WORKER="$root/51999d36-bootstrap-repaired-affinity.py" DSV41_FULL_MODE=benchmark
export DSV41_FULL_DIAGNOSTICS="$root/opt75-stack-indexer-plus.py" DSV41_FULL_EXPECT_FUSED_KL=1
export DSV41_SMALL_GAIN_PARTS=73,616,756
export HP_EP_EQUAL_A2A=1 TASK_QUEUE_ENABLE=2 PYTHONFAULTHANDLER=1
export DSV41_CSA_PERFORMANCE_PARENT_SHA=bb572c9ab5ed1eeacc16c6d9be36a04c164f20770a59c7990896eb55e13dce2b
export DSV41_SHARD_PARENT="$root/0fe83cff-observe-dspark-local-checkpoint.py"
export DSV41_SHARD_PARENT_SHA=0fe83cff302ca0b90820b8ada68970757b648118ddff520779ea83f713fa466b
export DSV41_EXPECT_DENSE_SHARDS=32
exec bash "$root/2e542f80-launch-full-portable.sh" "$run_name" \
    eec395c4d5b1456fa0338c3beac8cbf222b41fca dsv41-vlm-256die-16k-contained-recompute-20c1e5cd-repeat30-v1 30
