#!/usr/bin/env bash
set -eo pipefail
umask 022

root=/xw-tree/full-v41
experiment="$root/full-model"
name=${1:?unique run name}
commit=${2:?source commit}
recipe=${3:?frozen recipe directory}
steps=${4:?optimizer updates}
[[ "$name" =~ ^[a-zA-Z0-9_-]+$ && "$commit" =~ ^[0-9a-f]{40}$ ]]
[[ "$recipe" =~ ^[a-zA-Z0-9_-]+$ && "$steps" =~ ^[1-9][0-9]*$ ]]
test "$NPU_PER_NODE" -eq 8
test "$NNODES" -eq 32
test -d /xw-tree/full-v41/hf-metadata

repo="${DSV41_FULL_REPO:-$root/hp-backbone-$commit}"
prepared="$experiment/S1/$recipe"
export DSV41_FULL_RUN_DIR="$experiment/runs/$name"
export DSV41_FULL_WORLD=$((NNODES * NPU_PER_NODE)) DSV41_FULL_STEPS="$steps"
mkdir -p "$experiment/logs/$name"
exec > >(tee "$experiment/logs/$name/node-$NODE_RANK.log") 2>&1

test "$(cat "$repo/.backbone-v2-ready")" = "$commit"
test "$(cat "$repo/.native-megamoe-ready")" = "$commit"
cann_env="${DSV41_FULL_CANN_ENV:-/usr/local/Ascend/cann-9.1.0/set_env.sh}"
source "$cann_env"
export ASCEND_CUSTOM_OPP_PATH=""
source /xw-tree/omni-opp/vendors/omni_training_custom_transformer/bin/set_env.bash
native_root=$(readlink -f "$repo/build/native")
source "$native_root/payload/hyper_parallel/core/multicore/lib/set_env.bash"
csa2=/xw-tree/codex-s03/csa2-pinned/package
test "$(cat "$csa2/source.sha")" = 0b08075141913daeaf1e856faf7480ed393c8949
source "$csa2/runtime-env.sh"
mkdir -p /tmp/hp-vendors
cp -a /xw-tree/omni-opp/vendors/omni_training_custom_transformer /tmp/hp-vendors/
cp -a "$csa2/opp/vendors/custom_transformer" /tmp/hp-vendors/
cp -a "$native_root/payload/hyper_parallel/core/multicore/lib/vendors/hyper_parallel_multicore_nn" /tmp/hp-vendors/
chown -R 0:0 /tmp/hp-vendors
export ASCEND_CUSTOM_OPP_PATH="/tmp/hp-vendors/custom_transformer:/tmp/hp-vendors/hyper_parallel_multicore_nn:/tmp/hp-vendors/omni_training_custom_transformer:$ASCEND_CUSTOM_OPP_PATH"
export LD_LIBRARY_PATH="/tmp/hp-vendors/custom_transformer/op_api/lib:/tmp/hp-vendors/hyper_parallel_multicore_nn/op_api/lib:/tmp/hp-vendors/omni_training_custom_transformer/op_api/lib:$LD_LIBRARY_PATH"
overlay="$root/smla-dual-0100acf2"
printf '%s  %s\n' ab71a8ce052cb258b5219eb21f25a8dd4fbaa641d05a7c85801985fbb2b6109e "$overlay/overlay.json" | sha256sum -c -
export PYTHONPATH="$overlay:$repo${PYTHONPATH:+:$PYTHONPATH}:/xw-tree/omni-pydeps:/xw-tree/pydeps-extra"
export TORCH_EXTENSIONS_DIR="${DSV41_NODE_CACHE_ROOT:-/var/tmp}/dsv41-mhc-cache-20c1e5cd"
if [[ "$cann_env" != /usr/local/Ascend/cann-9.1.0/set_env.sh ]]; then
    TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR-$(basename "$(dirname "$cann_env")")"
fi
mkdir -p "$TORCH_EXTENSIONS_DIR"
export TRITON_CACHE_DIR="$(mktemp -d "/var/tmp/$name-triton.XXXXXXXX")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export HCCL_CONNECT_TIMEOUT=450 HCCL_HOST_SOCKET_PORT_RANGE=auto HCCL_NPU_SOCKET_PORT_RANGE=auto
unset NCCL_DETERMINISTIC FLASH_ATTENTION_DETERMINISTIC CLOSE_MATMUL_K_SHIFT CUBLAS_WORKSPACE_CONFIG
export HCCL_DETERMINISTIC=false HCCL_WHITELIST_DISABLE=1 TORCH_HCCL_ASYNC_ERROR_HANDLING=1
printf 'DETERMINISTIC_POLICY_BEFORE_WORKER=false\n'
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# This node capability is explicitly declared in the allocation file.
if [[ "${HP_PINNED_EXPANDABLE_SEGMENTS:-0}" == 1 ]]; then
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True,pin_memory_expandable_segments:True
fi
printf 'DSV41_ALLOCATOR_CONF=%s\n' "$PYTORCH_NPU_ALLOC_CONF"

if [[ "$NODE_RANK" == 0 ]]; then
    sha256sum -c "$prepared/inputs.sha256"
    python - "$prepared/recipe.yaml" "$steps" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as source:
    configured_steps = yaml.safe_load(source)["training"]["train_iters"]
if configured_steps != int(sys.argv[2]):
    raise ValueError("The frozen recipe and expected optimizer-update count differ")
PY
    sha256sum -c "$repo/.native-megamoe-verified.sha256" > "$experiment/logs/$name/native-payload-proof.txt"
    mkdir -p "$experiment/runs"
    mkdir "$DSV41_FULL_RUN_DIR"
    mkdir "$DSV41_FULL_RUN_DIR/receipts"
    touch "$DSV41_FULL_RUN_DIR/.ready"
else
    deadline=$((SECONDS + 150))
    until [[ -f "$DSV41_FULL_RUN_DIR/.ready" ]]; do
        if ((SECONDS >= deadline)); then echo "Full V4.1 recipe preflight timed out"; exit 1; fi
        sleep 1
    done
fi
export ASCEND_PROCESS_LOG_PATH="$DSV41_FULL_RUN_DIR/cann/node-$NODE_RANK"
export HYPER_PARALLEL_MODEL_INTEGRATION_OUTPUT_DIR="$DSV41_FULL_RUN_DIR/model-integration"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
cd "$repo"
worker="${DSV41_FULL_WORKER:-$experiment/S1/train-full-v41-vlm-worker.py}"
python -u "$root/direct-rank-launch.py" 29793 "$NPU_PER_NODE" "$worker" "$prepared/recipe.yaml" &
worker_pid=$!
trap 'kill -TERM "$worker_pid" 2>/dev/null || true' TERM INT
while kill -0 "$worker_pid" 2>/dev/null; do
    if compgen -G "$DSV41_FULL_RUN_DIR/failed-node-*" >/dev/null; then
        kill -TERM "$worker_pid" 2>/dev/null || true
        wait "$worker_pid" || true
        exit 1
    fi
    sleep 2
done
if wait "$worker_pid"; then
    printf 'FULL_V41_NODE_EXIT_OK node_rank=%s\n' "$NODE_RANK"
else
    status=$?
    printf '%s\n' "$status" > "$DSV41_FULL_RUN_DIR/failed-node-$NODE_RANK"
    exit "$status"
fi
