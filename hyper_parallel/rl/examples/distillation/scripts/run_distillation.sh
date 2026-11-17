#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
cd "$repo_root"
config=${1:?Usage: run_distillation.sh CONFIG [--section.field=value ...]}
shift
export PYTHONPATH="$repo_root/hyper_parallel/rl${PYTHONPATH:+:$PYTHONPATH}"
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" \
  hyper_parallel/rl/train_rl.py "$config" "$@"
