#!/usr/bin/env bash
# Activate the digest-pinned, user-scoped CANN 9.2.1 experiment runtime.
runtime=/xw-tree/full-v41/runtime-cann921/install
test -f "$runtime/python-cann921/.ready"
test -f "$runtime/torch210-cpu-base/.ready"
test -f "$runtime/hp-training-python-base/.ready"
test -x "$runtime/build-tools-ninja-1.13.2/bin/ninja"
export PATH="$runtime/build-tools-ninja-1.13.2/bin:$PATH"
export PYTHONPATH="$runtime/python-cann921/site-packages:$runtime/torch210-cpu-base:$runtime/hp-training-python-base${PYTHONPATH:+:$PYTHONPATH}"
export DSV41_FULL_CANN_ENV=/usr/local/Ascend/cann-9.2.1/set_env.sh
export DSV41_GATE_CANN_ENV="$DSV41_FULL_CANN_ENV"
export DSV41_SMOKE_CANN_ENV="$DSV41_FULL_CANN_ENV" DSV41_SMOKE_TORCH_NPU=2.10.0.post7
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
source "$DSV41_FULL_CANN_ENV"
