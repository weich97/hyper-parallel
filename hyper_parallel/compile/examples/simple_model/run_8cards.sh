#!/bin/bash
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Simple Model Training - Single Node 8 Cards
# 
# Configuration:
#   - Total cards: 8
#   - Data Parallel (FSDP): 8
#
# Usage:
#   bash run_8cards.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYPER_PARALLEL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="$(cd "${HYPER_PARALLEL_ROOT}/.." && pwd)"

CONFIG="${SCRIPT_DIR}/config.yaml"
TRAIN_ENTRY="${REPO_ROOT}/scripts/train_lm.py"

echo "=========================================="
echo "Starting 8-card distributed training"
echo "Configuration: FSDP=8"
echo "Entry: ${TRAIN_ENTRY}"
echo "Config: ${CONFIG}"
echo "=========================================="

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=29500 \
    "${TRAIN_ENTRY}" \
    "${CONFIG}"

echo "=========================================="
echo "Training completed!"
echo "=========================================="
