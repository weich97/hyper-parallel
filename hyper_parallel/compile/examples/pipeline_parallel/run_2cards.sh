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

# VPP (Interleaved 1F1B) Training - Single Node 2 Cards
#
# Pure PP contract: world_size == pp_degree. VPP adds pp_chunks virtual
# stages per physical rank, so this launcher runs 2 ranks x 2 chunks == 4
# virtual pipeline stages.
#
# Configuration:
#   - Total cards: 2 (physical PP ranks)
#   - Virtual stages: 4 (pp_chunks=2)
#   - Schedule: VPP / Interleaved 1F1B
#   - FSDP: disabled (v1 pure-PP; hybrid requires a mesh with a pp dim)
#
# Usage:
#   bash run_2cards.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Starting VPP training"
echo "Configuration: ranks=2, virtual stages=4, schedule=vpp, FSDP=off"
echo "=========================================="

torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=29511 \
    train.py \
    --config config.yaml

echo "=========================================="
echo "Training completed!"
echo "=========================================="
