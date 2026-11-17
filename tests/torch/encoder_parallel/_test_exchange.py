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
"""Gloo worker for TP gather, TP A2A, and parent A2A feature routes."""

from types import SimpleNamespace

import torch
import torch.distributed as dist

from hyper_parallel.distributed.encoder_parallel.contracts import EncoderWorkItem
from hyper_parallel.distributed.encoder_parallel.exchange import (
    FeatureExchangeMode,
    route_features,
    select_feature_exchange,
    unpack_records,
)
from hyper_parallel.distributed.encoder_parallel.planning import (
    DecoderSequenceLayout,
    assign_lpt,
    build_destinations,
)


class _WorldMesh:
    """Expose the feature-exchange mesh operations on the Gloo world group."""

    @staticmethod
    def size() -> int:
        """Return the process group size."""
        return dist.get_world_size()

    @staticmethod
    def get_local_rank() -> int:
        """Return this worker's group rank."""
        return dist.get_rank()

    @staticmethod
    def get_group() -> dist.ProcessGroup:
        """Return the world process group."""
        return dist.group.WORLD


def _items() -> tuple[EncoderWorkItem, ...]:
    return (
        EncoderWorkItem(0, 0, 576, 64, 96, 74, 0, 576, {}),
        EncoderWorkItem(1, 0, 256, 36, 32, 44, 576, 832, {}),
        EncoderWorkItem(2, 0, 144, 16, 200, 22, 832, 976, {}),
    )


def _check_route(*, cp_size: int, tp_size: int, encoder_cp_size: int,
                 sequence_parallel: bool, expected_mode: FeatureExchangeMode) -> None:
    """Compare transported records and producer gradients with exact owners."""
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    assert dist.get_world_size() == cp_size * tp_size
    context = SimpleNamespace(
        cp_size=cp_size, tp_size=tp_size, cp_rank=rank // tp_size,
        tp_rank=rank % tp_size, sequence_parallel=sequence_parallel,
    )
    items = _items()
    layouts = {
        peer: DecoderSequenceLayout.build(
            mesh_context=context, cp_local_length=256 // cp_size,
            cp_rank=peer // tp_size, tp_rank=peer % tp_size,
        ) for peer in range(cp_size * tp_size)
    }
    destinations = build_destinations(items, layouts)
    buckets = assign_lpt(items, cp_size * tp_size // encoder_cp_size, encoder_cp_size)
    producer_items = buckets[rank // encoder_cp_size] if rank % encoder_cp_size == 0 else ()
    features = [torch.full((item.output_rows, 2), float(item.item_id + 1)).requires_grad_()
                for item in producer_items]
    records = tuple(zip(producer_items, features))
    prototype = torch.zeros((1, 2), requires_grad=True)
    mode = select_feature_exchange(context)
    assert mode is expected_mode
    rows, received_items = route_features(
        mode=mode, producer_records=records, prototype=prototype,
        parent_rank=rank, destinations=destinations, buckets=buckets,
        encoder_cp_size=encoder_cp_size, exchange_mesh=_WorldMesh(),
    )
    received = unpack_records(rows, received_items)
    expected_ids = {item.item_id for item in items if rank in destinations[item.item_id]}
    assert {item.item_id for item, _ in received} == expected_ids
    for item, feature in received:
        assert feature.shape == (item.output_rows, 2)
        assert torch.equal(feature, torch.full_like(feature, float(item.item_id + 1)))
    (sum((feature.sum() for _, feature in received), rows.sum() * 0.0)).backward()
    for item, feature in zip(producer_items, features):
        expected_gradient = torch.full_like(feature, float(len(destinations[item.item_id])))
        assert torch.equal(feature.grad, expected_gradient), f"rank={rank}; item={item.item_id}"
    if rank == 0:
        print(f"{mode.value} records and producer gradients passed", flush=True)
    dist.destroy_process_group()


def test_tp_gather() -> None:
    """CP1 without SP sends each producer record to every TP rank."""
    _check_route(cp_size=1, tp_size=4, encoder_cp_size=1, sequence_parallel=False,
                 expected_mode=FeatureExchangeMode.TP_VARIABLE_ALL_GATHER)


def test_tp_a2a() -> None:
    """CP1 with SP sends records only to intersecting TP ranks."""
    _check_route(cp_size=1, tp_size=4, encoder_cp_size=1, sequence_parallel=True,
                 expected_mode=FeatureExchangeMode.TP_VARIABLE_A2A)


def test_parent_nosp() -> None:
    """CP2 without SP uses parent A2A with explicit TP fan-out."""
    _check_route(cp_size=2, tp_size=4, encoder_cp_size=2, sequence_parallel=False,
                 expected_mode=FeatureExchangeMode.PARENT_VARIABLE_A2A)
