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
"""CPU contracts for natural-document prefix dispatch and owner gradient sums."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.distributed.context_parallel import sequence_dispatch


def _document_oracle(boundaries, local_length, ratio):
    """Enumerate complete raw blocks independently of compressed offset helpers."""
    size = boundaries[-1] // local_length
    blocks = [(document, end) for document, (start, last) in enumerate(zip(boundaries, boundaries[1:]))
              for end in range(start + ratio, last + 1, ratio)]
    owned = [tuple(index for index, (_, end) in enumerate(blocks) if (end - 1) // local_length == rank)
             for rank in range(size)]
    demands = []
    for rank in range(size):
        first, last = rank * local_length, (rank + 1) * local_length
        documents = {document for document, (start, end) in enumerate(zip(boundaries, boundaries[1:]))
                     if max(start, first) < min(end, last)}
        demands.append(tuple(index for index, (document, end) in enumerate(blocks)
                             if document in documents and end <= last))
    return owned, demands, len(blocks)


class _CopyOnWait:
    """Model asynchronous visibility without any device or process group."""

    def __init__(self, output: torch.Tensor, values: torch.Tensor) -> None:
        """Retain the output buffer and the expected remote payload."""
        self.output = output
        self.values = values

    def wait(self) -> None:
        """Publish receives only after the tested implementation waits."""
        self.output.copy_(self.values)


class TestSequenceDispatch(unittest.TestCase):
    """Exercise nonuniform and empty ownership with a raw-document oracle."""

    CASES = (
        ((0, 3, 6, 8), 4, 2),
        ((0, 3, 9, 12), 3, 2),
        ((0, 1, 6, 7, 12), 3, 2),
        ((0, 1, 2, 4), 2, 2),
        ((0, 1, 4, 5, 8), 1, 2),
        ((0, 1, 2, 3, 4), 1, 2),
        ((0, 3, 6, 8), 4, 1),
        ((0, 4, 16), 4, 2),
    )

    def test_prefix_routes_match_complete_raw_blocks(self):
        """Prefix rows and peer splits agree with document-local completed raw blocks."""
        for boundaries, local_length, ratio in self.CASES:
            owned, demands, _ = _document_oracle(boundaries, local_length, ratio)
            size = len(owned)
            routes = [sequence_dispatch.document_prefix_route(boundaries, local_length, size, rank, ratio)
                      for rank in range(size)]
            for rank, route in enumerate(routes):
                with self.subTest(boundaries=boundaries, local_length=local_length, ratio=ratio, rank=rank):
                    self.assertEqual(tuple(range(route.global_start, route.global_end)), demands[rank])
                    self.assertEqual(route.local_length, len(owned[rank]))
                    self.assertEqual(route.local_output_start,
                                     sum(index < sum(map(len, owned[:rank])) for index in demands[rank]))
                    for peer in range(size):
                        expected_send = tuple(index for index in owned[rank] if index in demands[peer])
                        expected_send = expected_send if rank != peer else ()
                        self.assertEqual(route.input_splits[peer], len(expected_send))
                        self.assertEqual(route.input_splits[peer], routes[peer].output_splits[rank])
                        offset = route.send_offsets[peer]
                        self.assertEqual(owned[rank][offset:offset + route.input_splits[peer]], expected_send)
                    expected_exchange = any(set(demand) - set(owner) for demand, owner in zip(demands, owned))
                    self.assertEqual(route.needs_exchange, expected_exchange)

    def _check_dispatch_gradients(self, boundaries, local_length, ratio, rank):
        """Check one rank against all consumers' independent loss gradients."""
        owned, demands, length = _document_oracle(boundaries, local_length, ratio)
        size = len(owned)
        bank = torch.arange(length * 2, dtype=torch.float64).reshape(length, 1, 2)
        gradients = [torch.arange(len(demand) * 2, dtype=torch.float64).reshape(-1, 1, 2) + 100 * (peer + 1)
                     for peer, demand in enumerate(demands)]
        route = sequence_dispatch.document_prefix_route(boundaries, local_length, size, rank, ratio)
        local = bank[list(owned[rank])].movedim(0, 1).clone().requires_grad_(True)
        calls = []

        def values(indices: tuple[int, ...], peer: int | None = None) -> torch.Tensor:
            """Select independent canonical values or consumer gradients."""
            if peer is None:
                return bank[list(indices)]
            rows = [demands[peer].index(index) for index in indices]
            return gradients[peer][rows]

        def exchange(output: torch.Tensor, send: torch.Tensor, output_split_sizes: list[int],
                     input_split_sizes: list[int], group: None, async_op: bool) -> _CopyOnWait:
            """Check actual payloads and delay the mock receive until wait."""
            self.assertIsNone(group)
            self.assertTrue(async_op)
            backward = bool(calls)
            outgoing, incoming = [], []
            for peer in range(size):
                sent = tuple(index for index in owned[rank] if index in demands[peer] and peer != rank)
                received = tuple(index for index in owned[peer] if index in demands[rank] and peer != rank)
                outgoing.append(values(received, rank) if backward else values(sent))
                incoming.append(values(sent, peer) if backward else values(received))
            self.assertEqual(input_split_sizes, [part.shape[0] for part in outgoing])
            self.assertEqual(output_split_sizes, [part.shape[0] for part in incoming])
            torch.testing.assert_close(send, torch.cat(outgoing), rtol=0, atol=0)
            calls.append(backward)
            return _CopyOnWait(output, torch.cat(incoming))

        with patch.object(sequence_dispatch.dist, "all_to_all_single", side_effect=exchange) as collective:
            handle = sequence_dispatch.async_sequence_dispatch(local, 1, route, None)
            output = handle.wait()
            torch.testing.assert_close(output, bank[list(demands[rank])].movedim(0, 1), rtol=0, atol=0)
            (output * gradients[rank].movedim(0, 1)).sum().backward()
            self.assertEqual(collective.call_count, 2 if route.needs_exchange else 0)
            with self.assertRaisesRegex(RuntimeError, "cannot be waited twice"):
                handle.wait()
        expected = bank.new_zeros((len(owned[rank]), 1, 2))
        for row, index in enumerate(owned[rank]):
            for peer, demand in enumerate(demands):
                if index in demand:
                    expected[row] += gradients[peer][demand.index(index)]
        torch.testing.assert_close(local.grad, expected.movedim(0, 1), rtol=0, atol=0)

    def test_dispatch_and_backward_preserve_nonuniform_owners(self):
        """Forward restores bank rows; backward SUMs all consumers, including zero owners."""
        for boundaries, local_length, ratio in self.CASES:
            for rank in range(boundaries[-1] // local_length):
                with self.subTest(boundaries=boundaries, local_length=local_length, ratio=ratio, rank=rank):
                    self._check_dispatch_gradients(boundaries, local_length, ratio, rank)

    def test_raw_halo_uses_raw_document_boundaries(self):
        """A raw halo remains distinct from compact key IDs for odd-length documents."""
        route = sequence_dispatch.sliding_window_route((0, 3, 6, 8), 4, 2, 1, 128)
        self.assertEqual((route.global_start, route.global_end), (3, 8))
        self.assertEqual(route.output_splits, (1, 0))
        self.assertEqual(route.local_length, 4)

    def test_invalid_owner_intervals_are_rejected_before_launch(self):
        """Reject gaps, invalid ranges, and local lengths inconsistent with canonical owners."""
        cases = (
            (1, 0, ((0, 1), (1, 3)), (0, 1)),
            (1, 0, ((0, 1), (0, 1)), (0, 2, 1)),
            (1, 0, ((0, 0), (1, 3)), (0, 1, 3)),
            (1, 0, ((0, 1), (0, 4)), (0, 1, 3)),
            (2, 0, ((0, 1), (1, 3)), (0, 1, 3)),
            (1, 2, ((0, 1), (1, 3)), (0, 1, 3)),
        )
        for local_length, rank, intervals, owners in cases:
            with self.subTest(owners=owners, rank=rank, intervals=intervals), self.assertRaises(ValueError):
                sequence_dispatch.interval_route(local_length, rank, intervals, owner_offsets=owners)
        with self.assertRaisesRegex(ValueError, "physical layout"):
            route = sequence_dispatch.document_prefix_route((0, 3, 6, 8), 4, 2, 0, 2)
            sequence_dispatch.async_sequence_dispatch(torch.empty(1, 2, 2), 1, route, None)
