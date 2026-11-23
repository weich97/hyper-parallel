"""Check the all-rank optimizer collective plan before native communicator creation."""
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import runpy
import time

import torch.distributed as dist
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer import muon_shard


def dump(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value) + "\n")
    temporary.replace(path)


def validate(rows, size):
    calls = {}
    edges = {}
    for row in rows:
        rank = row["rank"]
        previous = None
        counts = {}
        for call in row["groups"]:
            domain = (tuple(call["ranks"]), call["group_name"])
            ordinal = counts.get(domain, 0)
            counts[domain] = ordinal + 1
            key = (domain, ordinal)
            calls.setdefault(key, []).append((rank, call))
            edges.setdefault(key, set())
            if previous is not None:
                edges[previous].add(key)
            previous = key
    for key, peers in calls.items():
        expected = set(key[0][0])
        if {rank for rank, _ in peers} != expected:
            raise ValueError(f"Missing/mismatched optimizer group participants: {key}")
        signatures = {json.dumps(call["parameters"], sort_keys=True) for _, call in peers}
        if len(signatures) != 1:
            raise ValueError(f"Parameter order/shape mismatch for optimizer group: {key}")
    incoming = {key: 0 for key in edges}
    for successors in edges.values():
        for key in successors:
            incoming[key] += 1
    ready = [key for key, degree in incoming.items() if degree == 0]
    visited = 0
    while ready:
        key = ready.pop()
        visited += 1
        for successor in edges[key]:
            incoming[successor] -= 1
            if incoming[successor] == 0:
                ready.append(successor)
    if visited != len(edges):
        raise ValueError("Optimizer subgroup invocation order contains a distributed cycle")
    return {"status": "PASS", "ranks": size, "collective_domains": len(calls), "acyclic": True}


def install():
    parent = Path("/xw-tree/full-v41/613f2e26-observe-npu-only-fill-phases.py")
    if hashlib.sha256(parent.read_bytes()).hexdigest() != "613f2e26bbd52f8146c85412c1a191af86d54f15ae7e0b35b3182a5e661843b8":
        raise ValueError("Sequential training policy changed")
    runpy.run_path(str(parent))["install"]()
    original = Muon._build_param_shard_metadata
    original_gather = muon_shard._allgather_shapes_for_group
    invocation = 0
    gather_index = 0
    output = None

    @wraps(original)
    def coordinate(optimizer):
        nonlocal invocation, output, gather_index
        rank, size = dist.get_rank(), dist.get_world_size()
        output = Path(os.environ["DSV41_FULL_RUN_DIR"]) / f"muon-init-{invocation}"
        invocation += 1
        gather_index = 0
        output.mkdir(exist_ok=True)
        groups = []
        for _, hsdp_groups in optimizer._hsdp_grouping.values():
            for group in hsdp_groups:
                if group.layout_spec is None or not group.layout_spec.shard_axes:
                    continue
                for pg in group.shard_pgs:
                    if pg is None or dist.get_world_size(pg) <= 1:
                        continue
                    groups.append({"ranks": dist.get_process_group_ranks(pg),
                                   "group_name": str(getattr(pg, "group_name", "unknown")),
                                   "parameters": [{"name": getattr(item.param, "model_name", str(item.index)),
                                                   "shape": list(item.param.shape)} for item in group.records]})
        dump(output / f"rank-{rank:03d}.json", {"rank": rank, "groups": groups,
             "host_ip": os.environ.get("HCCL_IF_IP"), "interface": os.environ.get("HCCL_SOCKET_IFNAME")})
        deadline = time.monotonic() + 420
        ready = output / "agreement.json"
        if rank == 0:
            paths = [output / f"rank-{index:03d}.json" for index in range(size)]
            while not all(path.exists() for path in paths):
                if time.monotonic() > deadline:
                    dump(ready, {"status": "FAIL", "error": "Ranks failed to enter optimizer metadata initialization"})
                    break
                time.sleep(0.25)
            else:
                try:
                    dump(ready, validate([json.loads(path.read_text()) for path in paths], size))
                except ValueError as error:
                    dump(ready, {"status": "FAIL", "error": str(error)})
        while not ready.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Optimizer collective-plan agreement did not complete")
            time.sleep(0.25)
        agreement = json.loads(ready.read_text())
        if agreement["status"] != "PASS":
            raise RuntimeError(agreement["error"])
        print("MUON_INIT_PLAN_AGREEMENT=" + json.dumps(agreement), flush=True)
        return original(optimizer)

    @wraps(original_gather)
    def gather(shapes, pg, size, device):
        nonlocal gather_index
        index = gather_index
        gather_index += 1
        rank = dist.get_rank()
        path = output / f"call-r{rank:03d}-i{index:03d}.json"
        row = {"rank": rank, "index": index, "ranks": dist.get_process_group_ranks(pg) if pg else [rank],
               "group_name": str(getattr(pg, "group_name", "none")), "shapes": shapes, "state": "ENTERED"}
        dump(path, row)
        result = original_gather(shapes, pg, size, device)
        row["state"] = "COMPLETED"
        dump(path, row)
        return result

    Muon._build_param_shard_metadata = coordinate
    muon_shard._allgather_shapes_for_group = gather
