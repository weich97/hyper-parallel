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
"""Set disjoint process-local CPU affinity before importing the training runtime."""

import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import sys


def parse_cpu_list(text):
    result = set()
    for part in text.strip().split(','):
        if '-' in part:
            first, last = map(int, part.split('-'))
            result.update(range(first, last + 1))
        elif part:
            result.add(int(part))
    return result


def anonymous_pages():
    result = {}
    for line in Path('/proc/self/numa_maps').read_text().splitlines():
        if 'anon=' not in line:
            continue
        for node, count in re.findall(r'\bN(\d+)=(\d+)', line):
            result[node] = result.get(node, 0) + int(count)
    return result


def main():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    workers = int(os.environ['NPU_PER_NODE'])
    if workers != 8 or not 0 <= local_rank < workers:
        raise ValueError('Unexpected local training allocation')
    allowed = set(os.sched_getaffinity(0))
    nodes = []
    for path in sorted(Path('/sys/devices/system/node').glob('node[0-9]*'),
                       key=lambda value: int(value.name[4:])):
        cpus = sorted(parse_cpu_list((path / 'cpulist').read_text()) & allowed)
        if cpus:
            nodes.append((int(path.name[4:]), cpus))
    if not nodes or len(nodes) > workers:
        raise ValueError('NUMA topology cannot be mapped to this worker allocation')
    masks = []
    assignments = []
    for index in range(workers):
        node_index = index % len(nodes)
        slot = index // len(nodes)
        node, cpus = nodes[node_index]
        slots = len(range(node_index, workers, len(nodes)))
        first = len(cpus) * slot // slots
        last = len(cpus) * (slot + 1) // slots
        mask = set(cpus[first:last])
        if not mask or any(mask & previous for previous in masks):
            raise ValueError('Worker CPU masks are empty or overlap')
        masks.append(mask)
        assignments.append(node)
    os.sched_setaffinity(0, masks[local_rank])
    if set(os.sched_getaffinity(0)) != masks[local_rank]:
        raise ValueError('The process-local affinity was not applied')
    worker = Path('/xw-tree/full-v41/full-model/S1/020bf2eb-train-full-v41-nondeterministic.py')
    if hashlib.sha256(worker.read_bytes()).hexdigest() != '020bf2eb1abf2e3832bc900805c010ff171bd2394c0b0854db737f38b4e66be2':
        raise ValueError('The accepted training worker changed')
    target = Path(os.environ['DSV41_FULL_RUN_DIR']) / 'cpu-affinity'
    target.mkdir(exist_ok=True)
    receipt = target / f'rank-{rank:03d}.json'
    row = dict(rank=rank, local_rank=local_rank, node_rank=int(os.environ['NODE_RANK']),
               selected_numa_node=assignments[local_rank], selected_cpus=sorted(masks[local_rank]),
               allowed_cpu_count=len(allowed), available_numa_nodes=[node for node, _ in nodes],
               affinity_before_import=True, memory_binding='Unchanged default first-touch policy',
               accelerator_numa_proximity='Unknown in sysfs; no proximity claim',
               inherited_cpu_quota_unchanged=True, state='STARTED')
    receipt.write_text(json.dumps(row) + '\n')
    print('BALANCED_PROCESS_AFFINITY=' + json.dumps(row), flush=True)
    original_exit = os._exit

    def exit_with_receipt(code):
        final_code = code
        try:
            updates = int(os.environ['DSV41_FULL_STEPS'])
            training = target.parent / 'receipts' / f'rank-{rank:03d}.json'
            completed = json.loads(training.read_text()) if training.exists() else {}
            valid = (code == 0 and completed.get('completed_steps') == updates
                     and len(completed.get('steps', [])) == updates)
            row.update(state='COMPLETED' if valid else 'FAILED', exit_code=code,
                       completed_steps=completed.get('completed_steps'),
                       final_affinity=sorted(os.sched_getaffinity(0)),
                       anonymous_pages_by_numa=anonymous_pages(), page_size_bytes=os.sysconf('SC_PAGE_SIZE'))
            temporary = receipt.with_suffix('.tmp')
            temporary.write_text(json.dumps(row) + '\n')
            temporary.replace(receipt)
            if code == 0 and not valid:
                final_code = 1
        except Exception as error:
            print(f'AFFINITY_RECEIPT_FAILED={error!r}', file=sys.stderr, flush=True)
            final_code = 1
        original_exit(final_code)

    # The training worker exits with os._exit after destroying its process group.
    # Flush evidence before that exit without changing its teardown sequence.
    os._exit = exit_with_receipt
    sys.argv[0] = str(worker)
    runpy.run_path(str(worker), run_name='__main__')
    exit_with_receipt(0)


if __name__ == '__main__':
    main()
