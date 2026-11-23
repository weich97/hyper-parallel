"""Record actual sampled draft geometry without synchronous training probes."""
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import runpy

from hyper_parallel.models.deepseek_v41.dspark import DeepseekV41DSpark, _document_geometry
from hyper_parallel.trainer.vlm_trainer import VLMTrainer


def union_size(intervals):
    end, size = -1, 0
    for left, right in sorted(intervals):
        size += max(0, right - max(left, end))
        end = max(end, right)
    return size


def install():
    parent = Path('/xw-tree/full-v41/c79c1f4d-observe-npu-only-fill-coordinated.py')
    if hashlib.sha256(parent.read_bytes()).hexdigest() != 'c79c1f4dfaabe26b9b77e70a16b858f5e7e3a326cf8cc85c97c1d32a48d4f8dc':
        raise ValueError('Best full-model runtime policy changed')
    runpy.run_path(str(parent))['install']()
    original, end = DeepseekV41DSpark.forward, VLMTrainer.on_train_end
    events = []

    @wraps(original)
    def sampled(module, target_hiddens, input_ids, *args, **kwargs):
        result = original(module, target_hiddens, input_ids, *args, **kwargs)
        selection = module.last_anchor_selection
        if input_ids.shape != (1, 16384) or module.anchor_budget != 1024 or selection is None:
            raise ValueError('Unexpected sampled workload geometry')
        if len(module.stages) != 3 or any(stage.self_attn.attention_backend != 'flash_v3' for stage in module.stages):
            raise ValueError('All three released draft stages must use the reference FlashAttention path')
        document_positions, _ = _document_geometry(input_ids.shape[1], kwargs.get('sequence_boundaries'))
        positions = selection.positions[:selection.valid_count]
        spans = [(index - min(document_positions[index] + 1, 128) + 1, index + 1) for index in positions]
        row = dict(update=len(events) + 1, positions=positions, population=selection.population,
                   active_anchors=selection.valid_count, dispatched_anchors=len(selection.positions),
                   block_size=module.block_size, window=128, main_kv_tokens=union_size(spans),
                   main_projection_executed_rows=input_ids.shape[1],
                   attention_pairs=sum(module.block_size * (right - left + module.block_size) for left, right in spans),
                   loss_inverse_inclusion_probability=selection.weight,
                   supervised_tokens=result[1]['dspark_supervised_tokens'].detach(),
                   full_supervised_tokens=result[1]['dspark_full_supervised_tokens'].detach())
        events.append(row)
        if len(events) == 1:
            print('DSPARK_ANCHOR_SAMPLING=budget1024 inverse_probability_weighted full_backbone_context three_stages', flush=True)
        return result

    @wraps(end)
    def finished(trainer):
        result = end(trainer)
        if len(events) != trainer.base.state.global_step:
            raise ValueError('Sampling occurred more or less than once per update')
        for row in events:
            for key in ('supervised_tokens', 'full_supervised_tokens'):
                row[key] = int(row[key].cpu())
        rank = int(os.environ['RANK'])
        directory = Path(os.environ['DSV41_FULL_RUN_DIR']) / 'dspark-anchors'
        directory.mkdir(exist_ok=True)
        path = directory / ('rank-%03d.json' % rank)
        path.write_text(json.dumps(dict(rank=rank, budget=1024, events=events)) + '\n')
        return result

    DeepseekV41DSpark.forward, VLMTrainer.on_train_end = sampled, finished
