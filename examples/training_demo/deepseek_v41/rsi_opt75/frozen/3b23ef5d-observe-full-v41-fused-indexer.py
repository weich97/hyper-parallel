"""Record native Indexer and KL dispatch without tensor probes or synchronization."""

from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import runpy

from hyper_parallel.components.modules import shared_compressed_dsa_attention as attention
from hyper_parallel.trainer.vlm_trainer import VLMTrainer


def install():
    """Keep the accepted native reduction mode and record actual fused calls."""
    root = Path('/xw-tree/full-v41')
    gate = os.environ['DSV41_FULL_MODE'] == 'gate'
    parent = (root / 'full-model/S1/2fd2889b-diagnose-full-v41-mhc-reduction-order.py'
              if gate else root / 'configure-full-v41-ordered-mhc.py')
    runpy.run_path(str(parent))['install']()
    native_kl_setting = os.environ['DSV41_FULL_EXPECT_FUSED_KL']
    if native_kl_setting not in ('0', '1'):
        raise ValueError('Native KL expectation must be explicitly declared')
    expected_fused_kl = native_kl_setting == '1'
    counts = {'native_indexer_calls': 0, 'native_kl_calls': 0, 'reference_kl_calls': 0}
    for attribute, key in (('_run_lightning_indexer', 'native_indexer_calls'),
                           ('sparse_lightning_indexer_kl_loss', 'native_kl_calls')):
        original = getattr(attention, attribute)

        def wrap(function, counter):
            @wraps(function)
            def counted(*args, **kwargs):
                result = function(*args, **kwargs)
                counts[counter] += 1
                return result
            return counted

        setattr(attention, attribute, wrap(original, key))
    reference_forward = attention._SharedCompressedIndexerKLLoss.forward

    @wraps(reference_forward)
    def reference_kl(*args, **kwargs):
        result = reference_forward(*args, **kwargs)
        counts['reference_kl_calls'] += 1
        return result

    attention._SharedCompressedIndexerKLLoss.forward = staticmethod(reference_kl)
    original_end = VLMTrainer.on_train_end

    @wraps(original_end)
    def end(trainer):
        result = original_end(trainer)
        required_kl = 'native_kl_calls' if expected_fused_kl else 'reference_kl_calls'
        unused_kl = 'reference_kl_calls' if expected_fused_kl else 'native_kl_calls'
        if min(counts['native_indexer_calls'], counts[required_kl]) <= 0 or counts[unused_kl]:
            raise RuntimeError(f'Missing fused Indexer/KL execution: {counts}')
        rank = int(os.environ['RANK'])
        report = {'rank': rank, 'phase': os.environ['DSV41_FULL_MODE'],
                  'completed_steps': trainer.base.state.global_step, **counts,
                  'expected_fused_kl': expected_fused_kl,
                  'added_tensor_probes': False, 'added_synchronization': False,
                  'observer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        directory = Path(os.environ['DSV41_FULL_RUN_DIR']) / 'fused-indexer'
        directory.mkdir(exist_ok=True)
        target = directory / f'rank-{rank:03}.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(report) + '\n', encoding='utf-8')
        temporary.replace(target)
        print('FULL_V41_FUSED_INDEXER_ENGAGED ' + json.dumps(report), flush=True)
        return result

    VLMTrainer.on_train_end = end
