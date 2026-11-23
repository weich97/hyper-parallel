"""Stage 5: Chain Stage 4 + patch host_table.host_lookup to skip per-step 256-rank stage_check barriers."""
import hashlib
from pathlib import Path
import runpy

from hyper_parallel.models.deepseek_v41.adapter.engram import host_lookup as host_lookup_mod
from hyper_parallel.models.deepseek_v41.adapter.engram import host_table as host_table_mod

PARENT = Path('/xw-tree/full-v41/opt4-observe-sync-and-foreach.py')
PARENT_SHA = '16285d2f068645e98f1f8d3d03a41141dddae05bae678564053cd4af6bc851a5'


def install():
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError('Stage 4 observer changed')
    runpy.run_path(str(PARENT))['install']()
    host_table_mod.host_lookup = host_lookup_mod.host_lookup
