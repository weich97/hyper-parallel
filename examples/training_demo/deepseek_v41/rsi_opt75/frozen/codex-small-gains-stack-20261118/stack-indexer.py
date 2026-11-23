"""Independent small-gain candidates, with one opt73 parent installation."""
from hashlib import sha256
import os
from pathlib import Path
import runpy


ROOT = Path('/xw-tree/full-v41/codex-small-gains-stack-20261118')
CHECKSUMS = {
    'part73.py': '284442b303574979d05ad80b9dacb57f102188692b304c3001baab88a09a1485',
    'part616.py': '708d73a36c85d3165c5bc680284d22e6e12d5eeacee534e2d2c3813e6dd1c3f8',
    'part756.py': 'e633884b3fc109621489beff82c51f87cb756c6b09dce8a02ca9c727755d4e7b',
    'part766.py': '9b0d5ccbfa110429a6978645e027969b915fa45b46c6b760309945fdc4f5e6f2',
}


def install():
    parts = os.environ.get("DSV41_SMALL_GAIN_PARTS", "73,616,756,766").split(",")
    if not parts or parts[0] != "73" or len(set(parts)) != len(parts):
        raise ValueError(f"Unexpected small-gain configuration: {parts}")
    for part in parts:
        name = f"part{part}.py"
        script = ROOT / name
        if sha256(script.read_bytes()).hexdigest() != CHECKSUMS[name]:
            raise ValueError(f"Unexpected observer checksum: {name}")
        runpy.run_path(str(script))["install"]()
