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
"""Verify and stage the frozen DeepSeek V4.1 Flash training sources."""

import argparse
import hashlib
from pathlib import Path, PurePosixPath


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "frozen"


def source_files() -> list[tuple[str, Path]]:
    """Read the source manifest and verify every frozen byte."""
    files = []
    seen = set()
    for line in (HERE / "sources.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or name in seen:
            raise ValueError(f"Invalid or repeated source path {name}")
        seen.add(name)
        source = ARCHIVE / relative
        if ARCHIVE not in source.resolve().parents or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source differs from its manifest {name}")
        files.append((name, source))
    if not files:
        raise ValueError("The frozen source manifest is empty")
    return files


def main() -> None:
    """Validate the package and optionally populate a caller-owned runtime tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, help="Directory mounted at /xw-tree/full-v41")
    parser.add_argument("--stage", action="store_true", help="Populate missing source files")
    args = parser.parse_args()
    if args.stage and args.runtime_root is None:
        parser.error("--stage requires --runtime-root")

    files = source_files()
    pending = []
    if args.runtime_root is not None:
        root = args.runtime_root.resolve()
        for name, source in files:
            destination = root / name
            if root not in destination.resolve().parents:
                raise ValueError(f"Runtime path escapes the selected directory {name}")
            payload = source.read_bytes()
            if destination.exists():
                if destination.read_bytes() != payload:
                    raise ValueError(f"Runtime source already has different content {name}")
            elif args.stage:
                pending.append((destination, payload))
            else:
                raise FileNotFoundError(f"Runtime source is missing {name}")

    for destination, payload in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(payload)
    print(f"Verified {len(files)} source files and staged {len(pending)} files")


if __name__ == "__main__":
    main()
