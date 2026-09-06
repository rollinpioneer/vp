#!/usr/bin/env python3
"""Reject oversized Git-tracked files before upload."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT_BYTES = 10 * 1024 * 1024


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
    )
    return [ROOT / name.decode("utf-8") for name in output.split(b"\0") if name]


def main() -> int:
    oversized: list[tuple[Path, int]] = []
    checked = 0
    for path in tracked_files():
        if not path.is_file():
            continue
        checked += 1
        size = path.stat().st_size
        if size > LIMIT_BYTES:
            oversized.append((path.relative_to(ROOT), size))

    if oversized:
        print(f"ERROR: tracked files exceed {LIMIT_BYTES} bytes:")
        for path, size in oversized:
            print(f"  {size:>12}  {path}")
        return 1

    print(f"OK: {checked} tracked files are each <= {LIMIT_BYTES} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
