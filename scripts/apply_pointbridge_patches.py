#!/usr/bin/env python3
"""Apply ViCo-Point compatibility patches to the pinned Point Bridge checkout."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COMMIT = "5d567a62d62b5a97c5960d45024e065349680cda"
DEFAULT_UPSTREAM = ROOT / "third_party" / "pointbridge"
PATCHES = (
    ROOT / "patches" / "pointbridge" / "0001-mimiclabs-saved-state-generator.patch",
    ROOT / "patches" / "pointbridge" / "0002-mujoco23-mesh-scale-compat.patch",
    ROOT / "patches" / "pointbridge" / "0003-mujoco23-mesh-path-compat.patch",
)


def _run_patch(upstream: Path, patch: Path, *options: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["patch", *options, "-p1", "-i", str(patch)],
        cwd=upstream,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    args = parser.parse_args()
    upstream = args.upstream.resolve()

    lock_path = upstream / ".vico_source_lock"
    if not lock_path.exists():
        parser.error(f"missing source lock: {lock_path}")
    actual_commit = lock_path.read_text(encoding="utf-8").strip().removeprefix("commit=")
    if actual_commit != EXPECTED_COMMIT:
        parser.error(f"expected Point Bridge {EXPECTED_COMMIT}, found {actual_commit}")

    for patch in PATCHES:
        result = _run_patch(upstream, patch, "--forward", "--batch")
        if result.returncode == 0:
            print(f"applied {patch.relative_to(ROOT)}")
            continue
        reverse = _run_patch(upstream, patch, "--reverse", "--dry-run", "--batch")
        if reverse.returncode == 0:
            print(f"already applied {patch.relative_to(ROOT)}")
            continue
        print(result.stdout, end="")
        print(result.stderr, end="")
        return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
