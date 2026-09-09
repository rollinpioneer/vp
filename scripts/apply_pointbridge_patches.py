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
    ROOT / "patches" / "pointbridge" / "0004-mimiclabs-delta-pose-contract.patch",
    ROOT / "patches" / "pointbridge" / "0005-mimiclabs-delta-pose-float32-identity-training.patch",
    ROOT / "patches" / "pointbridge" / "0006-mimiclabs-delta-pose-chunk-contract.patch",
)


def _run_patch(upstream: Path, patch: Path, *options: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "patch",
            *options,
            "--reject-file=-",
            "--no-backup-if-mismatch",
            "-p1",
            "-i",
            str(patch),
        ],
        cwd=upstream,
        text=True,
        capture_output=True,
        check=False,
    )


def _v1r_2k_contract_applied(upstream: Path) -> bool:
    dataset = upstream / "point_bridge" / "read_data" / "mimiclabs.py"
    agent = upstream / "point_bridge" / "agent" / "pb.py"
    if not all(path.exists() for path in (dataset, agent)):
        return False
    return (
        "from vico_point.action_contracts import encode_delta_label" in dataset.read_text(encoding="utf-8")
        and "from vico_point.action_contracts import decode_delta_command" in agent.read_text(encoding="utf-8")
    )


def _v1r_2k_chunk_contract_applied(upstream: Path) -> bool:
    dataset = upstream / "point_bridge" / "read_data" / "mimiclabs.py"
    agent = upstream / "point_bridge" / "agent" / "pb.py"
    if not all(path.exists() for path in (dataset, agent)):
        return False
    dataset_text = dataset.read_text(encoding="utf-8")
    agent_text = agent.read_text(encoding="utf-8")
    return (
        'if self._action_mode == "delta_pose":' in dataset_text
        and "self.all_time_actions_populated" in agent_text
        and "actions_populated = self.all_time_actions_populated[:, step]" in agent_text
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
        contract_applied = _v1r_2k_contract_applied(upstream)
        if patch.name.startswith("0005-") and contract_applied:
            print(f"already applied {patch.relative_to(ROOT)}")
            continue
        # 0005 replaces the normalization hunk introduced by 0004, so a
        # reverse dry-run of 0004 is no longer a valid applied-state check.
        if patch.name.startswith("0004-") and contract_applied:
            print(f"already applied {patch.relative_to(ROOT)} (superseded by 0005)")
            continue
        if patch.name.startswith("0006-") and _v1r_2k_chunk_contract_applied(upstream):
            print(f"already applied {patch.relative_to(ROOT)}")
            continue
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
