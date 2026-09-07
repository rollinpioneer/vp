#!/usr/bin/env python3
"""Compare frozen-state CPU/CUDA actions or emit an auditable CUDA blocker."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from state_utils import load_state_index


def probe_cuda() -> dict[str, Any]:
    import torch

    try:
        process = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        nvidia_smi = {
            "returncode": process.returncode,
            "stdout": process.stdout.strip(),
            "stderr": process.stderr.strip(),
        }
    except Exception as exc:
        nvidia_smi = {
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }
    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "nvidia_smi": nvidia_smi,
    }


def load_runner_records(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: dict[str, dict[str, Any]] = {}
    for record in payload.get("records", []):
        records[str(record["scenario_id"])] = record
    return records


def compare(
    state_ids: list[str], cpu_path: Path, cuda_path: Path, tolerance: float
) -> list[dict[str, Any]]:
    cpu = load_runner_records(cpu_path)
    cuda = load_runner_records(cuda_path)
    records = []
    for scenario_id in state_ids:
        cpu_path_record = cpu[scenario_id]["paths"]["official"]
        cuda_path_record = cuda[scenario_id]["paths"]["official"]
        cpu_actions = np.asarray(cpu_path_record["first_20_actions"], dtype=np.float64)
        cuda_actions = np.asarray(cuda_path_record["first_20_actions"], dtype=np.float64)
        max_difference = (
            float(np.max(np.abs(cpu_actions - cuda_actions)))
            if cpu_actions.shape == cuda_actions.shape and cpu_actions.size
            else None
        )
        records.append(
            {
                "scenario_id": scenario_id,
                "initial_state_cpu": cpu_path_record["initial_state_sha256"],
                "initial_state_cuda": cuda_path_record["initial_state_sha256"],
                "initial_state_equal": cpu_path_record["initial_state_sha256"]
                == cuda_path_record["initial_state_sha256"],
                "first_action_max_abs_diff": (
                    float(np.max(np.abs(cpu_actions[0] - cuda_actions[0])))
                    if cpu_actions.shape == cuda_actions.shape and len(cpu_actions)
                    else None
                ),
                "first_20_action_max_abs_diff": max_difference,
                "success_cpu": int(cpu_path_record["success"]),
                "success_cuda": int(cuda_path_record["success"]),
                "passed": (
                    max_difference is not None
                    and max_difference <= tolerance
                    and cpu_path_record["success"] == cuda_path_record["success"]
                    and cpu_path_record["initial_state_sha256"]
                    == cuda_path_record["initial_state_sha256"]
                ),
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-index", type=Path, required=True)
    parser.add_argument("--cpu-results", type=Path)
    parser.add_argument("--cuda-results", type=Path)
    parser.add_argument("--deployment-device", choices=("cpu", "cuda"))
    parser.add_argument("--action-tolerance", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    state_ids = list(load_state_index(args.state_index.resolve()))
    hardware = probe_cuda()
    blockers: list[str] = []
    records: list[dict[str, Any]] = []
    if not hardware["torch_cuda_available"]:
        blockers.append("torch.cuda.is_available() is false")
    if hardware["nvidia_smi"]["returncode"] != 0:
        blockers.append("nvidia-smi cannot access an NVIDIA driver")
    if not blockers:
        if args.cpu_results is None or args.cuda_results is None:
            blockers.append("both --cpu-results and --cuda-results are required")
        else:
            records = compare(
                state_ids,
                args.cpu_results.resolve(),
                args.cuda_results.resolve(),
                args.action_tolerance,
            )
    if blockers:
        status = "blocked_unavailable_cuda" if not hardware["torch_cuda_available"] else "blocked_missing_results"
    else:
        status = "passed" if records and all(record["passed"] for record in records) else "failed"
    deployment_device_protocol_frozen = bool(
        args.deployment_device == "cpu"
        or (args.deployment_device == "cuda" and hardware["torch_cuda_available"])
    )
    result = {
        "stage": "V1-R.2F",
        "audit": "cpu_cuda_parity",
        "status": status,
        "scenarios_requested": len(state_ids),
        "action_tolerance": args.action_tolerance,
        "hardware": hardware,
        "blockers": blockers,
        "deployment_device": args.deployment_device,
        "deployment_device_protocol_frozen": deployment_device_protocol_frozen,
        "formal_evaluation_device_constraint": (
            f"all formal rollouts must use {args.deployment_device}"
            if deployment_device_protocol_frozen
            else None
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# V1-R.2F CPU/CUDA Parity\n\n"
        f"状态：`{status}`。请求冻结场景：{len(state_ids)}。\n\n"
        f"`torch.cuda.is_available()`：`{hardware['torch_cuda_available']}`；"
        f"CUDA device count：`{hardware['torch_cuda_device_count']}`；"
        f"`nvidia-smi` return code：`{hardware['nvidia_smi']['returncode']}`。\n\n"
        f"正式评测设备协议：`{args.deployment_device or 'unresolved'}`；"
        f"冻结：`{deployment_device_protocol_frozen}`。\n\n"
        + ("阻塞原因：\n" + "\n".join(f"- {item}" for item in blockers) + "\n" if blockers else f"通过场景：{sum(record['passed'] for record in records)}/{len(records)}。\n"),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0 if status in {"passed", "blocked_unavailable_cuda"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
