#!/usr/bin/env python3
"""Run the one authorized B1-2K-20 seed-0 Point Bridge training job.

The default path is deliberately strict: it requires CUDA and the frozen
artifact identities, then invokes the upstream ``point_bridge/train.py``.
Use ``--smoke`` for a short, CPU-compatible technical check in a separate
output directory; smoke output is never a scientific result.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from seed0_contract import (  # noqa: E402
    load_yaml,
    sha256,
    training_overrides,
    validate_frozen_contract,
)


def _gpu_info() -> dict[str, object]:
    try:
        import torch

        return {
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:  # pragma: no cover - environment diagnostic
        return {"error": f"{type(exc).__name__}: {exc}"}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_smoke(
    python: str,
    upstream: Path,
    output: Path,
    overrides: list[str],
    env: dict[str, str],
) -> int:
    command = [python, "-u", str(upstream / "point_bridge/train.py"), *overrides]
    print("SMOKE COMMAND:", " ".join(command), flush=True)
    return subprocess.call(command, cwd=upstream, env=env)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/v1r/configs/b1_2k_20_seed0.yaml")
    parser.add_argument("--upstream", type=Path, default=Path("/home/xushijie/vico-point/third_party/pointbridge"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/v1r/training")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=20)
    args = parser.parse_args()

    config = args.config.resolve()
    upstream = args.upstream.resolve()
    freeze = validate_frozen_contract(config, upstream)
    if args.smoke and args.smoke_steps < 20:
        parser.error("--smoke-steps must be at least 20")
    if not args.smoke and args.device != "cuda":
        parser.error("formal seed-0 training is frozen to CUDA; use --smoke for CPU")

    if not args.smoke:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; formal seed-0 training was not started")

    experiment = "v1r_b1_2k_20_seed0_smoke" if args.smoke else "v1r_b1_2k_20_seed0"
    effective_device = "cpu" if args.smoke else args.device
    output = args.output_root.resolve() / experiment
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cfg = freeze["config"]
    overrides = training_overrides(cfg, effective_device, experiment)
    if args.smoke:
        overrides = [
            item
            for item in overrides
            if not item.startswith("suite.num_train_steps=")
            and not item.startswith("suite.save_every_steps=")
        ]
        overrides += [
            f"suite.num_train_steps={args.smoke_steps}",
            "suite.save_every_steps=10",
            "suite.log_every_steps=1",
            "device=cpu",
        ]
    overrides += [f"hydra.run.dir={output}"]
    runtime_env = os.environ.copy()
    upstream_pythonpath = [
        str(ROOT / "src"),
        str(upstream),
        str(upstream / "third_party/mimiclabs"),
        str(upstream / "third_party/LIBERO"),
        str(upstream / "third_party/mimicgen"),
    ]
    if runtime_env.get("PYTHONPATH"):
        upstream_pythonpath.append(runtime_env["PYTHONPATH"])
    runtime_env.update(
        {
            "PYTHONPATH": os.pathsep.join(upstream_pythonpath),
            "MUJOCO_GL": "egl",
            "USE_TF": "0",
            "TRANSFORMERS_NO_TF": "1",
            "TENSORBOARD_NO_TF": "1",
        }
    )
    mujoco_override = Path("/tmp/v1r_mujoco335")
    if mujoco_override.is_dir():
        runtime_env["PYTHONPATH"] = os.pathsep.join(
            [str(mujoco_override), runtime_env["PYTHONPATH"]]
        )
    metadata = {
        "stage": "V1-R.2K.seed0.smoke" if args.smoke else "V1-R.2K.seed0",
        "status": "started",
        "formal_training": not args.smoke,
        "device": effective_device,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "command": [sys.executable, "-u", str(upstream / "point_bridge/train.py"), *overrides],
        "output": str(output),
        "frozen": freeze,
        "gpu": _gpu_info(),
    }
    _write_json(output / "launcher_metadata.json", metadata)
    (output / "launcher_overrides.txt").write_text("\n".join(overrides) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)

    env = runtime_env.copy()
    # The smoke run keeps model tensors on CPU, but Point Bridge still needs
    # an EGL device to render point observations. Clearing CUDA_VISIBLE_DEVICES
    # here makes robosuite parse an empty device list and fail before training.
    return_code = _run_smoke(sys.executable, upstream, output, overrides, env)
    metadata["status"] = "completed" if return_code == 0 else "failed"
    metadata["return_code"] = return_code
    snapshots = sorted((output / "snapshot").glob("*.pt")) if (output / "snapshot").is_dir() else []
    metadata["snapshots"] = [str(path) for path in snapshots]
    if snapshots:
        metadata["checkpoint_sha256"] = {str(path): sha256(path) for path in snapshots}
    _write_json(output / "launcher_metadata.json", metadata)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
