#!/usr/bin/env python3
"""Run D1 E4 latest-chunk-first inference on the 20 trainfit initial states."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_seed0_trainfit import _episode  # noqa: E402
from seed0_d1_common import DEFAULT_UPSTREAM, RESOLVED_CONFIG, configure_runtime, read_csv, sha256, validate_preflight  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("latest_chunk_first",), required=True)
    parser.add_argument("--resolved-config", type=Path, default=RESOLVED_CONFIG)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    frozen=validate_preflight(args.upstream.resolve()); configure_runtime(args.upstream.resolve())
    import importlib.util
    legacy_path=ROOT/"scripts/evaluate_pointbridge_paired.py"; spec=importlib.util.spec_from_file_location("d1_agg_legacy", legacy_path)
    if spec is None or spec.loader is None: raise RuntimeError("cannot load legacy runner")
    legacy=importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy); legacy._add_paths(args.upstream.resolve()); eval_module=legacy._load_eval_module(args.upstream.resolve())
    from omegaconf import OmegaConf
    rows = read_csv(args.manifest.resolve())
    if len(rows) != 20:
        raise ValueError("E4 requires the frozen 20-row trainfit manifest")
    if len(args.checkpoints) != 2 or {int(path.stem) for path in args.checkpoints} != {200000, 300000}:
        raise ValueError("E4 requires exactly the 200000 and 300000 checkpoints")
    old=Path.cwd(); output=[]; latest=[]
    try:
        from run_clean_b1_2k_20_seed0 import load_snapshot
        for checkpoint_arg in args.checkpoints:
            checkpoint = checkpoint_arg.resolve()
            step = int(checkpoint.stem)
            frozen_path = ROOT / f"outputs/v1r/d1/seed0_trainfit_{step}.csv"
            frozen_rows = read_csv(frozen_path)
            if len(frozen_rows) != 20:
                raise ValueError(f"missing completed E1 frozen result: {frozen_path}")
            for result in frozen_rows:
                result["aggregation_mode"] = "frozen_exponential_average"
                output.append(result)

            cfg=OmegaConf.load(args.resolved_config.resolve())
            cfg.eval=True; cfg.device=args.device; cfg.save_video=False; cfg.use_tb=False
            cfg.bc_weight=str(checkpoint); cfg.suite.num_eval_episodes=1; cfg.expert_dataset=cfg.dataloader.bc_dataset
            os.chdir(args.upstream.resolve())
            workspace=eval_module.Workspace(cfg)
            load_snapshot(workspace, checkpoint, args.device)
            workspace.agent.train(False)
            try:
                original_act=workspace.agent.act
                def latest_act(obs, stats, step, global_step, **kwargs):
                    if hasattr(workspace.agent,"all_time_actions"):
                        workspace.agent.all_time_actions.zero_()
                        workspace.agent.all_time_actions_populated.zero_()
                    return original_act(obs, stats, step, global_step, **kwargs)
                workspace.agent.act=latest_act
                for i,row in enumerate(rows,1):
                    result=_episode(workspace,workspace.env[int(row["layout"])-1],row,checkpoint,ROOT)
                    result["aggregation_mode"]="latest_chunk_first"
                    latest.append(result)
                    print(f"{step}k latest {i}/20 {row['scenario_id']} success={result['success']}",flush=True)
            finally:
                workspace.agent.act=original_act
                for env in workspace.env:
                    try: env.close()
                    except Exception: pass
            os.chdir(old)
    finally:
        os.chdir(old)
    # Output contains paired rows for the same checkpoint and trainfit states.
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({"stage":"V1-R.2K.seed0-D1.E4","status":"complete","diagnostic_only":True,
        "checkpoint_sha256": {str(path.stem): sha256(path.resolve()) for path in args.checkpoints},
        "frozen_from_e1": output, "latest_chunk_first": latest,
        "new_rollouts": len(latest), "gate_impact":"diagnostic_only_not_a_clean_dev_gate"},indent=2)+"\n",encoding="utf-8")
    return 0


if __name__ == "__main__": raise SystemExit(main())
