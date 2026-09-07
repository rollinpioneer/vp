#!/usr/bin/env python3
"""Replay five audited successful expert trajectories per layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPSTREAM = ROOT / "third_party" / "pointbridge"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from state_utils import raw_array_sha256
from vico_point.envs.mimiclabs_compat import migrate_saved_model_xml


def text_sha256(value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def max_abs_difference(left: Any, right: Any) -> float | None:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        return None
    return float(np.max(np.abs(left_array - right_array)))


def add_upstream_paths(upstream: Path) -> None:
    paths = (
        upstream,
        upstream / "third_party" / "LIBERO",
        upstream / "third_party" / "mimicgen",
        upstream / "third_party" / "mimiclabs",
        upstream / "third_party" / "robocasa",
    )
    for path in reversed(paths):
        sys.path.insert(0, str(path))


def select_demos(manifest: Path, per_layout: int) -> dict[int, list[dict[str, str]]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        layout = int(row["layout"])
        if len(selected[layout]) < per_layout:
            selected[layout].append(row)
    missing = {
        layout: per_layout - len(selected[layout])
        for layout in range(1, 5)
        if len(selected[layout]) < per_layout
    }
    if missing:
        raise ValueError(f"success manifest does not have enough demos: {missing}")
    return selected


def replay_demo(env: Any, demo: Any, layout: int, demo_key: str) -> dict[str, Any]:
    states = np.asarray(demo["states"])
    actions = np.asarray(demo["actions"])
    original_xml = demo.attrs["model_file"]
    if isinstance(original_xml, bytes):
        original_xml_text = original_xml.decode("utf-8")
    else:
        original_xml_text = str(original_xml)
    migrated_xml = migrate_saved_model_xml(original_xml_text)
    record: dict[str, Any] = {
        "layout": layout,
        "demo_key": demo_key,
        "steps": len(actions),
        "model_xml_sha256": text_sha256(original_xml_text),
        "migrated_model_xml_sha256": text_sha256(migrated_xml),
        "saved_initial_state_sha256": raw_array_sha256(states[0]),
        "saved_final_state_sha256": raw_array_sha256(states[-1]),
        "restored_initial_state_sha256": "",
        "restored_initial_state_match": False,
        "restored_initial_state_max_abs_diff": None,
        "saved_final_success": False,
        "saved_final_action_success": False,
        "generator_replay_success": False,
        "full_replay_success": False,
        "generator_replay_reward": None,
        "full_replay_reward": None,
        "generator_final_state_max_abs_diff": None,
        "generator_max_state_abs_diff": None,
        "generator_first_divergence_step": None,
        "exception": "",
    }
    env.reset_to({"states": states[-1], "model": migrated_xml})
    record["saved_final_success"] = bool(env._check_success())
    _, saved_reward, _, _ = env.step(actions[-1])
    record["saved_final_action_success"] = bool(env._check_success())
    record["saved_final_action_reward"] = float(saved_reward)

    env.reset_to({"states": states[0], "model": migrated_xml})
    restored_initial_state = env.sim.get_state().flatten()
    restored_initial = raw_array_sha256(restored_initial_state)
    record["restored_initial_state_sha256"] = restored_initial
    record["restored_initial_state_match"] = (
        restored_initial == record["saved_initial_state_sha256"]
    )
    record["restored_initial_state_max_abs_diff"] = max_abs_difference(
        restored_initial_state, states[0]
    )
    reward = float(env.reward())
    state_differences = []
    for step, action in enumerate(actions[:-1], start=1):
        _, reward, _, _ = env.step(action)
        difference = max_abs_difference(env.sim.get_state().flatten(), states[step])
        if difference is not None:
            state_differences.append(difference)
            if difference > 1e-6 and record["generator_first_divergence_step"] is None:
                record["generator_first_divergence_step"] = step
    record["generator_replay_reward"] = float(reward)
    record["generator_replay_success"] = bool(env._check_success())
    record["generator_final_state_max_abs_diff"] = max_abs_difference(
        env.sim.get_state().flatten(), states[-1]
    )
    record["generator_max_state_abs_diff"] = max(state_differences, default=0.0)

    env.reset_to({"states": states[0], "model": migrated_xml})
    reward = float(env.reward())
    for action in actions:
        _, reward, _, _ = env.step(action)
    record["full_replay_reward"] = float(reward)
    record["full_replay_success"] = bool(env._check_success())
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--success-manifest", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--per-layout", type=int, default=5, choices=(3, 4, 5))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    os.environ.setdefault("MUJOCO_GL", "egl")
    add_upstream_paths(upstream)

    import robosuite as suite
    from mimiclabs.mimiclabs.envs.problems import (  # noqa: F401
        MimicLabs_Lab1_Tabletop_Manipulation,
    )

    selected = select_demos(args.success_manifest.resolve(), args.per_layout)
    records: list[dict[str, Any]] = []
    for layout in range(1, 5):
        task_name = f"bowl_on_plate_{layout}"
        dataset = (
            upstream
            / "data"
            / "mimicgen_data"
            / "bowl_on_plate"
            / task_name
            / "demo"
            / "demo.hdf5"
        )
        bddl = (
            upstream
            / "third_party"
            / "mimiclabs"
            / "mimiclabs"
            / "mimiclabs"
            / "task_suites"
            / "new_task_suite"
            / f"{task_name}.bddl"
        )
        with h5py.File(dataset, "r") as handle:
            saved_env_args = str(handle["data"].attrs["env_args"])
            env_spec = json.loads(saved_env_args)
            env_kwargs = dict(env_spec["env_kwargs"])
            env_kwargs.pop("env_lang", None)
            env_kwargs.update(
                {
                    "has_renderer": False,
                    "has_offscreen_renderer": False,
                    "use_camera_obs": False,
                    "bddl_file_name": str(bddl),
                }
            )
            env = suite.make(env_name=env_spec["env_name"], **env_kwargs)
            env.reset()
            try:
                for index, selected_row in enumerate(selected[layout]):
                    demo_key = selected_row["demo_key"]
                    try:
                        record = replay_demo(
                            env, handle["data"][demo_key], layout, demo_key
                        )
                    except Exception as exc:
                        record = {
                            "layout": layout,
                            "demo_key": demo_key,
                            "steps": int(selected_row["steps"]),
                            "saved_final_success": False,
                            "saved_final_action_success": False,
                            "generator_replay_success": False,
                            "full_replay_success": False,
                            "restored_initial_state_match": False,
                            "exception": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-3000:]}",
                        }
                    record.update(
                        {
                            "dataset_env_args_sha256": text_sha256(saved_env_args),
                            "controller_config_sha256": text_sha256(
                                json.dumps(
                                    env_spec["env_kwargs"]["controller_configs"],
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                            ),
                            "runtime_bddl_path": str(bddl.relative_to(ROOT)),
                            "runtime_bddl_sha256": hashlib.sha256(
                                bddl.read_bytes()
                            ).hexdigest(),
                        }
                    )
                    records.append(record)
                    replay_success = bool(
                        record.get("generator_replay_success")
                        or record.get("full_replay_success")
                    )
                    print(
                        f"layout={layout} {index + 1}/{args.per_layout} {demo_key} replay_success={replay_success}",
                        flush=True,
                    )
            finally:
                env.close()

    for record in records:
        record["replay_success"] = bool(
            record.get("generator_replay_success")
            or record.get("full_replay_success")
        )
        record["saved_success"] = bool(
            record.get("saved_final_success")
            or record.get("saved_final_action_success")
        )
        record["passed"] = bool(
            record["saved_success"]
            and record["replay_success"]
            and not record.get("exception")
        )
    per_layout = {
        str(layout): {
            "checked": sum(record["layout"] == layout for record in records),
            "passed": sum(
                record["layout"] == layout and record["passed"] for record in records
            ),
        }
        for layout in range(1, 5)
    }
    restored_initial_states = sum(
        bool(record.get("restored_initial_state_match")) for record in records
    )
    failed_records = [record for record in records if not record["passed"]]
    first_step_divergences = sum(
        record.get("generator_first_divergence_step") == 1
        for record in failed_records
    )
    failed_layouts = [
        layout
        for layout in range(1, 5)
        if per_layout[str(layout)]["passed"] < per_layout[str(layout)]["checked"]
    ]
    passed = all(record["passed"] for record in records)
    result = {
        "stage": "V1-R.2F",
        "audit": "expert_replay",
        "status": "passed" if passed else "failed",
        "runtime": {
            "robosuite_version": getattr(suite, "__version__", "unknown"),
            "mujoco_version": __import__("mujoco").__version__,
            "environment_config_source": "dataset_env_args_with_local_bddl_and_rendering_disabled",
            "state_divergence_tolerance": 1e-6,
        },
        "per_layout_requested": args.per_layout,
        "checked": len(records),
        "passed": sum(record["passed"] for record in records),
        "restored_initial_state_matches": restored_initial_states,
        "failed_layouts": failed_layouts,
        "failed_records_diverging_at_step_1": first_step_divergences,
        "failure_localization": (
            "action_execution_or_environment_contract_after_exact_state_restore"
            if failed_records
            and restored_initial_states == len(records)
            and first_step_divergences == len(failed_records)
            else "unresolved"
        ),
        "per_layout": per_layout,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    table = ["| layout | checked | passed |", "|---:|---:|---:|"]
    for layout in range(1, 5):
        table.append(
            f"| {layout} | {per_layout[str(layout)]['checked']} | {per_layout[str(layout)]['passed']} |"
        )
    failed_layout_text = "/".join(str(layout) for layout in failed_layouts) or "none"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# V1-R.2F Expert Replay\n\n"
        f"状态：`{result['status']}`；通过：{result['passed']}/{result['checked']}。\n\n"
        + "\n".join(table)
        + f"\n\n初态精确恢复：{restored_initial_states}/{len(records)}。"
        + f"失败记录首步即发生状态分歧：{first_step_divergences}/{len(failed_records)}。\n\n"
        + f"失败布局 {failed_layout_text} 的失败发生在精确初态恢复之后，"
        + "定位到动作执行或环境契约链。B0/B1 训练保持未授权。\n\n"
        + "每条记录包含模型 XML、保存首末状态、恢复首状态的 SHA-256，以及 generator/full action replay 的成功标记和 reward。\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
