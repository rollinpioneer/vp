from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/v1r/scripts"))

from evaluate_seed0_offline_chunks import _target_block  # noqa: E402


def load_yaml(relative: str) -> dict:
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


def test_d1_trainfit_manifest_is_frozen_balanced_set() -> None:
    path = ROOT / "experiments/v1r/manifests/b1_2k_20_trainfit.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    assert Counter(int(row["layout"]) for row in rows) == Counter({1: 5, 2: 5, 3: 5, 4: 5})
    assert len({row["scenario_id"] for row in rows}) == 20
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "fd7366506c9ceda9cfb57cf21dcfa09522a48b49ca1389e7e0654368274335f6"
    )


def test_d1_delta_chunk_padding_holds_only_final_gripper() -> None:
    actions = np.array(
        [[0.1, 0.2, 0.3, 0.01, 0.02, 0.03, -1.0], [0.4, 0.5, 0.6, 0.04, 0.05, 0.06, 1.0]],
        dtype=np.float32,
    )
    target, valid = _target_block(actions, 1, queries=4)
    np.testing.assert_array_equal(valid, [True, False, False, False])
    np.testing.assert_array_equal(target[0], actions[1])
    np.testing.assert_array_equal(target[1:, :6], np.zeros((3, 6), dtype=np.float32))
    np.testing.assert_array_equal(target[1:, 6], np.ones(3, dtype=np.float32))


def test_d1_result_and_authoritative_decision_are_synchronized() -> None:
    result = load_yaml("experiments/v1r/reports/v1r_2k_seed0_d1_result.yaml")
    decision = load_yaml("experiments/v1r/v1r_decision.yaml")
    assert result["status"] == "completed_diagnostic_only"
    assert result["classification"]["fit_classification"] == "fit_inconclusive"
    assert result["E1_trainfit_closed_loop"]["rollouts"] == 60
    assert result["E4_temporal_aggregation"]["new_rollouts"] == 40
    assert result["E4_temporal_aggregation"]["strong_sensitivity_passed"] is False
    assert decision["latest_completed_stage"] == "V1-R.2K.seed0-D1"
    assert decision["v1r_2k_seed0_d1"]["fit_classification"] == "fit_inconclusive"
    assert decision["next_stage"] == "V1-R.PB-R0_separate_protocol_pending"


def test_d1_does_not_open_training_or_formal_experiment_gates() -> None:
    result = load_yaml("experiments/v1r/reports/v1r_2k_seed0_d1_result.yaml")
    boundary = result["decision_boundary"]
    for key in (
        "seed0_retraining_authorized",
        "remaining_training_seeds_authorized",
        "confirm_rollouts_authorized",
        "v2_formal_experiment_authorized",
        "v3_formal_experiment_authorized",
        "pb_r0_authorized",
    ):
        assert boundary[key] is False
    assert boundary["formal_clean_dev_unchanged"] is True
    assert boundary["formal_clean_dev_successes"] == 3
    assert boundary["formal_clean_dev_rollouts"] == 40
