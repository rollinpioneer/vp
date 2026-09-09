from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_pointbridge_patch_uses_one_identity_contract() -> None:
    patch = (ROOT / "patches/pointbridge/0005-mimiclabs-delta-pose-float32-identity-training.patch").read_text(
        encoding="utf-8"
    )
    assert "from vico_point.action_contracts import encode_delta_label" in patch
    assert "from vico_point.action_contracts import decode_delta_command" in patch
    assert "self.preprocess[\"actions\"] = encode_delta_label" in patch
    assert 'post_process["actions"] = decode_delta_command' in patch
    assert "action_scale = np.maximum" in patch  # removed historical implementation
    assert "+            action_scale = np.maximum" not in patch


def test_patch_application_order_includes_0005() -> None:
    script = (ROOT / "scripts/apply_pointbridge_patches.py").read_text(encoding="utf-8")
    index_0004 = script.index("0004-mimiclabs-delta-pose-contract.patch")
    index_0005 = script.index("0005-mimiclabs-delta-pose-float32-identity-training.patch")
    assert index_0004 < index_0005


def test_delta_pose_chunk_contract_preserves_zero_motion_and_gripper() -> None:
    patch = (ROOT / "patches/pointbridge/0006-mimiclabs-delta-pose-chunk-contract.patch").read_text(
        encoding="utf-8"
    )
    assert 'if self._action_mode == "delta_pose":' in patch
    assert "np.zeros(" in patch
    assert "post[:, -1] = actions[-1, -1]" in patch
    assert "self.all_time_actions_populated" in patch
    assert "actions_populated = self.all_time_actions_populated[:, step]" in patch
    assert "+            actions_populated = torch.all(actions_for_curr_step != 0" not in patch


def test_training_fixture_uses_previous_gripper_command() -> None:
    source = (ROOT / "experiments/v1r/scripts/verify_v1r_2k_training_contract.py").read_text(
        encoding="utf-8"
    )
    assert "gripper_observation[0] = -1.0" in source
    assert "gripper_observation[1:] = labels[:-1, -1]" in source
    assert '"gripper_states": labels[:, -1]' not in source


def test_training_contract_script_exposes_repo_src_before_patch_imports() -> None:
    source = (ROOT / "experiments/v1r/scripts/verify_v1r_2k_training_contract.py").read_text(
        encoding="utf-8"
    )
    assert 'sys.path.insert(0, str(ROOT / "src"))' in source
