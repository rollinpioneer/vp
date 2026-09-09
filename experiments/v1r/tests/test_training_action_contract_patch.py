from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from experiments.v1r.scripts.state_utils import (
    pointbridge_core_env,
    refresh_pointbridge_observation,
)
from experiments.v1r.scripts.verify_v1r_2k_3_dataset import _expected_chunk


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


def test_pointbridge_core_traversal_ignores_forwarded_attributes() -> None:
    class Leaf:
        pass

    class PointWrapper:
        def __init__(self) -> None:
            self._env = Leaf()
            self._pixel_keys = ["pixels_right"]

        def get_gt_points(self) -> tuple[None, None]:
            return None, None

    class ForwardingWrapper:
        def __init__(self, env: object) -> None:
            self._env = env

        def __getattr__(self, name: str) -> object:
            return getattr(self._env, name)

    expected = PointWrapper()
    assert hasattr(ForwardingWrapper(expected), "get_gt_points")
    assert pointbridge_core_env(ForwardingWrapper(expected)) is expected


def test_refresh_helper_exposes_explicit_gripper_state() -> None:
    signature = inspect.signature(refresh_pointbridge_observation)
    assert signature.parameters["gripper_state"].default == -1.0


def test_point_dataset_builder_freezes_templates_and_previous_gripper() -> None:
    source = (ROOT / "experiments/v1r/scripts/build_b1_2k_point_pkls.py").read_text(
        encoding="utf-8"
    )
    assert "gripper[1:] = labels[:-1, -1]" in source
    assert "fixed_object_points" in source
    assert '"object_point_templates"' in source
    assert "point_sampling_seed" in source
    assert 'core._env.reset_to({"states": states[0], "model": model_xml})' in source


def test_real_delta_chunk_padding_is_zero_motion_with_last_gripper() -> None:
    actions = np.asarray(
        [[0.5, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 1]],
        dtype=np.float32,
    )
    chunk = _expected_chunk(actions, sample_idx=1, num_queries=4)
    expected = np.zeros((1, 4, 7), dtype=np.float32)
    expected[0, :, -1] = 1
    assert np.array_equal(chunk, expected)


def test_dataset_verifier_uses_formal_chunking_configuration() -> None:
    source = (ROOT / "experiments/v1r/scripts/verify_v1r_2k_3_dataset.py").read_text(
        encoding="utf-8"
    )
    assert "action_chunking=True" in source
    assert "num_queries=40" in source
    assert "history_len=1" in source
    assert 'action_mode="delta_pose"' in source
    assert '"chunked_dataloader_exact"' in source


def test_seed0_authorization_is_narrow_and_not_started() -> None:
    config = (ROOT / "experiments/v1r/configs/b1_2k_20_seed0.yaml").read_text(
        encoding="utf-8"
    )
    assert "status: authorized_not_started" in config
    assert "authorized_seeds: [0]" in config
    assert "num_demos_per_layout: 5" in config
    assert "success_threshold: 20" in config
    assert "confirm_rollouts_authorized: false" in config
