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
