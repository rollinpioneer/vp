from __future__ import annotations

import numpy as np

from experiments.v1r.scripts.numeric_contract import (
    decode_delta_for_controller,
    encode_expert_delta,
    exact_contract_check,
)


def test_float32_label_is_the_only_quantization() -> None:
    raw = np.asarray([[1.0 / 3.0, -2.0, 0.125, 0.0, 0.0, 0.0, -1.0]], dtype=np.float64)
    label = encode_expert_delta(raw)
    command = decode_delta_for_controller(label)
    assert label.dtype == np.float32
    assert command.dtype == np.float64
    assert np.array_equal(command, label.astype(np.float64))
    assert exact_contract_check(raw, label, command)["passed"]
