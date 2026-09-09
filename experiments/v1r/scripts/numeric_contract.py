"""The single V1-R.2K expert-action numeric contract.

The encoder is intentionally an identity conversion apart from the required
float32 quantization.  The decoder must be used by both capture and replay;
callers must not add normalization, scaling, clipping, smoothing, or a second
quantization step.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from vico_point.action_contracts import (
    DELTA_POSE_FLOAT32_IDENTITY,
    decode_delta_command,
    encode_delta_label,
)


CONTRACT_ID = DELTA_POSE_FLOAT32_IDENTITY
SOURCE_DTYPE = "float64"
LABEL_DTYPE = "float32"
CONTROLLER_DTYPE = "float64"


def _array(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if not np.isfinite(array).all():
        raise ValueError("numeric contract does not accept non-finite actions")
    return np.ascontiguousarray(array)


def encode_expert_delta(raw_action: Any) -> np.ndarray:
    """Convert a source float64 action to the stored training label."""

    return encode_delta_label(raw_action).copy()


def decode_delta_for_controller(label: Any) -> np.ndarray:
    """Convert a stored float32 label to the command sent to robosuite."""

    return decode_delta_command(label)


def encode_sequence(raw_actions: Any) -> np.ndarray:
    actions = _array(raw_actions, dtype=np.dtype(np.float64))
    return encode_expert_delta(actions)


def decode_sequence(labels: Any) -> np.ndarray:
    labels_array = _array(labels, dtype=np.dtype(np.float32))
    return decode_delta_for_controller(labels_array)


def array_sha256(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def exact_contract_check(raw_actions: Any, labels: Any, commands: Any) -> dict[str, Any]:
    raw = _array(raw_actions, dtype=np.dtype(np.float64))
    stored = _array(labels, dtype=np.dtype(np.float32))
    actual = _array(commands, dtype=np.dtype(np.float64))
    expected_labels = encode_sequence(raw)
    expected_commands = decode_sequence(stored)
    return {
        "raw_shape": list(raw.shape),
        "label_shape": list(stored.shape),
        "command_shape": list(actual.shape),
        "label_dtype": str(stored.dtype),
        "command_dtype": str(actual.dtype),
        "labels_exact": bool(np.array_equal(stored, expected_labels)),
        "commands_exact": bool(np.array_equal(actual, expected_commands)),
        "command_equals_label_promoted": bool(
            np.array_equal(actual, stored.astype(np.float64))
        ),
        "length_exact": bool(len(raw) == len(stored) == len(actual)),
        "passed": bool(
            raw.shape == stored.shape == actual.shape
            and np.array_equal(stored, expected_labels)
            and np.array_equal(actual, expected_commands)
        ),
        "raw_sha256": array_sha256(raw),
        "label_sha256": array_sha256(stored),
        "command_sha256": array_sha256(actual),
    }
