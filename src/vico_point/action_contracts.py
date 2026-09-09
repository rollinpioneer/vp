"""Frozen action contracts shared by data collection, training, and deployment."""

from __future__ import annotations

from typing import Any

import numpy as np


DELTA_POSE_FLOAT32_IDENTITY = "delta_pose_float32_identity"


def encode_delta_label(value: Any) -> np.ndarray:
    """Apply the single float32 quantization used for stored delta labels."""

    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("delta action contains non-finite values")
    return np.ascontiguousarray(array)


def decode_delta_command(value: Any) -> np.ndarray:
    """Promote a stored float32 delta label for the float64 controller."""

    return encode_delta_label(value).astype(np.float64, copy=True)
