"""Small compatibility helpers for the pinned Point Bridge MuJoCo stack."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np


def mesh_scales_from_xml(model: object) -> np.ndarray:
    """Return mesh scales when the installed MuJoCo binding lacks ``mesh_scale``.

    MuJoCo 3 exposes mesh scales through ``MjModel.mesh_scale``.  The pinned
    Point Bridge code also runs with MuJoCo 2.3, where that convenience array
    is absent even though the scale remains present in the compiled XML.
    """

    raw_model = model._model
    scales = np.ones((int(raw_model.nmesh), 3), dtype=np.float64)
    name_to_id = model._mesh_name2id
    root = ET.fromstring(model.get_xml())
    for mesh in root.iter("mesh"):
        name = mesh.get("name")
        if not name or name not in name_to_id:
            continue
        scale = mesh.get("scale")
        if scale:
            values = np.fromstring(scale, sep=" ", dtype=np.float64)
            if values.size == 3:
                scales[int(name_to_id[name])] = values
    return scales


def mesh_transforms_from_xml(model: object) -> tuple[np.ndarray, np.ndarray]:
    """Return mesh position and quaternion offsets from compiled model XML."""

    raw_model = model._model
    positions = np.zeros((int(raw_model.nmesh), 3), dtype=np.float64)
    quaternions = np.zeros((int(raw_model.nmesh), 4), dtype=np.float64)
    quaternions[:, 0] = 1.0
    name_to_id = model._mesh_name2id
    root = ET.fromstring(model.get_xml())
    for mesh in root.iter("mesh"):
        name = mesh.get("name")
        if not name or name not in name_to_id:
            continue
        mesh_id = int(name_to_id[name])
        for attribute, target, size in (
            ("refpos", positions, 3),
            ("refquat", quaternions, 4),
        ):
            value = mesh.get(attribute)
            if value:
                values = np.fromstring(value, sep=" ", dtype=np.float64)
                if values.size == size:
                    target[mesh_id] = values
    return positions, quaternions


def mesh_paths_from_xml(model: object) -> tuple[np.ndarray, bytes]:
    """Return MuJoCo-style mesh path offsets and the NUL-delimited path blob."""

    raw_model = model._model
    addresses = np.zeros(int(raw_model.nmesh), dtype=np.int32)
    chunks: list[bytes] = []
    cursor = 0
    name_to_id = model._mesh_name2id
    root = ET.fromstring(model.get_xml())
    for mesh in root.iter("mesh"):
        name = mesh.get("name")
        path = mesh.get("file")
        if not name or not path or name not in name_to_id:
            continue
        path_bytes = path.encode("utf-8")
        addresses[int(name_to_id[name])] = cursor
        chunks.append(path_bytes + b"\0")
        cursor += len(path_bytes) + 1
    return addresses, b"".join(chunks)
