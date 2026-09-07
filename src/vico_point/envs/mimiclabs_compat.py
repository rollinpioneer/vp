"""Compatibility helpers for saved MimicLabs MuJoCo model XML."""

from __future__ import annotations

import xml.etree.ElementTree as ET


_PLATE_METADATA = {
    "geoms": (
        {
            "name": "plate_reg_bbox",
            "type": "box",
            "pos": "0 -4.1625e-7 -3.5454325e-5",
            "size": "0.0915184492 0.0915176167 0.007932138625",
            "group": "1",
            "contype": "0",
            "conaffinity": "0",
            "rgba": "0 1 0 0",
        },
    ),
    "sites": (
        {"name": "plate_bottom_site", "pos": "0 0 -0.007967593", "size": "0.001"},
        {"name": "plate_top_site", "pos": "0 0 0.007896684", "size": "0.001"},
        {"name": "plate_horizontal_radius_site", "pos": "0.0915184492 0 0", "size": "0.001"},
    ),
}

_BOWL_VARIANTS = {
    "bowl_0": {
        "bbox_pos": "0 5.369e-7 0",
        "bbox_size": "0.0649999688 0.0649981696 0.0184044146",
        "interior_pos": "0 0 0.018",
        "interior_size": "0.058 0.058 0.025",
        "bottom": "0 0 -0.018404415",
        "top": "0 0 0.018404415",
        "radius": "0.064999969 0 0",
    },
    "bowl_1": {
        "bbox_pos": "-5.85e-7 -8.298e-7 0",
        "bbox_size": "0.059999415 0.0599991702 0.026286042",
        "interior_pos": "0 0 0.025",
        "interior_size": "0.054 0.054 0.045",
        "bottom": "-5.85e-7 -8.298e-7 -0.026286042",
        "top": "-5.85e-7 -8.298e-7 0.026286042",
        "radius": "0.059999415 0 0",
    },
    "bowl_2": {
        "bbox_pos": "4.69775e-7 4.0595e-7 0",
        "bbox_size": "0.057499530225 0.05749959405 0.02421663445",
        "interior_pos": "0 0 0.024",
        "interior_size": "0.05 0.05 0.032",
        "bottom": "4.69775e-7 4.0595e-7 -0.02421663445",
        "top": "4.69775e-7 4.0595e-7 0.02421663445",
        "radius": "0.05749959405 0 0",
    },
    "bowl_4": {
        "bbox_pos": "0 4.2e-7 1.5e-9",
        "bbox_size": "0.05 0.04999958 0.0248779995",
        "interior_pos": "0 0 0.025",
        "interior_size": "0.047 0.047 0.046",
        "bottom": "0 4.2e-7 -0.024877998",
        "top": "0 4.2e-7 0.024878001",
        "radius": "0.05 0 0",
    },
}


def _bowl_metadata(variant: str) -> dict[str, tuple[dict[str, str], ...]]:
    values = _BOWL_VARIANTS[variant]
    passive = {"group": "1", "contype": "0", "conaffinity": "0"}
    return {
        "geoms": (
            {
                "name": "bowl_reg_bbox",
                "type": "box",
                "pos": values["bbox_pos"],
                "size": values["bbox_size"],
                "rgba": "0 1 0 0",
                **passive,
            },
            {
                "name": "bowl_reg_int",
                "type": "ellipsoid",
                "pos": values["interior_pos"],
                "size": values["interior_size"],
                "rgba": "0 0 0 0",
                **passive,
            },
        ),
        "sites": (
            {"name": "bowl_bottom_site", "pos": values["bottom"], "size": "0.001"},
            {"name": "bowl_top_site", "pos": values["top"], "size": "0.001"},
            {"name": "bowl_horizontal_radius_site", "pos": values["radius"], "size": "0.001"},
        ),
    }


def _detect_bowl_variant(xml: str) -> str:
    for variant in _BOWL_VARIANTS:
        if f"/bowl/{variant}/" in xml:
            return variant
    return "bowl_0"


def migrate_saved_model_xml(xml: str) -> str:
    """Apply non-physical compatibility fixes for saved MimicLabs XML.

    The saved model's meshes, bodies, joints, collision parameters, masses and
    poses remain untouched. The function is idempotent.
    """

    bowl_variant = _detect_bowl_variant(xml)
    root = ET.fromstring(xml)
    for texture in root.findall(".//texture[@colorspace]"):
        texture.attrib.pop("colorspace", None)
    for light in root.findall(".//light[@type]"):
        light.attrib.pop("type", None)
    existing = {
        element.attrib["name"]
        for element in root.iter()
        if "name" in element.attrib
    }
    object_metadata = {
        "plate": _PLATE_METADATA,
        "bowl": _bowl_metadata(bowl_variant),
    }
    for object_name, metadata in object_metadata.items():
        body = root.find(f".//body[@name='{object_name}_main']")
        if body is None:
            continue
        for attributes in metadata["geoms"]:
            if attributes["name"] not in existing:
                ET.SubElement(body, "geom", attributes)
                existing.add(attributes["name"])
        for attributes in metadata["sites"]:
            if attributes["name"] not in existing:
                ET.SubElement(body, "site", attributes)
                existing.add(attributes["name"])
    return ET.tostring(root, encoding="unicode")
