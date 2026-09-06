"""Compatibility helpers for saved MimicLabs MuJoCo model XML."""

from __future__ import annotations

import xml.etree.ElementTree as ET


_OBJECT_METADATA = {
    "plate": {
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
    },
    "bowl": {
        "geoms": (
            {
                "name": "bowl_reg_bbox",
                "type": "box",
                "pos": "0 5.369e-7 0",
                "size": "0.0649999688 0.0649981696 0.0184044146",
                "group": "1",
                "contype": "0",
                "conaffinity": "0",
                "rgba": "0 1 0 0",
            },
            {
                "name": "bowl_reg_int",
                "type": "ellipsoid",
                "pos": "0 0 0.018",
                "size": "0.058 0.058 0.025",
                "group": "1",
                "contype": "0",
                "conaffinity": "0",
                "rgba": "0 0 0 0",
            },
        ),
        "sites": (
            {"name": "bowl_bottom_site", "pos": "0 0 -0.018404415", "size": "0.001"},
            {"name": "bowl_top_site", "pos": "0 0 0.018404415", "size": "0.001"},
            {"name": "bowl_horizontal_radius_site", "pos": "0.064999969 0 0", "size": "0.001"},
        ),
    },
}


def migrate_saved_model_xml(xml: str) -> str:
    """Add only non-physical metadata nodes required by current robosuite mappings.

    The saved model's meshes, bodies, joints, collision parameters, masses and
    poses remain untouched. The function is idempotent.
    """

    root = ET.fromstring(xml)
    existing = {
        element.attrib["name"]
        for element in root.iter()
        if "name" in element.attrib
    }
    for object_name, metadata in _OBJECT_METADATA.items():
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
