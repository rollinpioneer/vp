#!/usr/bin/env python3
"""Add legacy robosuite extent sites to the local RoboCasa task assets."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = (
    ROOT
    / "third_party"
    / "pointbridge"
    / "third_party"
    / "robocasa"
    / "robocasa"
    / "models"
    / "assets"
    / "objects"
    / "objaverse"
)
TASK_MODELS = (
    "plate/plate_19/model.xml",
    "bowl/bowl_0/model.xml",
    "bowl/bowl_1/model.xml",
    "bowl/bowl_2/model.xml",
    "bowl/bowl_4/model.xml",
)


def _numbers(value: str) -> list[float]:
    return [float(item) for item in value.split()]


def add_extent_sites(path: Path) -> bool:
    tree = ET.parse(path)
    root = tree.getroot()
    root_body = root.find("./worldbody/body")
    bbox = root.find(".//geom[@name='reg_bbox']")
    if root_body is None or bbox is None:
        raise ValueError(f"missing root body or reg_bbox in {path}")
    pos = _numbers(bbox.attrib["pos"])
    size = _numbers(bbox.attrib["size"])
    existing = {site.attrib.get("name") for site in root.iter("site")}
    sites = {
        "bottom_site": (pos[0], pos[1], pos[2] - size[2]),
        "top_site": (pos[0], pos[1], pos[2] + size[2]),
        "horizontal_radius_site": (max(size[0], size[1]), 0.0, 0.0),
    }
    changed = False
    for name, xyz in sites.items():
        if name in existing:
            continue
        ET.SubElement(
            root_body,
            "site",
            {
                "name": name,
                "pos": " ".join(f"{value:.12g}" for value in xyz),
                "size": "0.001",
                "rgba": "0 0 0 0",
            },
        )
        changed = True
    if changed:
        tree.write(path, encoding="unicode")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    args = parser.parse_args()
    for relative in TASK_MODELS:
        path = args.asset_root / relative
        changed = add_extent_sites(path)
        print(f"{'patched' if changed else 'ready'} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
