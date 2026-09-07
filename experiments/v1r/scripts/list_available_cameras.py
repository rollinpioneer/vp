#!/usr/bin/env python3
"""List camera-like names found in the pinned upstream source/configuration."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    names: set[str] = set()
    for path in args.upstream.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml", ".xml"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for match in re.finditer(r"(?:camera_name|camera_names|pixel_keys|camera)\s*[=:]\s*[\"']([A-Za-z0-9_.-]+)", text):
            names.add(match.group(1))
    print("\n".join(sorted(names)) or "NO_CAMERA_NAMES_FOUND")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
