#!/usr/bin/env python3
"""Audit Point Bridge configuration/source declarations without copying it."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/v1r/reports/original_environment_input_audit.md"))
    args = parser.parse_args()
    hits: dict[str, list[str]] = {key: [] for key in ("obj_of_interest", "use_full_scene_pcd", "num_points_per_obj", "use_vlm_points", "point_tracking", "segment_depth")}
    for path in args.upstream.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml", ".json"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for key in hits:
            if re.search(re.escape(key), text):
                hits[key].append(str(path.relative_to(args.upstream)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Point Bridge 原始环境输入审计", "", "这是源码/配置关键词审计，不是策略成功率证据。", ""]
    for key, files in hits.items():
        lines.append(f"## `{key}`")
        lines.extend(f"- `{item}`" for item in sorted(files)[:20])
        if len(files) > 20:
            lines.append(f"- ... 共 {len(files)} 个文件命中")
        if not files:
            lines.append("- 未发现")
    lines.extend(["", "结论：必须在正式别名场景中重新核对障碍物是否已经属于 `obj_of_interest`；本审计不把已有 Point Bridge 输入能力写成新方法。"])
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
