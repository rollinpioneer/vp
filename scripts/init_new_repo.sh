#!/usr/bin/env bash
# Initialize only this project directory; do not touch the legacy repository.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ -n "${LEGACY_ROOT:-}" ]]; then
  LEGACY="$(cd -- "$LEGACY_ROOT" && pwd -P)"
  if [[ "$ROOT" == "$LEGACY" || "$ROOT" == "$LEGACY/"* ]]; then
    printf 'ERROR: Move this directory outside LEGACY_ROOT before initialization.\n' >&2
    exit 1
  fi
fi
if git -C "$ROOT" rev-parse --show-toplevel >/dev/null 2>&1; then
  TOP="$(git -C "$ROOT" rev-parse --show-toplevel)"
  TOP="$(cd -- "$TOP" && pwd -P)"
  if [[ "$TOP" != "$ROOT" ]]; then
    printf 'ERROR: This folder is inside another repository: %s\n' "$TOP" >&2
    exit 1
  fi
  printf 'Existing independent repository: %s\n' "$ROOT"
else
  git -C "$ROOT" init -b main
fi
printf '\nNext: read docs/视觉稀疏点策略_分阶段实验计划.md and begin V0.\n'
printf 'No remote repository was created. No upstream code or data was downloaded.\n'
