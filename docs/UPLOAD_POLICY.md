# GitHub upload policy

This repository uploads source code, configuration, documentation, manifests,
tests, and small reproducibility metadata while preserving the repository
directory structure.

## Size policy

- A tracked file must not exceed 10 MiB without an explicit review.
- Datasets, RGB-D recordings, videos, model weights, cached features, archives,
  and experiment output directories are not stored in Git.
- The current tracked tree contains no file above 10 MiB. The largest tracked
  file is `manifests/scenario_registry.csv`, approximately 44 KiB.
- `docs/视觉稀疏点策略_分阶段实验计划.docx` is approximately 36 KiB and is
  intentionally uploaded because it is a small source document, not a large
  binary artifact.

Exact local files and external asset locations intentionally omitted from the
upload are recorded in `manifests/not_uploaded_files.csv`, including their type,
purpose, and exclusion reason. Future externally hosted assets should add a
filename, URL or storage identifier, byte size, and SHA-256 checksum to that
manifest before being used for a reproducible run.

## Pre-push check

Run:

```bash
python scripts/check_upload_size.py
```

The command checks every Git-tracked file and fails if any file exceeds the
10 MiB repository limit.
