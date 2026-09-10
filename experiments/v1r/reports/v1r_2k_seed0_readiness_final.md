# V1-R.2K seed-0 readiness completion

This report closes the five readiness items shown in the execution checklist.

## 1. Training launcher

`experiments/v1r/scripts/run_b1_2k_20_seed0.py` validates the frozen config,
dataset manifest, verification artifact, ordered episode mapping, 0005/0006
patch hashes, upstream identity, and seed-0-only authorization before invoking
`point_bridge/train.py`. It supports explicit `--cuda-visible-devices`.

## 2. Clean-dev entry

`experiments/v1r/scripts/run_clean_b1_2k_20_seed0.py` loads the resolved Hydra
config beside the checkpoint, requires `delta_pose`, points, history length 1,
40-query chunking, proprioception, no language, and no dataset min-max path. It
also requires the pre-frozen checkpoint selection and the 300000-step primary
checkpoint SHA. Relative state/output paths are resolved before changing into
the upstream directory. The default evaluation device is CUDA.

## 3. Regression checks

- `pytest -q`: `69 passed`
- YAML/JSON parsing: passed
- Python compilation: passed
- upload-size check: passed
- `git diff --check`: passed

## 4. Independent smoke run

With `CUDA_VISIBLE_DEVICES=3`, the launcher completed a non-scientific 20-step
smoke run with return code `0`, two checkpoints (`0.pt`, `10.pt`), valid point
batches, finite losses, and the expected 7-D delta action path. The smoke run
is not used for success-rate claims. A separate one-row smoke evaluation also
passed with initial-state match `1/1`, simulator exceptions `0`, and action
decode errors `0`; it used only the smoke checkpoint and cannot affect the
formal clean-dev gate.

## 5. Formal seed-0 run and one clean-dev gate

The authorized CUDA run completed 300000 steps. The primary checkpoint was
frozen before evaluation. The repaired clean-dev entry then completed all 40
frozen rollouts on CUDA:

- successes: `3/40`
- per layout: `0/10`, `2/10`, `0/10`, `1/10`
- initial-state matches: `40/40`
- simulator exceptions: `0`
- action decode errors: `0`
- failure stages: `no_approach=9`, `no_grasp=22`, `post_grasp_drop=6`

The registered gate requires at least `20/40`; therefore the gate failed.
This does not invalidate the passed action/data/point/chunk contracts. It keeps
confirm rollouts, remaining seeds, V2, and V3 unauthorized. The next stage is
`diagnose_pilot_coverage_or_learnability`.
