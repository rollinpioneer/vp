# V1-R.2K seed-0 D1 diagnostic result

Date: 2026-09-10
Runtime: robosuite 1.4.1, MuJoCo 3.3.5, Point Bridge commit `491db4c4652ebfbcfce241c1c5e83c6d0c9eea75`, CUDA at 20 Hz.
Protocol: `experiments/v1r/configs/v1r_2k_seed0_d1_protocol.yaml`

## Decision

`V1-R.2K.seed0-D1` is complete as a diagnostic-only stage. The formal
300k clean-dev result remains immutable at `3/40`; no clean-dev result was
reselected and no training was rerun.

The training-fit classification is `fit_inconclusive`: the best checkpoint
replayed `14/20` training initial states, below the preregistered `16/20`
threshold. The 300k checkpoint replayed `11/20`. The next route is a separate
PB-R0 protocol review; PB-R0 itself is not authorized by this D1 run.

## E1: closed-loop fit coverage

All 60 rollouts restored the frozen quantized-source initial state exactly.
There were zero simulator exceptions and zero action-decode errors. Results:

| Checkpoint | Success | Layout 1 | Layout 2 | Layout 3 | Layout 4 |
|---:|---:|---:|---:|---:|---:|
| 100k | 4/20 | 0/5 | 1/5 | 2/5 | 1/5 |
| 200k | 14/20 | 2/5 | 3/5 | 5/5 | 4/5 |
| 300k | 11/20 | 1/5 | 2/5 | 3/5 | 5/5 |

Each row records the source NPZ/PKL/XML/template hashes, initial-state hash,
normalized network-input hash, action shape `(7,)`, CUDA device, and 20 Hz
runtime metadata. Raw tables remain local under `outputs/v1r/d1/`.

## E2: offline action-block fit

E2 used zero simulator rollouts. It evaluated the raw 40-step predictions
against the BCDataset target blocks with valid-action masks; terminal padding
was reported separately and excluded from the valid metrics. The Point Bridge
object-point noise path was reproduced with `noise_std=0.1` and a deterministic
diagnostic seed per sample.

| Checkpoint | Valid MSE | h=0 MSE | h=1-9 MSE | h=10-19 MSE | h=20-39 MSE |
|---:|---:|---:|---:|---:|---:|
| 100k | 0.0032734 | 0.0029894 | 0.0022093 | 0.0031331 | 0.0039324 |
| 200k | 0.0010686 | 0.0018394 | 0.0008945 | 0.0008008 | 0.0012625 |
| 300k | 0.0008451 | 0.0012151 | 0.0008949 | 0.0006473 | 0.0009059 |

The h=0 error was not consistently lower than later horizons at 200k or
300k, so E2 does not establish a simple immediate-action-only bottleneck.

## E3: checkpoint transitions

The existing 40-scenario clean-dev tables were joined without rerunning them.
Scenario IDs and initial-state contracts matched across all three tables.
Pattern counts for 100k/200k/300k were `000:33`, `001:1`, `010:2`,
`011:2`, and `110:2`. The 200k/300k success sets had intersection `2`, union
`7`, Jaccard `0.285714`, with five 200k-to-300k transitions.

## E4: temporal aggregation diagnostic

The preregistered trigger was met by checkpoint-transition evidence. The
frozen exponential-average rows were taken directly from E1; only 40 new
`latest_chunk_first` rollouts were executed, 20 at each of 200k and 300k.
There were zero simulator exceptions and zero action-decode errors.

| Checkpoint | Frozen | Latest chunk first | Added successes | Regressions |
|---:|---:|---:|---:|---:|
| 200k | 14 | 12 | 2 | 4 |
| 300k | 11 | 5 | 0 | 6 |

The preregistered strong-sensitivity rule (at least five added successes and
at most one regression) failed. `latest_chunk_first` is therefore not frozen
as an inference repair.

## Authorization boundary

The following remain false: seed-0 retraining, remaining training seeds,
confirm rollouts, V2, V3, and PB-R0 execution. The formal clean-dev checkpoint
remains the preselected 300k checkpoint with `3/40`; D1 is diagnostic only.

Machine-readable details and source hashes are in
`experiments/v1r/reports/v1r_2k_seed0_d1_result.yaml`.
