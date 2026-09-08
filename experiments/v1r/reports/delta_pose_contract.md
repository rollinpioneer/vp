# V1-R.2J Delta-Pose Contract Validation

## Scope

- Same 20 accepted V1-R.2I sequential-success demonstrations; no policy training or inference.
- Runtime: robosuite `1.4.1`, MuJoCo `3.3.5`, official Point Bridge suite, delta OSC, 20 Hz.
- Strict execution gate: `20/20`; normalization, gripper sign, runtime mode, and task execution are separate checks.
- The raw 7-D actions are the commands issued during the continuous demonstrations; no intermediate state is restored.

## Gates

| Gate | Checked | Passed | Status |
| --- | ---: | ---: | --- |
| normalization_roundtrip | 20 | 20 | passed |
| gripper_sign_preservation | 20 | 20 | passed |
| official_delta_runtime | 20 | 20 | passed |
| execution_replay | 20 | 17 | failed |

## Per Layout

| Layout | Initial state | Roundtrip | Gripper sign | Delta runtime | Execution |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 5/5 | 5/5 | 5/5 | 5/5 | 4/5 |
| 2 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| 3 | 5/5 | 5/5 | 5/5 | 5/5 | 4/5 |
| 4 | 5/5 | 5/5 | 5/5 | 5/5 | 4/5 |

## Failures

The strict execution failures were: layout 1 `demo_18`, layout 3 `demo_10`, layout 4 `demo_2`.

## Decision

The delta-pose fallback is executable in the official runtime for 17/20 demonstrations, but it does not satisfy the predeclared 20/20 contract. It is retained as an explicit unvalidated interface patch; no label contract is selected and training remains blocked.

## Invariants

- HDF5 action-prefix matches: `20/20`.
- Initial-state matches: `20/20`.
- Normalization roundtrips: `20/20`.
- Gripper signs preserved: `20/20`.
- Official delta runtime: `20/20`.
- Policy training or inference: `false`.
- Intermediate state restoration: `false`.
- Simulator exceptions: `0`.

## Local Evidence

- `outputs/v1r/delta_pose_contract_2j/results.json`: `585617ed7dc88c30cf5ac1af0eb1ce6b4cdb343e8f1b41399df2007bcf690379` (5917081 bytes; ignored, not uploaded).
- Full per-step telemetry remains local; this tracked report contains compact summaries only.

## V1-R.2J-F follow-up

The follow-up 24-replay numeric-path diagnostic used four frozen demonstrations and compared both environment entrypoints with raw `float64`, raw `float32`, and shared min-max -> `float32` -> inverse `float64` actions. Results were A/C `4/4`, E/F `3/4`, and B/D `1/4`; the environment entrypoint was equivalent for each numeric condition. Direct `float32` already failed `demo_18`, while the min-max roundtrip additionally failed `demo_10` and `demo_2`. The formal 20-demo contract remains failed and training remains blocked.

Compact evidence: `experiments/v1r/reports/delta_pose_path_diagnostic_2j_f.json`.
