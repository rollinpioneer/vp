# V1-R.2J-F Delta-Pose Execution-Path Diagnostic

## Scope

- Same four frozen V1-R.2I demonstrations: `demo_18`, `demo_27`, `demo_10`, `demo_2`.
- Four records crossed with six conditions: 24 short replays.
- No policy training or inference; no intermediate state restoration.
- Runtime: robosuite `1.4.1`, MuJoCo `3.3.5`, 20 Hz, 7-D `delta_pose`.

## Results

| Condition | Environment | Action values | Passed |
| --- | --- | --- | ---: |
| A | HDF5 `env_args` | raw `float64` | 4/4 |
| B | HDF5 `env_args` | min-max -> `float32` -> inverse `float64` | 1/4 |
| C | Point Bridge `delta_pose` | raw `float64` | 4/4 |
| D | Point Bridge `delta_pose` | min-max -> `float32` -> inverse `float64` | 1/4 |
| E | HDF5 `env_args` | raw `float32` | 3/4 |
| F | Point Bridge `delta_pose` | raw `float32` | 3/4 |

Per-record outcome:

| Record | A | B | C | D | E | F |
| --- | --- | --- | --- | --- | --- | --- |
| layout 1 `demo_18` | pass | fail | pass | fail | fail | fail |
| layout 2 `demo_27` | pass | pass | pass | pass | pass | pass |
| layout 3 `demo_10` | pass | fail | pass | fail | pass | pass |
| layout 4 `demo_2` | pass | fail | pass | fail | pass | pass |

All failed replays ended as `contact_without_grasp`. The four environment/action-path pairs had identical bottom-step action arrays for the corresponding numeric condition, and all 24 replays had matching bottom-step counts.

## Interpretation

The Point Bridge execution entrypoint is not the differentiating factor: raw actions pass in both environments, normalized actions fail in both environments, and direct `float32` has the same result in both environments.

Direct `float32` is not harmless: it changes `layout_1/demo_18` from success to failure. The shared min-max -> `float32` -> inverse path is more damaging on this sample set and additionally changes `layout_3/demo_10` and `layout_4/demo_2` to failure. The result localizes the next problem to numeric action representation and precision sensitivity, but it does not yet identify a training-safe numeric contract.

The formal `20/20` gate remains unchanged. `selected_label_contract` remains `null`; B0/B1, seed 0, confirm, V2, and V3 remain unauthorized. No retraining or new data capture was performed.

## Evidence

- Full local telemetry: `outputs/v1r/delta_pose_path_diagnostic_2j_f/results_24.json`
- SHA-256: `cf36ac6f395eb1e4fc2cf7608193ed1557b3f0e12cf69014099eb0f13d27a318`
- Size: `9,539,543` bytes; ignored and not uploaded.
