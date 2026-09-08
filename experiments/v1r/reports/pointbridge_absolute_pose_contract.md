## Material Passport

- Schema: ARS Material Passport 9
- Material type: reproducibility validation report
- Source: local Point Bridge PKL/HDF5 demonstrations and pinned source tree
- Raw data handling: local only; no raw dataset or model artifact uploaded
- Verification status: VERIFIED

# V1-R.2G Point Bridge Absolute-Pose Contract Gate

Status: `failed`. Decision case: `B`.

## Runtime And Contract

- Formal runtime: robosuite `1.4.1`, MuJoCo `3.3.5`.
- Environment entrypoint: `point_bridge.suite.mimiclabs.make`.
- Controller: `OSC_POSE`, `control_delta=False`, 20 Hz, one 10-D pose action per step.
- Label source: native `BCDataset` with `act_subsample=1`; no policy or language-model inference was used.
- Rotation path: native SciPy quaternion-to-matrix conversion and Point Bridge 6-D helpers.
- Normalization path: `BCDataset.preprocess['actions']` followed by the pose branch of `PB.act` postprocessing.
- Dtype path: normalized training labels are `float32`; deployment postprocessing yields `float64` actions from the saved float64 statistics.
- Coordinate and gripper checks: each PKL `robot_base` matches the official environment, and the scalar open/close sign encoding is preserved.
- Exact PKL initial state is restored after environment reset and checked before action 0.

## Results

| Layout | Checked | Passed | Success rate | Position error max (m) | Rotation error max (rad) |
|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 1 | 0.200 | 0.062838213 | 0.086250211 |
| 2 | 5 | 3 | 0.600 | 0.057655587 | 0.082232141 |
| 3 | 5 | 4 | 0.800 | 0.062728925 | 0.098126289 |
| 4 | 5 | 5 | 1.000 | 0.069003455 | 0.065474295 |

Total: `13/20`; telemetry rows: `3832`.
All restored initial states matched: `20/20`.
Translation boundary hits: `0`; rotation boundary hits: `0`. The official absolute OSC controller has no position or orientation goal limits configured.

## Decision

- Case: `B`
- Conclusion: `absolute_pose_contract_still_fails_on_layout_1_or_2`
- B0/B1 training authorized: `false`
- Clean baseline gate: `blocked`
- V2/V3 formal experiments authorized: `false`
- Next stage: `repair_absolute_transform_OSC_frequency_gripper_and_saved_state_compatibility_without_retraining`

Every executed step's absolute target, actual EEF pose, tracking error, boundary status, grasp state, and task-success flag is retained in `pointbridge_absolute_pose_contract.json`.

## Failures

- Layout 1 `demo_0`: `no_grasp`; exception=`False`.
- Layout 1 `demo_2`: `no_grasp`; exception=`False`.
- Layout 1 `demo_3`: `no_grasp`; exception=`False`.
- Layout 1 `demo_4`: `no_grasp`; exception=`False`.
- Layout 2 `demo_27`: `no_grasp`; exception=`False`.
- Layout 2 `demo_32`: `no_grasp`; exception=`False`.
- Layout 3 `demo_3`: `no_grasp`; exception=`False`.
