# V1-R.2J-N Numeric Action Contract Candidate

## Scope

- Point Bridge `delta_pose` entrypoint only.
- Same four frozen V1-R.2J-F records; no tail and no intermediate state restore.
- Candidate: `label_float32 = raw.astype(np.float32)` then `controller_action = label_float32.astype(np.float64)`.
- No dataset min-max translation or inverse transform.

## Result

| Record | Result |
| --- | --- |
| layout 1 `demo_18` | failed: `contact_without_grasp` |
| layout 2 `demo_27` | passed |
| layout 3 `demo_10` | passed |
| layout 4 `demo_2` | passed |

The selected-record gate is `3/4`, below the fixed `4/4` prerequisite. The full 20-record expansion was therefore not run.

The promoted-float64 candidate and the prior direct-float32 Point Bridge condition had exactly equal bottom action values, controller goals, EEF positions, bowl positions, and success outcomes on all four records. Controller input dtype alone does not repair the float32-sensitive `demo_18` trajectory.

No action contract is selected. Training, confirm, V2, and V3 remain unauthorized. The next stage is a separately versioned sequential data protocol in which the deployment numeric path is applied before each command is executed and the resulting states are captured.

## Local Evidence

- `outputs/v1r/delta_numeric_contract_2j_n/selected_results.json`
- SHA-256: `8c681548419e4f5f50059fe8f73de82ec1efb6d76555b369d80d18c1bebbabc8`
- Size: `1,490,795` bytes; ignored and not uploaded.
