## Material Passport

- Schema: ARS Material Passport 9
- Material type: controlled simulation factorial report
- Source: the same 20 local Point Bridge demonstrations used by V1-R.2G
- Raw data handling: PKL/HDF5/checkpoints remain local and gitignored
- Verification status: VERIFIED

# V1-R.2H Pose-Target / Gripper-Alignment Factorial

Status: `failed`. Outcome: `result_4_no_group_reaches_20_of_20`.

## Fixed Contract

- Runtime: robosuite `1.4.1`, MuJoCo `3.3.5`, `OSC_POSE`, `control_delta=False`, 20 Hz.
- Selection: the identical 20 V1-R.2G demonstrations and exact saved initial states.
- Only pose source P0/P1 and gripper index G0/G1 vary; XML, robot base, rotation helpers, sign convention, source trajectory length, and success predicate remain fixed.
- Stable grasp means `_check_grasp=True` for at least 5 consecutive control steps.
- Closed-without-bowl means commanded close, actual finger aperture <= 0.020 m, and neither bowl contact nor grasp is present.
- The formal direct-expert action-contract threshold remains 20/20.

## Results

| Group | Pose | Gripper | L1 | L2 | L3 | L4 | Total | Formal gate |
|---|---|---|---:|---:|---:|---:|---:|---|
| A | P0_next_measured_eef_pose | G0_next_raw_gripper_state | 1/5 | 3/5 | 4/5 | 5/5 | 13/20 | fail |
| B | P0_next_measured_eef_pose | G1_current_transition_gripper_command | 0/5 | 3/5 | 2/5 | 4/5 | 9/20 | fail |
| C | P1_saved_controller_absolute_target | G0_next_raw_gripper_state | 0/5 | 4/5 | 5/5 | 4/5 | 13/20 | fail |
| D | P1_saved_controller_absolute_target | G1_current_transition_gripper_command | 0/5 | 2/5 | 5/5 | 4/5 | 11/20 | fail |

Baseline A replication: `true` (13/20 versus authoritative 13/20).

## Paired Baseline-Failure Effects

| Group | A failures repaired | A successes regressed | Repairs >=6/7 |
|---|---:|---:|---|
| B | 0/7 | 4 | false |
| C | 2/7 | 2 | false |
| D | 1/7 | 3 | false |

## Decision

- Result: `result_4_no_group_reaches_20_of_20`
- Conclusion: `current_saved_state_sequences_do_not_admit_a_simple_pose_or_gripper_index_fix`
- Formal factorial gate: `failed`
- Next stage: `regenerate_sequential_success_demos_in_formal_runtime_and_stop_using_current_pkl_for_training`
- B0/B1 training, confirm rollouts, V2, and V3 remain unauthorized.

## Per-Trajectory Grasp Table

| Layout | Demo | Group | Raw close | Applied close | Offset | Distance at close (m) | First contact | Stable grasp | Grasp run | Closed empty | Success |
|---:|---|---|---:|---:|---:|---:|---|---:|---:|---|---|
| 1 | demo_0 | A | 85 | 84 | -1 | 0.052757 | 88 | None | 0 | true | false |
| 1 | demo_0 | B | 85 | 85 | 0 | 0.051747 | 89 | None | 0 | true | false |
| 1 | demo_0 | C | 85 | 84 | -1 | 0.049793 | 87 | None | 0 | true | false |
| 1 | demo_0 | D | 85 | 85 | 0 | 0.049032 | 88 | None | 0 | true | false |
| 1 | demo_2 | A | 89 | 88 | -1 | 0.051896 | 113 | None | 0 | true | false |
| 1 | demo_2 | B | 89 | 89 | 0 | 0.051910 | 114 | None | 0 | true | false |
| 1 | demo_2 | C | 89 | 88 | -1 | 0.052768 | 111 | None | 0 | true | false |
| 1 | demo_2 | D | 89 | 89 | 0 | 0.052777 | 111 | None | 0 | true | false |
| 1 | demo_3 | A | 84 | 83 | -1 | 0.052643 | 105 | None | 0 | true | false |
| 1 | demo_3 | B | 84 | 84 | 0 | 0.052986 | 105 | None | 0 | true | false |
| 1 | demo_3 | C | 84 | 83 | -1 | 0.053719 | 103 | None | 0 | true | false |
| 1 | demo_3 | D | 84 | 84 | 0 | 0.054091 | 103 | None | 0 | true | false |
| 1 | demo_4 | A | 86 | 85 | -1 | 0.055987 | 109 | None | 0 | true | false |
| 1 | demo_4 | B | 86 | 86 | 0 | 0.055986 | 110 | None | 0 | true | false |
| 1 | demo_4 | C | 86 | 85 | -1 | 0.054179 | 106 | None | 0 | true | false |
| 1 | demo_4 | D | 86 | 86 | 0 | 0.053869 | 106 | None | 0 | true | false |
| 1 | demo_5 | A | 85 | 84 | -1 | 0.052536 | 88 | None | 0 | true | true |
| 1 | demo_5 | B | 85 | 85 | 0 | 0.051884 | 89 | None | 0 | true | false |
| 1 | demo_5 | C | 85 | 84 | -1 | 0.050299 | 87 | None | 0 | true | false |
| 1 | demo_5 | D | 85 | 85 | 0 | 0.050056 | 88 | None | 0 | true | false |
| 2 | demo_0 | A | 80 | 79 | -1 | 0.060460 | 101 | None | 0 | true | true |
| 2 | demo_0 | B | 80 | 80 | 0 | 0.059685 | 102 | None | 0 | true | true |
| 2 | demo_0 | C | 80 | 79 | -1 | 0.057178 | 100 | None | 0 | true | true |
| 2 | demo_0 | D | 80 | 80 | 0 | 0.056638 | 99 | None | 0 | true | true |
| 2 | demo_27 | A | 80 | 79 | -1 | 0.071150 | 83 | None | 0 | true | false |
| 2 | demo_27 | B | 80 | 80 | 0 | 0.071888 | 85 | None | 0 | true | false |
| 2 | demo_27 | C | 80 | 79 | -1 | 0.064039 | 85 | None | 0 | true | false |
| 2 | demo_27 | D | 80 | 80 | 0 | 0.063405 | 87 | None | 0 | true | false |
| 2 | demo_32 | A | 80 | 79 | -1 | 0.061255 | 99 | None | 0 | true | false |
| 2 | demo_32 | B | 80 | 80 | 0 | 0.060497 | 100 | None | 0 | true | false |
| 2 | demo_32 | C | 80 | 79 | -1 | 0.058030 | 85 | None | 0 | true | true |
| 2 | demo_32 | D | 80 | 80 | 0 | 0.057188 | 99 | None | 0 | true | false |
| 2 | demo_33 | A | 80 | 79 | -1 | 0.062731 | 105 | None | 0 | true | true |
| 2 | demo_33 | B | 80 | 80 | 0 | 0.061866 | 99 | None | 0 | true | true |
| 2 | demo_33 | C | 80 | 79 | -1 | 0.058439 | 85 | None | 0 | true | true |
| 2 | demo_33 | D | 80 | 80 | 0 | 0.057990 | 100 | None | 0 | true | false |
| 2 | demo_52 | A | 80 | 79 | -1 | 0.062001 | 105 | None | 0 | true | true |
| 2 | demo_52 | B | 80 | 80 | 0 | 0.061273 | 99 | None | 0 | true | true |
| 2 | demo_52 | C | 80 | 79 | -1 | 0.058393 | 100 | None | 0 | true | true |
| 2 | demo_52 | D | 80 | 80 | 0 | 0.057830 | 87 | None | 0 | true | true |
| 3 | demo_0 | A | 75 | 74 | -1 | 0.049303 | 79 | None | 0 | true | true |
| 3 | demo_0 | B | 75 | 75 | 0 | 0.049840 | 79 | None | 0 | true | false |
| 3 | demo_0 | C | 75 | 74 | -1 | 0.050196 | 77 | None | 0 | true | true |
| 3 | demo_0 | D | 75 | 75 | 0 | 0.050681 | 77 | None | 0 | true | true |
| 3 | demo_3 | A | 100 | 99 | -1 | 0.048593 | 107 | None | 0 | true | false |
| 3 | demo_3 | B | 100 | 100 | 0 | 0.047364 | 108 | None | 0 | true | false |
| 3 | demo_3 | C | 100 | 99 | -1 | 0.044232 | 107 | None | 1 | false | true |
| 3 | demo_3 | D | 100 | 100 | 0 | 0.043326 | 108 | None | 2 | false | true |
| 3 | demo_5 | A | 83 | 82 | -1 | 0.051335 | 80 | None | 0 | true | true |
| 3 | demo_5 | B | 83 | 83 | 0 | 0.051315 | 80 | None | 0 | true | true |
| 3 | demo_5 | C | 83 | 82 | -1 | 0.050844 | 77 | None | 0 | true | true |
| 3 | demo_5 | D | 83 | 83 | 0 | 0.051204 | 77 | None | 0 | true | true |
| 3 | demo_6 | A | 83 | 82 | -1 | 0.054179 | 80 | None | 3 | true | true |
| 3 | demo_6 | B | 83 | 83 | 0 | 0.054229 | 80 | None | 0 | true | false |
| 3 | demo_6 | C | 83 | 82 | -1 | 0.054674 | 77 | 118 | 50 | true | true |
| 3 | demo_6 | D | 83 | 83 | 0 | 0.054715 | 77 | 116 | 68 | false | true |
| 3 | demo_10 | A | 75 | 74 | -1 | 0.048567 | 78 | None | 0 | true | true |
| 3 | demo_10 | B | 75 | 75 | 0 | 0.049318 | 79 | None | 0 | true | true |
| 3 | demo_10 | C | 75 | 74 | -1 | 0.050097 | 77 | None | 0 | true | true |
| 3 | demo_10 | D | 75 | 75 | 0 | 0.051354 | 77 | None | 0 | true | true |
| 4 | demo_0 | A | 79 | 78 | -1 | 0.051714 | 82 | None | 0 | true | true |
| 4 | demo_0 | B | 79 | 79 | 0 | 0.051767 | 83 | None | 0 | true | false |
| 4 | demo_0 | C | 79 | 78 | -1 | 0.051473 | 82 | None | 0 | true | false |
| 4 | demo_0 | D | 79 | 79 | 0 | 0.051648 | 83 | None | 0 | true | false |
| 4 | demo_1 | A | 85 | 84 | -1 | 0.039218 | 90 | None | 0 | true | true |
| 4 | demo_1 | B | 85 | 85 | 0 | 0.039000 | 91 | None | 0 | true | true |
| 4 | demo_1 | C | 85 | 84 | -1 | 0.040012 | 88 | None | 0 | true | true |
| 4 | demo_1 | D | 85 | 85 | 0 | 0.040865 | 88 | None | 0 | true | true |
| 4 | demo_2 | A | 85 | 84 | -1 | 0.039781 | 90 | None | 0 | true | true |
| 4 | demo_2 | B | 85 | 85 | 0 | 0.039813 | 91 | None | 0 | true | true |
| 4 | demo_2 | C | 85 | 84 | -1 | 0.040680 | 90 | None | 0 | true | true |
| 4 | demo_2 | D | 85 | 85 | 0 | 0.041390 | 91 | None | 0 | true | true |
| 4 | demo_5 | A | 85 | 84 | -1 | 0.040605 | 90 | None | 0 | true | true |
| 4 | demo_5 | B | 85 | 85 | 0 | 0.041037 | 91 | None | 0 | true | true |
| 4 | demo_5 | C | 85 | 84 | -1 | 0.041858 | 90 | None | 0 | true | true |
| 4 | demo_5 | D | 85 | 85 | 0 | 0.042507 | 91 | None | 0 | true | true |
| 4 | demo_6 | A | 81 | 80 | -1 | 0.044053 | 87 | None | 0 | true | true |
| 4 | demo_6 | B | 81 | 81 | 0 | 0.043441 | 88 | None | 0 | true | true |
| 4 | demo_6 | C | 81 | 80 | -1 | 0.042729 | 86 | None | 0 | true | true |
| 4 | demo_6 | D | 81 | 81 | 0 | 0.042694 | 87 | None | 0 | true | true |

The companion JSON retains the close-switch +/-5-step EEF error, contact, grasp, actual finger-joint, aperture, and closed-without-bowl evidence for every group and trajectory.
