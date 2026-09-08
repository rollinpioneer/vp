# V1-R.2I Sequential Success Demonstration Gate

## Scope

- Runtime: robosuite `1.4.1`, MuJoCo `3.3.5`, `OSC_POSE`, delta commands, 20 Hz.
- Candidate command source: the official HDF5 7-D delta action streams.
- Each candidate restores its initial state once; every later state is produced by `env.step()`.
- The old saved-state PKL generator and patch `0001-mimiclabs-saved-state-generator.patch` are not called.
- Task success is the acceptance predicate. Five-step stable grasp remains diagnostic only.

## Sequential Capture

| Layout | Candidates executed | Continuous successes | Accepted demos |
| --- | ---: | ---: | --- |
| 1 | 22 | 5 | demo_2, demo_3, demo_18, demo_21, demo_25 |
| 2 | 9 | 5 | demo_27, demo_32, demo_33, demo_59, demo_73 |
| 3 | 5 | 5 | demo_0, demo_3, demo_5, demo_6, demo_10 |
| 4 | 5 | 5 | demo_0, demo_1, demo_2, demo_5, demo_6 |

All 41 attempted initial states restored exactly; simulator exceptions: 0. Intermediate state restores: 0.

## Replay Gates

| Layout | Actual issued commands | S0 original labels | S1 recorded targets |
| --- | ---: | ---: | ---: |
| 1 | 5/5 | 1/5 | 4/5 |
| 2 | 5/5 | 0/5 | 5/5 |
| 3 | 5/5 | 1/5 | 3/5 |
| 4 | 5/5 | 1/5 | 4/5 |

The actual-command gate passed `20/20`. S0 passed `3/20` and S1 passed `16/20`; neither met the strict `20/20` label threshold.

## Decision

V1-R.2I is complete as a sequential demonstration reconstruction experiment, but the absolute training-label contract remains blocked. No S0/S1 contract is selected, and B0/B1 training, confirm rollouts, V2, and V3 remain unauthorized.

The next action is limited to diagnosing and reconstructing an executable absolute label contract on these same 20 sequential demonstrations. It is not a replay of V1-R.2H and does not authorize policy retraining.

None of the 20 accepted demonstrations met the five-consecutive-step stable-grasp diagnostic, while all 20 met task success and actual-command replay. This diagnostic therefore remains non-gating. These 20 demonstrations validate the contract sample only and do not establish future data coverage.

## Local Evidence

- `outputs/v1r/sequential_success_demos_2i/manifest.json`: `87b4ac57bd02c8ab77ef81e5e22cfafb777f3682b7c4c8c0092083c15ac1786f`
- `outputs/v1r/sequential_success_demos_2i/verification.json`: `8227453583f7a424da61fc8a4cdb98a058dfd5af61a73090ec5928b6eed94679`
