# V1-R.2J Executable Absolute-Action Label Contract

## Scope

- Same 20 V1-R.2I sequential-success demonstrations; no policy training or inference.
- Runtime: robosuite `1.4.1`, MuJoCo `3.3.5`, `OSC_POSE`, 20 Hz.
- Strict gate: `20/20`; base and post-success-tail results are separate.
- Every absolute replay records its input action, controller goal, target error, delta-reference state error, first divergence, contact, close command, grasp diagnostic, success, and failure stage.

## Base Gate

| Layout | S0-old | S0-transition | S1-PB | S1-world |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1/5 | 0/5 | 4/5 | 5/5 |
| 2 | 0/5 | 0/5 | 5/5 | 5/5 |
| 3 | 1/5 | 2/5 | 3/5 | 3/5 |
| 4 | 1/5 | 1/5 | 4/5 | 4/5 |
| **Total** | **3/20** | **3/20** | **16/20** | **17/20** |

## Post-Success Tail Diagnostic

The predeclared rule appends ten real remaining HDF5 delta transitions when available. When fewer than ten remain, it appends five repeats of each contract's final absolute target. The rule is selected from source length before replay outcomes are known.

| Layout | S0-old | S0-transition | S1-PB | S1-world |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2/5 | 1/5 | 5/5 | 5/5 |
| 2 | 0/5 | 0/5 | 5/5 | 5/5 |
| 3 | 4/5 | 4/5 | 3/5 | 3/5 |
| 4 | 5/5 | 4/5 | 5/5 | 5/5 |
| **Total** | **11/20** | **9/20** | **18/20** | **18/20** |

## Invariants

- Same demonstrations: `True`.
- HDF5 action-prefix matches: `20/20`.
- Exact initial-state matches: `20/20`.
- Delta reference successes: `20/20`.
- Exact regenerated base-state sequences: `20/20`.
- 2I S0-old/S1-PB outcomes reproduced: `True`.
- Simulator exceptions: `0`.

## Decision

Decision case: `D`.
Selected label contract: `None`.
Next action: `register_delta_pose_contract_and_verify_normalization_execution_roundtrip`.

B0/B1 training, confirm rollouts, V2, and V3 remain unauthorized. Seed-0 authorization, if present, applies only after rebuilding data with the frozen contract; it does not authorize confirm, V2, or V3.

Five-step stable grasp remains diagnostic and is not substituted for task success.

## Local Evidence

- `outputs/v1r/executable_absolute_contracts_2j/results.json`: `c3127a4edcf16b69dc2c8157c122a91eba775dff12aaadf2cc05e86833247dbc` (52489778 bytes; ignored, not uploaded).
- Per-step telemetry remains in that local result. The tracked JSON contains compact per-trajectory summaries only.
