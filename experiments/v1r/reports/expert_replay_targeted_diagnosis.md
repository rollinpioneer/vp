# V1-R Expert Replay Targeted Diagnosis

Runtime: robosuite `1.4.1`, MuJoCo `3.3.5`.

Scope: layout 1 `demo_0`, layout 2 `demo_0`, and layout 3 `demo_3`; first 20 transitions plus full success replay.

| layout | demo | stale pos (m) | stale ori (rad) | delta baseline | delta + sync | saved absolute target |
|---:|---|---:|---:|---|---|---|
| 1 | demo_0 | 0.0132485429 | 0.0289126401 | False | False | False |
| 2 | demo_0 | 0.0132485429 | 0.0289126401 | False | False | True |
| 3 | demo_3 | 0.00748728319 | 0.0563111829 | True | True | True |

## First Explanatory Difference

The first mismatch is action execution after exact state restoration, not action decoding: after controller synchronization, action 0 produces the saved controller target, but the post-step EEF state diverges. In delta OSC, the next target is based on that actual EEF, so the target error from action 1 onward carries the previous EEF error.

| layout | action 0 target err (mm) | action 0 EEF err (mm) | action 0 target tracking (mm) | first translation saturation | max EEF err in first 20 (mm) | carry-over residual (mm) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.22045e-13 | 0.221033 | 27.2218 | 2 | 0.774286 | 0.122674 |
| 2 | 2.22045e-13 | 0.40556 | 48.0779 | 0 | 0.733501 | 0.323833 |
| 3 | 8.67362e-16 | 0.0603507 | 7.77889 | none | 0.272521 | 0.00981767 |

`initial_joint` is set once from restored `state[0]` and held fixed during continuous replay; it is not updated per frame.

## Time Alignment

| layout | expected action interval (s) | actual interval max error (s) | supported |
|---:|---:|---:|---|
| 1 | 0.05 | 0 | True |
| 2 | 0.05 | 0 | True |
| 3 | 0.05 | 0 | True |

- Action semantics: saved configuration is delta OSC; `actions_abs` is absent.
- Time alignment: collection stores `state[t]` before executing `action[t]`; saved targets reconstruct from `eef_pose[t] + scaled action[t]`.
- Controller restoration: `reset_to()` restores MuJoCo state but leaves the controller cache at the deterministic reset pose until explicitly synchronized; the sync-only result is reported separately.
- Absolute-target check: `datagen_info/target_pose` is converted to the world-frame absolute OSC representation used by robosuite, without changing position/orientation scaling or the saved gripper command.
- Runtime qualification: The first-step execution mismatch is present, but the strict delta-target carry-over check is not asserted for this runtime.
- Units: position m, orientation angle rad, linear velocity m/s, angular velocity rad/s, Panda finger joint position m.

The JSON companion contains per-frame restored references, continuous replay telemetry, actual controller targets, and separated error channels.
