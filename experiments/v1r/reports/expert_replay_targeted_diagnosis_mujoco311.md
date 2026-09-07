# V1-R Expert Replay Targeted Diagnosis

Runtime: robosuite `1.4.1`, MuJoCo `3.1.1`.

Scope: layout 1 `demo_0`, layout 2 `demo_0`, and layout 3 `demo_3`; first 20 transitions plus full success replay.

| layout | demo | stale pos (m) | stale ori (rad) | delta baseline | delta + sync | saved absolute target |
|---:|---|---:|---:|---|---|---|
| 1 | demo_0 | 0.0132485429 | 0.0289126401 | False | False | False |
| 2 | demo_0 | 0.0132485429 | 0.0289126401 | False | False | False |
| 3 | demo_3 | 0.00748728319 | 0.0563111829 | True | True | False |

- Action semantics: saved configuration is delta OSC; `actions_abs` is absent.
- Time alignment: collection stores `state[t]` before executing `action[t]`; saved targets reconstruct from `eef_pose[t] + scaled action[t]`.
- Controller restoration: `reset_to()` restores MuJoCo state but leaves the controller and Panda gripper caches stale. The recheck resets `PandaGripper.current_action` and synchronizes the OSC current and null-space reference joints before each mode; layout 1/2 still fail after this audit repair.
- Absolute-target check: `datagen_info/target_pose` is converted to the world-frame absolute OSC representation used by robosuite, without changing position/orientation scaling or the saved gripper command. This is a counterfactual diagnostic only, not an adopted action contract.
- Units: position m, orientation angle rad, linear velocity m/s, angular velocity rad/s, Panda finger joint position m.

The JSON companion contains per-frame restored references, continuous replay telemetry, actual controller targets, and separated error channels.
