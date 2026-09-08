# V1-R.2J-N Numeric Branch Summary

The installed robosuite `1.4.1` file `robosuite/controllers/osc.py` contains the `math.isclose(elem, 0.0)` orientation-update branch. Its local SHA-256 is `cfa62a0e719bc53ef0c701efa66f7b3e2272d4fca2150ec73c05bb145eba85cb`.

Across all four selected trajectories:

- raw float64, direct float32, and min-max roundtrip paths each had zero all-zero rotation-action steps;
- raw-to-direct-float32 branch decision changes: `0`;
- raw-to-min-max-roundtrip branch decision changes: `0`;
- raw zero rotation components changed to nonzero: `0`.

The proposed zero/nonzero branch-switch mechanism is therefore not observed in these records.

The physical divergence is trajectory dependent. `demo_10` under min-max roundtrip first exceeds `1e-6 m` EEF difference at action 80, `1e-4 m` at 85, and `1e-3 m` at 89, eventually reaching `0.05796 m` EEF and `0.14694 m` bowl difference. `demo_18` exceeds `1e-4 m` EEF difference only near action 165/166 under the two float32 paths. `demo_2` changes the terminal outcome without exceeding `1e-6 rad` orientation difference; its position difference appears at the final action. The successful control `demo_27` never exceeds `1e-6 m` EEF or `1e-6 rad` orientation difference.

These thresholds are descriptive diagnostics, not tuned controller thresholds and not acceptance criteria.
