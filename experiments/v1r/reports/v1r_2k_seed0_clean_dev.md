# V1-R.2K seed-0 clean-dev

## Decision

- Stage: `V1-R.2K.seed0`
- Status: `completed_seed0_clean_dev_failed`
- Decision: `blocked_seed0_clean_dev`
- Variant: `B1-2K-20`, seed `0`
- Primary checkpoint: `300000.pt`
- Primary checkpoint SHA-256: `e827f24d228c295c3f5f09d0d051692d573f73d7b41a181eefe4f84d590f7589`

The checkpoint was frozen before clean-dev evaluation. The 100000-step and
200000-step snapshots remain diagnostic only and were not selected using the
clean-dev outcome.

## Runtime and data contract

- Device: CUDA, NVIDIA A100-SXM4-40GB
- PyTorch: `2.8.0+cu126`
- MuJoCo: `3.3.5`; robosuite: `1.4.1`
- Action contract: `delta_pose_float32_identity`
- Dataset min-max normalization: disabled
- Dataset: balanced 20-episode pilot, four layouts, five episodes per layout
- Resolved training config SHA-256: `fdd53b580f625c0325da6126684fe31dadea6177f3b431ce404340549559c3cb`
- Dataset manifest SHA-256: `761254629727b0035a794f86fc7ceca6f2e5fb429282d5259f37a53ad7c39c5e`

The training job completed 300000 effective steps with return code 0. This
confirms that the authorized CUDA training run completed; it does not imply
that the policy passed clean-dev.

## Frozen clean-dev result

| Layout | Successes | Rollouts |
|---|---:|---:|
| 1 | 0 | 10 |
| 2 | 2 | 10 |
| 3 | 0 | 10 |
| 4 | 1 | 10 |
| **Total** | **3** | **40** |

- Success rate: `3/40 = 0.075`
- Required gate: at least `20/40 = 0.50`
- Current initial-state matches: `40/40`
- Historical compatibility hash matches: `35/40` (diagnostic only)
- Simulator exceptions: `0`
- Action decode errors: `0`
- Failure stages: `no_approach=9`, `no_grasp=22`, `post_grasp_drop=6`
- Final rollout CSV SHA-256: `3bff63cdde28a8a4df102ee605fe999271e47974000d63e9f6634ac81e0f078c`

The repaired dedicated entry was then run independently over all 40 rows. Its
authoritative output is
`outputs/v1r/training/v1r_b1_2k_20_seed0/clean_dev_seed0_verified.csv` with
SHA-256 `b46682d68579a6bde8408d66511c410a95443e8a53da4622d60f238c74fa53b2`.
The verified JSON summary has SHA-256
`6341fd112599e9e4b47e737bc6f4b5c110d325e3e7af7292b92128fa70cbcc81` and
records the same `3/40` result using CUDA and the frozen primary checkpoint.

## Authorization boundary

The clean-dev gate failed. Do not run confirm rollouts, remaining seeds, V2,
or V3. The next stage is diagnosis of pilot coverage and learnability. The
passed action/data contracts remain valid and are not retroactively rejected
because the first policy baseline was weak.
