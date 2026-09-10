# V1-R.2K seed-0 checkpoint learning-curve diagnostic

## Scope

This is a read-only diagnostic of the three checkpoints selected before the
clean-dev run. All checkpoints used the same 40 frozen dev initial states,
CUDA runtime, runner, success predicate, point inputs, and action contract.
The 100000-step and 200000-step runs use the explicit diagnostic-only entry;
the 300000-step row is the already completed formal clean-dev result.

This report does not select a checkpoint, reopen training authorization, or
change the formal clean-dev result.

## Results

| checkpoint | role | successes | per layout (1/2/3/4) | no approach | no grasp | post-grasp drop | initial states | simulator exceptions | action decode errors |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 100000 | diagnostic only | 2/40 | 0/10, 1/10, 0/10, 1/10 | 6 | 21 | 11 | 40/40 | 0 | 0 |
| 200000 | diagnostic only | 6/40 | 1/10, 2/10, 0/10, 3/10 | 4 | 24 | 6 | 40/40 | 0 | 0 |
| 300000 | frozen formal clean-dev | 3/40 | 0/10, 2/10, 0/10, 1/10 | 9 | 22 | 6 | 40/40 | 0 | 0 |

## Interpretation

The same-seed curve improves from 2/40 at 100000 steps to 6/40 at 200000
steps, then falls to 3/40 at the pre-frozen 300000-step checkpoint. This is
evidence of non-monotonic checkpoint behavior on this small pilot, with a
local mid-training improvement that was not retained at the final checkpoint.

The improvement is not uniform across layouts: layout 3 remains 0/10 at all
three checkpoints, while the 200000-step gain is concentrated in layouts 1,
2, and 4. The failure-stage counts also change, but they are diagnostic
endpoint categories rather than root-cause labels.

The result does not establish multi-seed reproducibility, generalization,
that the action contract is invalid, or that 200000 steps should replace the
pre-registered 300000-step clean-dev checkpoint. The formal gate remains
`blocked_seed0_clean_dev` at `3/40`; confirm rollouts, remaining seeds, V2,
and V3 remain unauthorized.

## Reproducibility identities

- Point Bridge source: `491db4c4652ebfbcfce241c1c5e83c6d0c9eea75`
- Resolved training config SHA-256: `fdd53b580f625c0325da6126684fe31dadea6177f3b431ce404340549559c3cb`
- Dataset manifest SHA-256: `761254629727b0035a794f86fc7ceca6f2e5fb429282d5259f37a53ad7c39c5e`
- Checkpoint selection SHA-256: `e9ce902fffe00da4744a730e0c937e985b3a66855971c128c640c3e0f6c317c8`
- 100000 checkpoint SHA-256: `afeb91d34deee6edd353d4cda6de7198f7ba08ed97f512d26bf2e42765095716`
- 200000 checkpoint SHA-256: `297b9e614bcd7196b5c2e04743d42c57c6296dd41721f0f5470a8f198b03892a`
- 300000 checkpoint SHA-256: `e827f24d228c295c3f5f09d0d051692d573f73d7b41a181eefe4f84d590f7589`
- 100000 rollout CSV SHA-256: `1bcfd159b8e94fda1d8ee3293e255c3f2673709ebbbad954036734770628c122`
- 100000 summary JSON SHA-256: `a1e3ab07680a9d4484722b79212cf4bed06699749f12e3870dd6e8788486027e`
- 200000 rollout CSV SHA-256: `bc2a2983dcd5de8ff10ad030d7ba9a63c58ca353911f0513c8c367374cff07ed`
- 200000 summary JSON SHA-256: `c7b764a559ffaebf4508780471534ecf3a9d9b88aea211c65b3746668f72a056`

Raw rollout CSV/JSON files remain local under the gitignored `outputs/v1r`
directory and are listed in `manifests/not_uploaded_files.csv`.
