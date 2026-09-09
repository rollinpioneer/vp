# V1-R.2K Quantized-at-Source Result

## Decision

V1-R.2K completed under the frozen protocol. The selected contract is:

```python
label_float32 = raw_action_float64.astype(np.float32)
controller_action = label_float32.astype(np.float64)
```

No dataset min-max normalization, clipping, smoothing, extra scaling, or
second quantization was used. Capture and replay call the same implementation
in `experiments/v1r/scripts/numeric_contract.py`.

## Runtime

- Repository commit: `491db4c4652ebfbcfce241c1c5e83c6d0c9eea75`
- Point Bridge commit: `5d567a62d62b5a97c5960d45024e065349680cda`
- Python: `/home/__compress_data/xushijie/.venvs/point-bridge/bin/python`
- robosuite: `1.4.1`
- MuJoCo: `3.3.5`
- Controller: `OSC_POSE`, `control_delta=true`, 20 Hz
- PyTorch: `2.13.0+cu130`; CUDA was unavailable because the NVIDIA driver was not available. No training was run.

## Collection

Candidates were processed in ascending numeric demo order. Each candidate was
reset once, then run continuously until task success or source-action
exhaustion. All attempts were retained locally, including `layout_1/demo_18`.

| layout | attempted | accepted | contact_without_grasp | no_contact | other |
|---:|---:|---:|---:|---:|---:|
| 1 | 22 | 5 | 15 | 0 | 2 |
| 2 | 9 | 5 | 4 | 0 | 0 |
| 3 | 5 | 5 | 0 | 0 | 0 |
| 4 | 5 | 5 | 0 | 0 | 0 |
| **total** | **41** | **20** | **19** | **0** | **2** |

The prior V1-R.2J-N result on old trajectories remains unchanged. In this
source-quantized collection, `layout_1/demo_18` was accepted and later passed
the strict replay gate; this is a new causal data-generation result, not a
retroactive change to the old diagnostic.

## Strict Replay Gate

The 20 accepted records were restored from their saved initial states once and
replayed with the saved float32 labels promoted to float64. The gate passed
`20/20`. Values, ordering, and sequence lengths of replay commands matched
the shared decoder exactly. Captured and replayed state sequences were exactly
equal for all 20 records, with maximum absolute error `0.0`.

## Authorization Boundary

Passing V1-R.2K freezes the numeric contract and authorizes a separate decision
about rebuilding training data and starting a seed-0 baseline. It does not
authorize training, confirmation rollouts, V2, or V3. This run performed zero
training runs, zero API calls, and zero confirmation rollouts.

Raw HDF5, NPZ, and XML artifacts remain outside Git. The runtime manifest and
verification JSON are retained locally; tracked reports record their paths,
runtime, hashes, and intended recovery location.
