# Stage-E Shape Residual Dose Diagnostic

## Status

Stage-E remains blocked, but the generated-lattice failure has a narrower
root cause: the counts-fixed lattice adapter preserves the volume correction
while its shape residual is too large along the generated reverse trajectory.
Scaling only the shape residual is a zero-training intervention that tests
this causal path without changing checkpoints, tensor conditioning, the VP
sampler, or random streams.

## Evidence

The dose experiment used the frozen smoke32 protocol
`configs/gates/stage_e_e1a_factorial_rollout_v2_data_clean_smoke32.json` and
the counts-fixed adapter
`stage_e_lattice_generated_exposure_jarvis_countsfix_v1/adapter.pt`.

The adapter output was decomposed as:

```text
volume = base_volume + adapter_delta_volume
shape  = base_shape  + alpha * adapter_delta_shape
alpha in {0, 0.25, 0.5, 0.75, 1.0}
```

No hard clipping or retraining was used.

Server evidence:

```text
/home/workspace/lrh/DATA/T2C-Flow/evaluations/stage_e_countsfix_shape_dose_smoke32_v1/
  shape_residual_dose_quick_123_316_v1.json
  shape_residual_dose_smoke32_v1.json

/home/workspace/lrh/DATA/T2C-Flow/evaluations/stage_e_countsfix_shape_scale025_official_smoke32_v1/
  result.json
  samples.json
  trajectory.json
```

## Smoke32 Results

Conditioned role only:

| alpha | arm | tensor RMSE | volume-W1 | NN-W1 | valid distance | max condition |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.00 | oracle_ca | 0.8468 | 0.0823 | 0.5989 | 1.000 | 6.15 |
| 0.25 | oracle_ca | 0.7862 | 0.0792 | 0.5281 | 1.000 | 5.26 |
| 0.50 | oracle_ca | 0.8077 | 0.0773 | 0.5952 | 1.000 | 7.00 |
| 0.75 | oracle_ca | 0.8133 | 0.0783 | 0.6339 | 1.000 | 9.76 |
| 1.00 | oracle_ca | 0.7178 | 0.0823 | 0.6645 | 1.000 | 12.99 |
| 0.25 | oracle_c | 1.0153 | 0.0759 | 0.7010 | 1.000 | 12.48 |
| 0.25 | free | 0.9833 | 0.3061 | 0.3292 | 1.000 | 3.59 |

The official evaluator reproduced the alpha `0.25` oracle_ca result:

```text
oracle_ca conditioned:
  tensor RMSE = 0.786164
  volume-W1  = 0.079249
  NN-W1      = 0.528087
  failures   = 0
```

## Tail Targets

Target `123` is the clearest shape-tail example:

```text
alpha=0.25 oracle_ca: shape_norm=2.3092, condition=4.6776, min_distance=2.2970
alpha=1.00 oracle_ca: shape_norm=3.8058, condition=12.9924, min_distance=1.6711
```

Target `316` is not catastrophic, but it shows the same monotone shape/condition
growth as alpha increases.

## Interpretation

`alpha=0.25` preserves the volume improvement from the counts-fixed adapter and
recovers oracle_ca NN-W1 close to the no-adapter E3 baseline, while avoiding the
condition-number tail introduced by the full shape residual. This supports the
root cause:

```text
counts bug -> old adapter volume drift
full shape residual -> C-new generated-coordinate NN/shape degradation
```

The free arm is not materially improved by shape scaling, and oracle_c still has
a target-123 condition-number tail. Therefore this is a local Stage-E adapter
scale fix, not a Stage-E pass and not a reason to start Stage-F.

## Next

Use `--lattice-adapter-shape-scale 0.25` as the current diagnostic production
candidate for C-new smoke comparisons. A principled retraining follow-up should
replace the scalar dose with a data-whitened shape trust-region penalty rather
than hard clipping.
