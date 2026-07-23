# Stage-E Orderless-Partial Lattice Exposure Diagnostic

## Status

Stage-E remains blocked.  The `oracle_c` failure is now narrowed beyond the
previous counts and shape-scale bugs: the lattice adapter was trained on clean
element tokens while the joint sampler asks the lattice head to operate on
orderless partial/MASK assignment states.  Exposing the adapter to those
partial states is a real coverage repair, but by itself it does not solve the
remaining tail and it can conflict with the clean-assignment `oracle_ca` path.

## Code Changes

- `50a1c93d` added protocol field `element_exposure`.
  - default `clean` preserves historical adapter training.
  - `orderless_partial` supplies exact `composition_counts` while replacing
    lattice carrier tokens with a nested orderless partial/MASK state.
- `b747613c` restored clean-retention semantics: generated exposure uses
  partial/MASK tokens, but the retention query still uses clean element tokens.

The loss form and adapter architecture are unchanged.  The change is only the
element-state provenance of the generated lattice exposure carrier.

Validation:

```text
PYTHONPATH=src:. pytest -q \
  tests/test_tensor_conditioning.py \
  tests/test_stage_e_factorial_sample_rows.py \
  tests/test_production_training.py

ruff check scripts/train_stage_e_lattice_generated_exposure.py \
  tests/test_tensor_conditioning.py

PYTHONPATH=src:. mypy scripts/train_stage_e_lattice_generated_exposure.py \
  tests/test_tensor_conditioning.py
```

All passed on the server conda environment.

## Training Evidence

The new adapter used the same Stage-C 30k/global 40523 base, E3 tensor
checkpoint, Stage-D JARVIS train split, target-exclusion contract, seed,
optimizer, step count, batch size and loss weights as the previous counts-fixed
JARVIS adapter.  The new protocol differs only by:

```json
"element_exposure": "orderless_partial"
```

Server artifacts:

```text
/home/workspace/lrh/DATA/T2C-Flow/runs/
  stage_e_lattice_generated_exposure_jarvis_orderless_partial_smoke_v1/
  stage_e_lattice_generated_exposure_jarvis_orderless_partial_v1/
  stage_e_lattice_generated_exposure_jarvis_orderless_partial_v2/

/home/workspace/lrh/DATA/T2C-Flow/evaluations/
  stage_e_orderless_partial_shape_scale025_smoke32_v1/
  stage_e_orderless_partial_v2_shape_scale025_smoke32_v1/
  stage_e_orderless_partial_v2_shape_scale0_smoke32_v1/
```

The v2 training smoke was finite and confirmed nontrivial partial exposure:

```text
final loss = 0.603669
gradient_norm = 3.728695
first_mask_fraction = 0.740741
exposed_mask_fraction = 0.574074
```

## Smoke32 Results

Conditioned role, common smoke32 panel:

| adapter | shape scale | arm | tensor RMSE | volume-W1 | NN-W1 | failures |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| counts-fixed clean exposure | 0.25 | oracle_ca | 0.786164 | 0.079249 | 0.528087 | 0 |
| counts-fixed clean exposure | 0.25 | oracle_c | 1.015259 | 0.075922 | 0.701007 | 0 |
| counts-fixed clean exposure | 0.25 | free | 0.983278 | 0.306149 | 0.329158 | 0 |
| partial v2 | 0.25 | oracle_ca | 0.785358 | 0.157782 | 0.541020 | 0 |
| partial v2 | 0.25 | oracle_c | 0.988655 | 0.086297 | 0.667458 | 0 |
| partial v2 | 0.25 | free | 0.951712 | 0.339584 | 0.319875 | 0 |
| partial v2 | 0.00 | oracle_ca | 0.977589 | 0.149005 | 0.639191 | 0 |
| partial v2 | 0.00 | oracle_c | 0.997591 | 0.086886 | 0.673520 | 0 |
| partial v2 | 0.00 | free | 1.012761 | 0.339344 | 0.321570 | 0 |

Target 123 remains the decisive tail:

| adapter | scale | arm | tensor error | min distance | condition |
| --- | ---: | --- | ---: | ---: | ---: |
| counts-fixed clean exposure | 0.25 | oracle_c | 0.707198 | 1.457349 | 12.483171 |
| partial v2 | 0.25 | oracle_c | 0.536516 | 1.539463 | 12.530057 |
| partial v2 | 0.00 | oracle_c | 0.480039 | 1.658166 | 11.041600 |

## Interpretation

Orderless-partial exposure is a real missing carrier state: it slightly improves
`oracle_c` NN-W1 and target-123 tensor error.  However, it does not remove the
target-123 condition-number tail, and it worsens the clean-assignment
`oracle_ca` volume path.  This means the remaining Stage-E failure is not
explained by a simple absence of partial/MASK assignment exposure.

The current single shared lattice adapter is trying to serve incompatible
carrier regimes:

```text
clean/full assignment lattice carrier -> needed for oracle_ca retention
partial/MASK orderless carrier        -> needed for oracle_c joint sampling
```

With the present adapter and loss, adding partial exposure trades away part of
the clean/full-assignment volume correction without fixing the joint tail.
This rejects the partial-only adapter as a production candidate.

## Decision

- Do not run the 256-sample Stage-E Gate with the partial adapter.
- Do not start Stage-F.
- Do not keep tuning scalar shape scale around this partial adapter.
- Treat `element_exposure=orderless_partial` as a diagnostic interface option,
  not a qualified mechanism.
- The next evidence-authorized repair must either separate/regularize the
  clean/full and partial/MASK lattice carrier regimes, or freeze E-v1 and move
  to the A-v2 generated-state coverage contract described in the retraining
  plan.
