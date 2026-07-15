# P5-D0.7 long-range stability cycle: frozen outcome

This directory contains the bounded D0.7/D0.8 cycle. D0.6 and all earlier
reports are historical inputs and have not been modified.

## Phase 1: read-only D0.6 diagnosis

The D0.6 runner did not retain a checkpoint, so the diagnostic deterministically
reconstructed its exact 64-source, seed-5201, 5,000-step setup without saving a
new checkpoint. `phase1_diagnostics/span_and_semigroup.csv` shows a clear
long-span failure: direct map MSE grows from `3.92e-5` for span `1/32` to
`1.869e-2` for span `1`; quotient semigroup defect RMS grows from `0.00507` to
`0.11102`. Final-RMS sampling deteriorates with composition: `0.0947` (one
map), `0.0914` (two), `0.1084` (four), and `0.2299` (100).

The fixed 0.001 quotient perturbation has a first learned-step amplification
of `3.5566`, a mean step amplification of `1.1749`, and reaches RMS `0.05351`
after 32 maps. This selects the contractive/Lipschitz option if D0.7 fails.
The accompanying `rationale.md` cites Flow Map Matching and the original
Consistency Models papers; these motivate map composition and cross-resolution
consistency, but do not constitute a claim that the present model satisfies
them.

## D0.7: multiscale semigroup-consistent quotient flow map

The protocol uniformly samples `1/32, 1/16, 1/8, 1/4, 1/2, 1` horizons,
reserves 25% of direct maps for an endpoint, and uses equal energy-normalized
analytic direct-map, differentiable two-map rollout, and semigroup losses.
Its only run retained excellent local map MSE (`3.9098e-5`) and zero sampling
failures, but failed the frozen RMS criteria:

| Metric | Result | Limit |
|---|---:|---:|
| Adjacent 33-grid flow-map MSE | `3.91e-5` | `<= 1e-3` |
| Mean teacher-forced endpoint RMS | `0.11001` | `<= 0.02` |
| 100-map free-running RMS | `0.22484` | `<= 0.05` |
| Sampling failures | `0` | `0` |

Therefore D0.7 did not qualify the coordinate substrate.

## D0.8: single contractive follow-up

The only authorized follow-up retained every D0.7 setting and added one
unit-weight finite-difference penalty for map expansion above quotient
Lipschitz constant one. It used a fresh zero-mean 0.001 RMS quotient
perturbation on every update. The constraint substantially reduced the
post-training perturbation amplification, but accurate learned maps did not
coexist with that reduction under the frozen objective:

| Metric | D0.7 | D0.8 | Limit |
|---|---:|---:|---:|
| Adjacent flow-map MSE | `3.91e-5` | `6.19e-5` | `<= 1e-3` |
| Teacher-forced RMS | `0.11001` | `0.12562` | `<= 0.02` |
| 100-map RMS | `0.22484` | `0.24745` | `<= 0.05` |
| First-step perturbation amplification | `3.5566` | `1.0154` | diagnostic |
| Mean perturbation amplification | `1.1749` | `1.0333` | diagnostic |
| Sampling failures | `0` | `0` | `0` |

The D0.8 training trace shows the direct normalized loss close to one at the
end, so this is not evidence that a stable accurate flow map was learned. It
is a negative trade-off result: a local finite-difference expansion constraint
suppressed measured expansion but did not repair the off-path or endpoint map.

No D0.9, P5-D1, harmonic, oracle, or real-tensor experiment is authorized by
this cycle. The next scientific change must be proposed separately rather than
retuning any D0.7/D0.8 parameter after observing these results.
