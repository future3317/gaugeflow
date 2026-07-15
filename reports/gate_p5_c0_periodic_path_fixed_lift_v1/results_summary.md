# P5-C0 periodic-path audit and fixed-lift result

P5-C0 is complete and **did not pass**. Its thresholds, 64 fixed sources,
33-point time grid, model capacity, seed, 5,000-step budget and historical
D0.4--D0.8 results were not changed.

## Read-only branch audit

| Quantity | Result |
|---|---:|
| Sources / time points | 64 / 33 |
| Unique type-preserving assignments | 1 |
| Temporal lift-change rows | 234 |
| Temporal permutation-change rows | 0 |
| Perturbation lift/permutation changes | 1 / 0 |
| Material branch events | 112 |
| Maximum temporal path jump RMS | 0.463271 |
| Maximum `1e-4` perturbation target jump RMS | 0.249983 |

The endpoint types are `[5, 7, 14, 32]`, so each species occurs once and there
is no type-preserving permutation ambiguity. The exact audit nevertheless
confirms material switching of the optimal periodic lift/translation gauge.
This authorizes only the pre-registered fixed-lift repair.

## Repair

For each source--endpoint coupling, the runtime solves and freezes
`(pi*, K*, tau*)` once. The integer search is complete inside a rigorous finite
radius obtained from a feasible-cost upper bound and the lattice's smallest
singular value. The entire analytic path, supervision target and sampler then
remain on that universal-cover lift. They do not call a time-dependent torus
Log, Hungarian assignment, or translation alignment, and they do not wrap
coordinates before terminal decoding. The shared PBC edge primitive centers
its local image shell on the relative coordinate, making distance/RBF features
invariant to arbitrary universal-cover integer lifts.

## Frozen training result

| Metric | Result | Pass limit | Pass? |
|---|---:|---:|:---:|
| Mean 33-time velocity MSE | 0.00374435 | <= 0.001 | No |
| Mean 33-time map MSE | 0.000843078 | report-only | -- |
| Teacher-forced quotient RMS | 0.0235882 | <= 0.02 | No |
| 100-step free-running quotient RMS | 0.130658 | <= 0.05 | No |
| Sampling failures | 0 | 0 | Yes |

The free-running RMS curve is `0.08355, 0.11127, 0.11660, 0.12141, 0.12918,
0.13066` for `1, 2, 4, 8, 32, 100` steps. More, smaller Euler steps therefore
do not cure the error and instead expose accumulated vector-field error.

## Interpretation and stop decision

The data audit found real issues in the future real-tensor benchmark and fixed
them in a separately versioned full-O(3) v2 artifact, but those split/target
issues cannot cause this result: P5-C0 is a synthetic single-endpoint test with
frozen sources and no tensor condition. Within P5-C0, dynamic periodic branch
switching was a genuine target-definition defect. Fixing it materially improves
D0.4, yet it is insufficient for qualification. The remaining failure is in
full-time vector-field fitting and free-running error accumulation, not source
data provenance, tensor labels, node permutation, non-finite sampling, or the
now-fixed dynamic lift target.

P5-D1 and all harmonic, oracle, real-tensor, relaxation, DFT and DFPT work
remain prohibited by this result.
