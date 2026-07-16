# Revised-paper architecture compliance

Design source: `GaugeFlow_PiezoGen_Revised.tex`, SHA-256
`9ad4ed018600a62b5f663255a1e0a4d59abcdc26303e523a4f151bdfaf07dd31`.

This table describes code availability, not scientific performance. Historical
negative results remain evidence about the old prototype only.

| Paper component | Current implementation | Qualification state |
|---|---|---|
| 118 elements + absorbing MASK | `gaugeflow.production.categorical_mask` | S0 tests implemented |
| Translation-quotient wrapped Gaussian | `gaugeflow.production.wrapped_coordinates` | S0 tests implemented for exact small-site evaluation; resource guard fails closed |
| Log-volume / trace-free log-shape lattice | `gaugeflow.production.lattice_volume_shape` | S0 round-trip and symmetry tests implemented |
| Full-O(3) Reynolds compatibility | `gaugeflow.production.space_group_router` | S0 point-group/rank/router tests implemented |
| Exact asymmetric-unit expansion | `gaugeflow.production.symmetry_expand` | S0 expansion test implemented |
| Harmonic relative-frame posterior | `gaugeflow.production.harmonic_gaugeflow` | S0 covariance, zero/null, and grid tests implemented |
| Time/condition-in-every-block backbone | `gaugeflow.production.equivariant_denoiser` | S0 head/block and quotient tests implemented |
| Structural pretraining and reverse sampler | Not implemented in this S0 change | S1 locked |
| Complete Wyckoff autoregressive blueprint | Not implemented in this S0 change | S2 locked |
| Exact synthetic tensor control | Historical prototype result is inapplicable | S3 locked |
| MLIP representation transfer | Not implemented; no teacher selected | S1 ablation locked |
| Multi-task piezoelectric encoder | Existing preparation is not this production encoder | S4 locked |
| Real tensor / relaxation / DFT / DFPT | Not run | S4/S5 locked |

## Correctness decisions

- Pymatgen point-group matrices are first interpreted in their crystallographic
  fractional basis. A group-invariant metric is constructed and used to
  conjugate the entire finite group into Cartesian O(3). Per-operation polar
  projection of fractional matrices is forbidden because it breaks group
  multiplication for hexagonal and trigonal settings.
- Harmonic geometry queries use condition-free directed-edge weights derived
  from the current element, metric, and time state. An unweighted sum over a
  bidirectional edge list would cancel the polar odd-degree queries.
- The wrapped kernel expands integer images adaptively and stops only after a
  Gaussian-tail bound closes. Resource exhaustion raises an error; it never
  switches to a fixed image cube.
- A one-site asymmetric unit has no coordinate degree of freedom after global
  translation quotienting and therefore has an identically zero coordinate
  score.

## Remaining blocker

Passing S0 qualifies only the mathematics and software contracts. It does not
show that the new generator produces valid crystals. S1 requires a separately
versioned training corpus, reverse hybrid sampler, fixed three-seed protocol,
at least 10,000 samples per seed, and frozen MLIP qualification criteria.
