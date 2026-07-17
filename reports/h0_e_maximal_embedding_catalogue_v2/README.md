# H0-E-v2 E0 maximal embedding catalogue

E0 is qualified within its frozen scope. It is a prerequisite for the
parent-occurrence E1 pilot, not a qualification of H0-E and not authorization
for H1a.

The offline source is the MIT-licensed PyXtal 0.6.1 packaged maximal
translationengleiche/klassengleiche subgroup and Wyckoff-splitting data. All
source files and the license are pinned by SHA-256. GaugeFlow does not import
PyXtal at generation runtime.

The compiler retained all 1,103 maximal t-subgroup records and all 2,641
k-subgroup records with index at most four. After exact rationalization and
physical duplicate aggregation, 3,744 raw rows became 2,843 affine embeddings
with 2,845 normalized Wyckoff relation variants. The 901 duplicate source rows
are provenance multiplicity only and cannot change candidate measure.

Every group setting was independently identified by spglib. Every edge passed
the vectorized Seitz inclusion check

```text
T H T^-1 subset G modulo parent integer translations.
```

The exhaustive independent audit recomputed all embedding keys, rational
denominators, relation labels, source hashes, reverse-child ordering and
inclusion certificates. Maximum float64 rotation and periodic-translation
errors were `2.22e-16` and `4.44e-16`. The deterministic compressed artifact is
178,974 bytes with SHA-256
`3a1bb3ad08ce576fe11bfa0515314da9d8ecb42346805f2cdfa3b975602c4747`.

The authoritative large artifacts remain under:

```text
E:/DATA/T2C-Flow/processed/gaugeflow_h0_v5/maximal_subgroup_embeddings_v2/
```

The independent audit file has SHA-256
`dfa83dbd01542d0b909e9952de9ee3e7e96dfb5a9ac909592eb72c0c1ed8c47d`.

