# Gate A4.0 analytic probability-path closure

Protocol: `gate_a4_generator_substrate_v1.json` (`b31c80b7f14f233e8fff1e1b2b5c9d03b057bcf02437aaff2995bb17783e7bb4`).

This test uses no neural network: it integrates the exact production interpolant velocity with the production Euler sampler and a fixed base noise.

| Subspace | rows | max continuous endpoint error | decoded endpoint accuracy | closed |
|---|---:|---:|---:|---:|
| type | 2 | 1.314e-07 | 1.000 | True |
| coordinate | 2 | 1.827e-07 | 1.000 | True |
| lattice | 2 | 1.312e-07 | 1.000 | True |
| joint | 2 | 3.109e-07 | 1.000 | True |

Decision: **PASS** analytic closure.
The production time direction, constant velocity target, torus wrapping, SPD-log lattice coordinate, and final type argmax recover the chosen endpoints under exact velocity. A4 may therefore attribute any subsequent failure to learned generator substrate behavior rather than this analytic integration identity.
