# Stage-E E0 paired direct-condition rollout

This restricted Gate uses 64 held-out response structures only to provide the
target tensor and node count. Composition, assignment, lattice, and coordinates
are generated from paired random streams. Stage-C is the null baseline;
the selected common-noise E0 checkpoint is condition-required. The frozen
Stage-D step-4,500 checkpoint independently evaluates tensor-orbit error.

| metric | Stage-C base | E0 conditioned |
|---|---:|---:|
| normalized Stage-D tensor-orbit RMSE | 1.06673 | 1.40389 |
| normalized NN-W1 | 0.24877 | 0.36640 |
| normalized volume-W1 | 0.33600 | 0.31868 |
| distance >= 0.5 Å | 1.00000 | 0.98438 |
| finite positive lattice | 1.00000 | 1.00000 |
| sampling failures | 0 | 0 |

The paired conditioned-minus-base orbit-error bootstrap interval is
`[-0.00394, 0.76972]`; it does not establish improvement. Geometry retention
also fails. E0 therefore remains an offline mechanism result and does not
qualify Stage E or unlock F.

The failure localizes the next work to the generative interface. E0 was trained
on noised clean response structures, while rollout visits generated
composition/assignment/lattice states. Moreover, the current composition law
is sampled before the tensor-conditioned denoiser and the lattice readout has
no direct tensor input. Formal E must first close these conditional side-state
interfaces and train on generated-side exposure; more orbit-mimic weight alone
cannot solve the observed failure.
