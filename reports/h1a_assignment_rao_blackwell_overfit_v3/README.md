# Rao--Blackwellized assignment overfit v3

Status: **passed** on the frozen single-seed software Gate.

The model, optimizer, eight carriers and 2,000-step budget are unchanged from
the failed pair-context screen.  The training estimator is the only material
change: eight independent prefix paths are packed per carrier, and every
prefix analytically averages the log probability over all legal next sites.
This is an unbiased Rao--Blackwellization of the uniform reveal-order lower
bound, not an exact order-marginal likelihood.

The run at commit `a2d2a17` on one RTX 4090 achieved:

- target quotient-probability lower bound: `0.93833`;
- relative NLL reduction from count-uniform: `0.96720`;
- sampled target-orbit retrieval: `0.93750`;
- sampled orbit-aligned site accuracy: `0.96875`;
- exact composition and finite-gradient fractions: `1.0`;
- elapsed time: `68.80 s`; peak allocated CUDA memory: `813.17 MiB`.

All preregistered checks passed.  This result qualifies the representation and
training estimator for one single-seed IID assignment Gate.  It does not
qualify OOD assignment, generated composition, atom-count, lattice, joint
generation, tensor conditioning, relaxation, DFT or DFPT.
