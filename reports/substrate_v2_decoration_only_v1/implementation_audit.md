# Post-run implementation audit: substrate-v2 decoration-only v1

The executed v1 artifact remains preserved at this path, including its original
CSV and manifest.  It is **invalid for promotion**, not evidence for either a
pass or a scientific failure.

Two defects were found after the run:

1. Its node-relabeling diagnostic fed `scores[permutation]` to the exact
   categorical-law checker.  That tests only bookkeeping after a score matrix
   has already been permuted; it does not test whether a separately relabelled
   geometry input produces the correspondingly permuted score law.
2. The exact finite-assignment energy was reduced in FP32.  Once the quotient
   loss saturated, reassociating four large site scores produced measurable
   log-probability changes despite identical mathematical assignments.  This
   is a numerical artifact rather than a valid equivariance measurement.

`configs/substrate_v2_decoration_only_v2.json` retains the entire v1 science
contract (panel, seeds, capacity, optimizer, 1,200 updates, thresholds and
ablations), but uses float64 exact enumeration and a second model forward pass
on the pre-registered `[2, 0, 3, 1]` node relabeling.  The v2 result must be
used for any promotion decision.
