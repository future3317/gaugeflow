# Stage-E1a lattice-only NFE smoke

This is a small, non-qualification diagnostic on 32 clean-panel validation
structures using the Stage-C step-30,523 checkpoint.  The same initialized
lattice states were reused across 25/50/100/200 reverse steps for each batch;
Brownian increments were not nested, so the numbers are directional rather
than a solver Gate.  Both reverse-SDE and probability-flow modes were tested.

The result is non-monotone.  Reverse-SDE volume tails appear in different
batches at different NFE values (one 16-structure batch reaches a maximum
volume-per-atom of 72,582 and normalized W1 273.88 at 100 steps), while the
other batch remains below 65 and W1 0.40.  Probability-flow becomes steadily
worse in the tested batch: normalized volume W1 is 0.59, 1.88, 4.46 and 7.24
at 25/50/100/200 steps.  All outputs are finite and positive in this smoke.

This does not support replacing the reverse sampler, increasing NFE as a
repair, or clipping terminal volumes.  It points to a state-dependent
generated-lattice field/exposure problem (with solver variance still requiring
a properly nested-Brownian audit).  The next implementation experiment should
train or expose the lattice field on the generated side while preserving the
P1 chart and exact composition context, then re-run the full factorial panel.
