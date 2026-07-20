# E1.1 failure attribution

## Frozen outcome

`h1a_e1_uniform_count_projection_v1` failed its preregistered E1 thresholds at
seed 5705 and 2,111 updates.  The failure does not authorize L1, M1, joint
generation, tensor conditioning, oracle work, relaxation, DFT, or DFPT.

The uniform D3PM path is numerically valid and self-correcting: all 256 reverse
trajectories are finite, terminal masks are zero, atom count is preserved, and
the final/initial teacher-forced NLL ratio is 0.66925.  Nevertheless,
teacher-forced top-1/top-5 accuracy at `t=0.5` is only 0.41713/0.60587,
top-1 at `t=0.9` is 0.07929, free-reverse site accuracy is 0.06175, and exact
composition is 0/256.

## Composition versus assignment decomposition

The terminal model-predicted counts overlap the target counts on only 0.08144
of atom mass.  Their mean graphwise count L1 error is 18.59375.  Count
projection therefore cannot rescue the reverse sample: projected site
accuracy is 0.06175 and exact assignment is 0/256.

For attribution only, the frozen target counts were supplied to the same
terminal site logits.  This does not enter the denoiser or production sampler.
Under that oracle-count constraint, site accuracy rises to 0.70861 and exact
assignment to 0.30859.  The absolute site-accuracy gain is 0.64685.

This separates the two hypotheses:

- the model has learned a substantial **relative site preference** among the
  correct species once the species multiset is known;
- the existing `mean(site posterior)` construction has not learned the
  **global species set and integer abundance** from the clean geometry,
  lattice, graph size, and noisy element state.

The primary bottleneck is therefore graph-level composition inference, not the
analytic uniform reverse kernel, terminal Hungarian solver, or local
site-assignment ranking.  The next bounded E1 mechanism may add one
permutation-invariant graph composition posterior and feed only its predicted
distribution back to the site head.  It must never consume target formula or
target counts, and it must be tested from scratch under a separately frozen
single-seed budget.
