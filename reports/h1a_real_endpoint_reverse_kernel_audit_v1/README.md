# H1a real-endpoint reverse-kernel audit v1

Status: **completed; all methods close generic real endpoints, so no kernel is
selected**.

The generic analytic panel closed exactly under every candidate and therefore
could not reproduce the branch failures found in special four-site structures.
This version keeps the probability path, exact quotient score, methods,
quadrature and step grid unchanged, but replaces the hand-written endpoints by
a deterministic, hashed panel of 64 six-site validation structures from the
qualified H1a cache.  Each endpoint receives 16 common-random-number
trajectories from both its exact terminal heat kernel and the production
uniform quotient prior.

The audit reports endpoint RMS in each structure's Cartesian metric, exact
recovery, final-step translation-posterior margins, and type-preserving
self-automorphisms.  The latter are diagnostic strata only: the production
coordinate target retains site correspondence and does not silently quotient
permutations.  Thresholds and endpoint selection were frozen before reading
any trajectory result.

The first launch stopped before trajectory construction because the cache has
no four-site validation rows.  A read-only node-count census found 18,296
six-site rows; the frozen-not-run protocol was corrected to six sites without
changing endpoint count, total trajectories, methods, thresholds, or seed.

Passing can qualify only one score-only kernel for existing-checkpoint sampling.
It cannot qualify H1a or authorize training or any later Gate.

## Result

The deterministic selection is recorded in `results.json` by cache hash,
validation row indices, material IDs and a selection hash.  None of the 64
six-site endpoints has a nonidentity type-preserving pure-translation
self-automorphism at `1e-5 A`.

For both exact-forward and uniform-quotient initialization, the endpoint-aware
reference and all four score-only methods achieved `1.0` recovery at 200 steps,
zero cut-locus failures, and mean endpoint errors below `8e-15 A`.  Refining
from 100 to 200 steps did not increase mean RMS materially.  This confirms
that the probability path, quotient score, and each integrator close ordinary
real endpoints; it still does not identify which finite kernel is robust to a
learned score.

The archived `0.433/0.612` failures must therefore be reproduced on a
non-generic symmetry panel or with the learned checkpoint before they can be
attributed to the reverse kernel.  No production code or checkpoint is changed
by this result.
