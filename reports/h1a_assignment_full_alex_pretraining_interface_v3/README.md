# Full-Alex assignment pretraining interface v3

Status: **PASS**. This Gate performed no model training.

The active interface excludes every material in the 454-carrier Gold
catalogue, leaving 539,983 of the 540,164 Alex-MP train structures. Model
batches contain only clean species-free fractional geometry, lattice, exact
unordered composition counts and the masked occupation target. Audit IDs and
roles never enter `PackedAlexModelBatch`.

The exact dual-lattice periodic solver scanned all 55,358,736 directed pairs
with zero fail-closed event. Its largest finite integer box contained 450
candidates. On an independent 2,048-structure panel, the maximum difference
from the float64 sphere decoder was `1.5249e-6`. Refinement was used by 49,692
graphs (1,269,844 pairs) and remains a report-only execution diagnostic.

On an NVIDIA GeForce RTX 4090, the 8,192-structure feature panel reached
`5469.78 graphs/s`, used `172.60 MiB` peak allocated CUDA memory, and produced
zero nonfinite compilation. All frozen correctness, leakage and resource
checks passed.

This PASS authorizes freezing and testing one full-Alex masked-occupation
representation pretrainer. It does not qualify parent-conditioned assignment,
generated composition, `p(N)`, lattice, coordinates, joint generation, tensor
conditioning, relaxation, DFT or DFPT.
