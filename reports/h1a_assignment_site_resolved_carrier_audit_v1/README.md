# H1a assignment site-resolved carrier audit v1

Decision: `build a versioned assignment-carrier derivative from the certified parent candidate and supercell transform; do not train or infer missing expanded geometry from target coloring`.

| metric | value |
|---|---:|
| carriers | 454 |
| full site geometry | 0.348018 |
| explicit expanded geometry | 0.000000 |
| explicit expanded lattice | 0.000000 |
| HNF on nontrivial supercells | 0.000000 |
| translation cosets on nontrivial supercells | 0.000000 |
| action-node alignment | 1.000000 |

The archived O1 carrier stores primitive parent coordinates but omits the
expanded species-free geometry, HNF and translation-coset ordering needed
by 296 nontrivial-supercell carriers. These fields existed in the certified
parent decomposition object but were not serialized into the assignment
interface. They must be rebuilt at a versioned data boundary, not guessed
from target coloring or patched with a model fallback.

Boundary: This audit cannot qualify assignment, generated-C, p(N), L1/M1, tensor work, relaxation, DFT or DFPT.
