# H1a tensor-free benchmark v2

**Decision: failed; H1b remains closed.**

Version 1 compared a sampler that deliberately draws the empirical training
node-count prior against a formula/prototype-disjoint test node-count
distribution. The real train-to-test node-count JSD is 0.26641, already far
above the v1 bound, so that check was internally inconsistent. The v1 protocol,
result, thresholds, and failed decision remain unchanged.

Version 2 uses 8,192 fixed training structures for every unconditional target
distribution: node count, element marginal, volume per atom, and nearest
periodic distance. A separate 8,192-structure test subset is used only for
formula novelty/leakage diagnosis. It evaluates 256 generated samples from
each of the three qualified 20,000-step checkpoints with the unchanged
100-step uniform-variance reverse sampler. No metric threshold was selected
after observing v2 samples.

## Result

| Metric | Result | Frozen bound | Check |
|---|---:|---:|:---:|
| sampling failures | 0 | 0 | pass |
| terminal masks | 0 | 0 | pass |
| finite positive lattices | 1.00000 | 1.00000 | pass |
| minimum distance >= 0.5 A | 0.98307 | >= 0.98000 | pass |
| element marginal JSD | 0.00427 | <= 0.10000 | pass |
| node-count JSD | 0.00302 | <= 0.01000 | pass |
| normalized volume Wasserstein | 0.03187 | <= 0.50000 | pass |
| volume reference envelope | 0.99609 | >= 0.98000 | pass |
| formula uniqueness | 1.00000 | >= 0.50000 | pass |
| normalized nearest-distance Wasserstein | 1.97287 | <= 0.75000 | **fail** |

The corrected node-count result confirms that the v1 node-count failure came
from its inconsistent held-out reference. The remaining failure is physical
and coordinate-specific. Generated nearest-distance quantiles at
0/5/50/95/100% are `0.0915 / 0.7184 / 1.6031 / 2.3506 / 3.2363 A`; the fixed
training reference gives `0.7479 / 1.8125 / 2.6982 / 3.4364 / 4.7117 A`.
Longer, stratified training improved the generated median from the v1 value of
1.2711 A but did not close the packing-distribution gap.

Element, node-count, and volume distributions now pass by wide margins, so the
next action is restricted to a coordinate causal audit. It must distinguish a
teacher-forced low-noise score failure from rollout-state distribution shift
before any new loss or architecture is introduced. H1b, tensor conditioning,
oracle work, relaxation, DFT, and DFPT remain prohibited.
