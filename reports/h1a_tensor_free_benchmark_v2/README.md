# H1a tensor-free benchmark v2

**Status: frozen before execution.**

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
