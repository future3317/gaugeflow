# GaugeFlow vNext legacy evidence freeze

The vNext execution contract is locked to SHA-256
`3bdac52ba00a14c40e8bb6f9de732d16d8a91eb5f81e1a9a2e9b2334e8dd952b`.
The deterministic manifest freezes 316 configuration, report, artifact,
runner, and regression-test files from historical Gate A--A11 and P5--P5-C0.
Every file has a byte count and SHA-256 digest under
`artifacts/vnext_legacy_frozen_v1/manifest.json`.

The freeze does not change any historical threshold, checkpoint, metric, or
conclusion. Gate A--A11 remain negative or diagnostic-only; P5 and P5-C0 remain
not passed. Real tensor training, oracle use, relaxation, DFT, and DFPT remain
unauthorized.

The inventory also records a blocking Q0 input defect: P5-C0 and D0.4--D0.8
did not save a model checkpoint. The frozen sources, time grid, couplings, and
metrics exist, but learned-vector-field Jacobians and checkpoint rollout cannot
be reconstructed from them. The Q0 protocol forbids retraining, so the future
Q0 runner must report `blocked` unless the original weights are independently
recovered without modifying historical evidence.
