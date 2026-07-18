# H1a coordinate branch minimality v1

Status: **failed before training; neither single branch is sufficient. H1a
remains failed.**

The preceding FP32/BF16 audit measured `32.31x` cancellation between the final
vector and edge coordinate fields. This protocol tests the simplest causal
repair: remove a redundant branch rather than add another basis, loss, or
fallback. It uses the same first 16 fixed states, model initialization, noise,
path and exact graph-equal affine solve.

Vector-only, edge-only and combined bases are evaluated from the same captured
production features. A single branch may proceed only if it retains full
one-state quotient rank, target projection, 16-state FP32 fit, low-time endpoint
accuracy, moderate solution norm, and stable BF16 prediction and backbone
gradient. If both single branches pass, the smaller vector branch is selected;
if neither passes, production remains unchanged.

An explicit Helmert basis removes exactly three translation modes. On the first
11-site state, vector-only, edge-only and combined designs all have exact
quotient rank `30/30` and target projection residual below `1.8e-13`. Thus both
single branches contain all physical directions locally; the decision is about
cross-state representation and numerical scale, not missing equivariance.

Vector-only is relatively BF16-stable (`0.9988x` FP32 MSE and prediction
relative RMS `0.0981`) but cannot fit the panel: FP32 MSE is `0.56437`, low-time
endpoint RMS is `0.05046 A`, solution norm is `1022.67`, and BF16/FP32 gradient
cosine is only `0.6147`. Edge-only fits more closely (`0.13474` MSE and
`0.02401 A`) but still misses the frozen `0.12` bound and requires norm
`1325.83`; BF16 MSE is `10.2160` (`75.82x` FP32), gradient norm is `16794.1`
versus `4.295`, and gradient cosine is `-0.1419`. The combined branch retains
the best FP32 MSE (`0.09947`) but reproduces the known BF16 instability.

Neither single branch qualifies, so no branch is deleted and no optimizer step
or production mutation occurs. The result cannot be rescued by changing
precision, thresholds, states, seeds or training budget. A successor must
qualify a target-free compact equivariant orthogonal-residual basis that keeps
the combined cross-state span without cancellation.

Implementation note: the first debug execution used numerical SVD rank after
FP32 mean subtraction and counted three translation roundoff modes. Commit
`7c9cacb` replaced that diagnostic with the preregistered exact Helmert
quotient; no threshold, state, seed or model setting changed, and only the
corrected output in `result.json` is scientific evidence. The frozen runner,
protocol and tests are recoverable from commits `586736b` and `7c9cacb`.
