# H1a middle-noise reciprocal attribution v1

Decision: **do_not_implement_reciprocal_carrier**.

This is a frozen-checkpoint diagnostic. It does not retrain the generator, change H1a, or authorize any later Gate.

## Three independent checks

- middle-noise endpoint recoverability: `False` (mean top-1 `0.403150`);
- low-k residual spectral excess: `False` (mean low/high ratio `1.053482`);
- frozen low-k probe generalization: `False` (low `0.002257`, high control `-0.000682`).

A reciprocal production carrier is permitted only when all three checks pass. The detailed curves are stored in the adjacent CSV files.

The retrieval accuracy falls from `0.535433` at `t=.35` to `0.259843` at
`t=.65`; the five low/high residual ratios are `1.12653`, `1.11654`,
`1.12259`, `1.00626`, and `0.89549`. Low-band graph coverage is
`0.9766--0.9883`, so the negative result is not an empty-band artifact. The
low-minus-high held-out probe improvement is `0.002939 < 0.03`.

`h1a_reciprocal_attribution.pdf` and `.png` are generated from the CSV files by
`scripts/plot_h1a_midnoise_reciprocal_attribution.py`. The independent audit in
`independent_audit.json` recomputes all checks, metrics, decisions, and hashes.
No reciprocal carrier was implemented and this result does not authorize any
later Gate.

The independent Bridge audit is synthesized in
`bridge_no_go_synthesis.md`. It reports a middle-noise held-out low-frequency
explained fraction of `-0.001368`, only `0.000695` over random Fourier and
`-0.001368` relative to a graph token. The two audits jointly establish the
reciprocal NO-GO; neither should be rerun or tuned.
