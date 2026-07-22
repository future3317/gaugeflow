# LeMat v3 balanced-stream smoke

The qualified v3 index passed a 1,000-global-batch, two-rank stream smoke.
PBE/PBEsol/SCAN sampling fractions were 0.32967/0.33353/0.33680, exact resume
was preserved, and all three functionals passed random-access finite-value
checks. Block-local ordering required a mean 3.081 and maximum 5 parquet row
groups per 64-graph global batch.

This is an engineering closure only. It does not authorize Stage-C continued
pretraining before the Stage-B physical and generation-retention Gate passes.
