# H1a E1 absolute calibration split v1: frozen failure

The v1 builder was run once on the qualified 540,164-graph child-train cache
at commit `fac1fd38dc62daaebfb6a9a0a479ed8d45913bf3`. It stopped before writing an
assignment artifact, as required by the fail-closed protocol.

The source data and primary IID construction were not the cause. The split
contained 486,340 fit, 26,912 calibration, and 26,912 test graphs. Every panel
composition partition retained fit support; partition TV from fit was
`0.00444659` for both panels; all 76 eligible elements passed their floor.

The failed check was the family-wise pair floor. Among 2,499 element pairs with
at least 100 source graphs, calibration contained 10 pairs only twice, while
test contained 5 pairs once and 12 pairs twice. No eligible pair was absent.
Requiring every one of 2,499 independent rare events to occur at least three
times in each 5% panel is not equivalent to checking whether the panel is IID.

Target-aware row swapping was rejected because it would condition the IID
calibration assignment on the chemical target being evaluated. The successor
protocol keeps the same random partition-stratified row assignment, requires
identity presence for every train-eligible pair, and reserves per-pair
calibration for the subset observed at least three times in both panels. The
remaining pairs are evaluated only in preregistered pooled frequency strata.

This failure does not pass E1 and does not authorize site-assignment training,
L1, M1, tensor conditioning, oracle work, relaxation, DFT, or DFPT.
