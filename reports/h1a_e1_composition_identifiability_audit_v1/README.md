# E1 composition identifiability audit

The cache/metadata join and 4,096-row formula reconstruction check pass in all
three splits.  The audit shows that a paired exact formula is not determined
by the species-free E1 context.  In the train split, the maximum
anonymous-prototype/matcher ambiguous-sample fraction is `0.9997667`, far
above the frozen `0.01` threshold.  Even the audit-only count-partition and
space-group metadata leave high conditional formula entropy.  None of these
metadata fields is a legal model input.

The split contract is also deliberately extrapolative: validation and test
contain zero exact compositions, anonymous prototypes, count-partition groups
or matcher-envelope groups seen in train.  Exact paired composition recovery
therefore remains a diagnostic of structural inference, not the sole
free-generation objective.  The next state must be a normalized stochastic
composition law evaluated by held-out likelihood, exact conservation and
distributional calibration.

This is a read-only data audit.  It does not qualify a learned composition
model or authorize assignment, L1/M1, tensor/oracle work, relaxation, DFT or
DFPT.
