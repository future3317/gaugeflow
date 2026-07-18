# H1a compact Cartesian Krylov production integration v1

Status: **frozen before implementation and run; no target fitting or training
has occurred.**

The target-free operator qualification passed for an 80-channel Cartesian
carrier. This protocol now requires a single clean production implementation:
the old `coordinate_vector_head` and `coordinate_edge_head` are removed, not
retained as dispatch or checkpoint fallbacks. The existing scalar edge encoder
feeds one learned 32-output projection, split into 16 first-moment and 16 STF
second-moment coefficients. A single 80-channel readout consumes the existing
vector stream and `(m,Qm,Q^2m)` carriers.

All moment reductions and the final coordinate carrier/readout accumulate in
FP32 under BF16 backbone autocast. This is a typed geometry reduction, not an
FP32-only model or a selectable precision fallback. The integration must have
exactly 4,479,161 parameters, no legacy readout keys, no coordinate-target
input, finite FP32/BF16 output and gradients, at least 200 graphs/s on the fixed
RTX 4060 Ti batch, and at most 2.5 GiB peak allocation.

Passing authorizes only a separately frozen single-seed fixed-state
memorization experiment. It does not qualify H1a or permit later Gates.
