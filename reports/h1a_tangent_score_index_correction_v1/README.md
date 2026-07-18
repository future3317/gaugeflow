# H1a tangent-score index correction v1

The zero-training CUDA qualification failed, so no coordinate pretraining was
started.  The Cartesian-tangent chart itself qualified: FP32/BF16 prediction
and target round trips were at most `2.38e-7`, GL(3,Z) and O(3) chart errors
were at most `1.19e-7`, all gradients were finite, the parameter/carrier
contracts held, and tensor-free execution constructed zero atlas candidates.

The sole failed check was the frozen `1e-4` full-model translation-consistency
limit.  BF16 fractional output relative RMSE was `1.4569e-4` (Cartesian
`1.0433e-4`); FP32 remained `1.5529e-5`.  A read-only edge audit found the same
`12,192` physical edges before and after translation and no cutoff switch, but
the current separately-wrapped endpoint subtraction changed reconstructed
displacements by as much as `1.43e-6 A`.  This identifies a periodic-lift
arithmetic issue rather than a tangent-index, capacity, data, or optimizer
failure.  The result remains failed and cannot authorize training.  A future
protocol may test only the algebraically equivalent direct-difference plus
integer-lift reconstruction; it may not change this result or its thresholds.
