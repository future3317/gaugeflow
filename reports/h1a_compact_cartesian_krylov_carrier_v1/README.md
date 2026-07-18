# H1a compact Cartesian Krylov carrier v1

Status: **qualified in its target-free operator scope; production remains
unchanged and H1a remains failed.**

Post-hoc whitening made the existing combined readout algebraically orthogonal
but left its effective raw norm and BF16 feature sensitivity unchanged. This
protocol moves the repair before the final readout. It replaces the diagnostic
edge basis by a compact Cartesian moment carrier without modifying production.

For bounded scalar edge channels `a_ec,b_ec`, unit edge direction `d_e`, and a
smooth cutoff `w_e`, define

```text
m_ic = sum_(e -> i) w_e a_ec d_e / sqrt(deg_i),
Q_ic = sum_(e -> i) w_e b_ec (d_e d_e^T - I/3) / sqrt(deg_i).
```

After invariant graphwise RMS normalization, the carrier is

```text
[v_i, m_i, Q_i m_i, Q_i^2 m_i].
```

Here `v` is the existing vector stream and there are 16 moment channels, giving
80 Cartesian vector channels rather than the previous 225 affine edge/vector
columns. Cayley--Hamilton makes powers above `Q^2 m` redundant in three
dimensions. Under every `R in O(3)`, `m -> m R`, `Q -> R^T Q R` in row-vector
notation, and all three Krylov vectors transform as polar vectors. The
construction is translation invariant, node-permutation equivariant, smooth at
zero, frame free, harmonic free, and fully vectorized over edges and channels.

The fixed random orthonormal scalar projection is only an operator probe; it
uses model features and no coordinate target. This audit checks quotient rank,
conditioning, `O(3)` covariance, translation horizontality, FP32/BF16 carrier
and probe-gradient agreement, latency, and memory on the same 16 states as the
preceding diagnostics. It performs zero optimizer steps and cannot change H1a
or authorize a later Gate.

Mathematically the construction follows Cartesian moment-tensor and atomic
cluster expansions, while using Cayley--Hamilton closure to keep the runtime
finite: Shapeev, *Multiscale Model. Simul.* 14 (2016); Drautz, *Phys. Rev. B*
99 (2019).

## Result

All 16 fixed real states attain their complete translation-quotient rank. Node
counts range from 4 to 14, expected ranks from 9 to 39, and the maximum
quotient condition number is `14657.96`, below the frozen `100000` bound. The
improper-inclusive `O(3)` covariance error is `6.76e-6` and the graphwise mean
error is `1.54e-7`.

The carrier is stable under the actual mixed-precision backbone path. BF16
relative RMS error is `0.08966`, cosine with FP32 is `0.99598`, and the
target-free probe-gradient norms are `9.448/9.459`; their norm ratio is
`1.00121` and cosine is `0.99269`. Every forward and backward value is finite.
On the 16-graph, 12,192-edge CUDA panel, the fully vectorized 80-channel
carrier costs `3.043 ms` and `11.609 MiB` incremental peak memory.

The audit reads zero coordinate targets and executes zero optimizer steps. It
qualifies only a separately frozen production-integration check. It does not
qualify H1a, authorize target fitting, or open any later Gate.
