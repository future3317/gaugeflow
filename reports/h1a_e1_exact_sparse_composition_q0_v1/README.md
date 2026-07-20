# Exact sparse composition Q0

Q0 qualifies the exact law over increasing element-token/positive-count pairs.
For vocabulary size `K`, node count `N` and support size `S`, every state is
represented once and only once, with `sum(n_r)=N`.  Exhaustive FP64 tests over
`K=5`, `N<=6`, `S<=3` attain maximum normalization error
`4.44e-16`.  All enumerated states are duplicate-free; 50,000 stochastic draws
visit all 65 states at `N=4`, have total variation `0.01263` from the exact law,
and produce no invalid state.  Formula reconstruction, finite gradients,
FP32/FP64 and BF16/FP32 checks pass.

The first implementation missed only the frozen CUDA teacher-forced latency
bound (`13.57 ms > 10 ms` for 256 graphs).  Its result is retained as
`pre_vectorization_latency_failure.json`.  The probability law and thresholds
were not changed.  Teacher-forced prefixes are known, so the seven serial
`GRUCell` launches were replaced by one mathematically equivalent batched GRU;
sampling remains autoregressive.  Final latency is `4.01 ms`, sampling latency
is `14.05 ms`, and incremental peak memory is `13.72 MiB` on the RTX 4060 Ti.

All frozen Q0 checks now pass.  This authorizes one separately frozen,
single-seed composition-only training screen.  It does not qualify learned
composition, site assignment, L1/M1, tensor/oracle work, relaxation, DFT or
DFPT.
