# S0.4-v1 official small qualification

Date: 2026-07-16  
Decision: **failed_no_advance**  
Failed pre-registered checks: **CUDA latency only**

The official run used the weighted `24×7×24 = 4032` generic Cartesian
production prior. It did not use the frozen 24-frame-only S0.3-v1 method.

The first invocation aborted before producing metrics because CUDA BF16 AMP
mixed BF16 messages with FP32 `index_add_` accumulators and attempted BF16
`det/eigh`. Before the valid run, the implementation was corrected to use FP32
graph accumulation and FP32/FP64 descriptor eigensystems, SO(3) nodes, and
prior masses; learned contractions and MLPs remain autocast. Thresholds were
not changed. A CUDA BF16 regression test now covers this path.

## Passed scientific and numerical checks

- Generic prior: 4,032 raw / 4,032 unique candidates.
- Candidate order permutation, deduplication, and measure-preserving duplicate
  expansion changed posterior and aligned tensor by exactly zero in FP64.
- Maximum CPU FP64 representative errors across eight rotations:
  posterior L1 `5.29e-8`, aligned tensor `3.79e-8`, response relative
  `1.42e-7`, token `1.48e-3`.
- Maximum CUDA FP32 representative errors across eight rotations:
  posterior L1 `1.79e-7`, aligned tensor `6.47e-7`, response relative
  `6.94e-6`, token `8.66e-4`.
- Maximum panel/infinitesimal stratum jumps: `2.61e-2` / `9.70e-6`; all
  audited gradients were finite.
- Axial `K=8,16,32,64` successive normalized differences:
  `3.596e-2`, `5.285e-3`, `3.907e-6` (monotone).
- FP32 versus identical-weight FP64 relative errors:
  aligned `4.68e-7`, response `3.90e-5`, token `3.61e-7`; sorted posterior
  L1 `2.44e-7`.
- BF16 autocast versus identical-weight FP64 relative errors:
  aligned `7.81e-3`, response `1.36e-2`, token `1.05e-2`; sorted posterior
  L1 `6.39e-4`; all outputs finite.
- Peak CUDA memory: `15.42 MB`, below the frozen `64 MB` threshold.

## Failed check

The RTX 4060 Ti atlas latency was `41.89 ms/forward`, exceeding the frozen
`20 ms/forward` threshold. The result is therefore frozen as a Gate failure
even though every scientific-correctness check passed. No tensor-conditioned
training is authorized. A separately versioned performance-only remediation
would be required before requalification; S0.4-v1 itself must not be rerun or
overwritten.
