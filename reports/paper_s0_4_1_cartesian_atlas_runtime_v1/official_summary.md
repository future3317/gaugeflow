# S0.4.1 official runtime qualification

Date: 2026-07-16  
Decision: **passed_runtime_qualification**

This is a separately versioned performance-only successor. It does not edit or
reinterpret the frozen S0.4-v1 `failed_no_advance` result.

The production generic measure remains the weighted `24×7×24 = 4032`
Cartesian prior. The optimization caches the state-independent base cubature
and skips numerical deduplication only when the active chart pair proves that
all candidates are unique under a bijective two-sided rotation. Axial and
mixed-stratum paths retain full multiplicity-corrected deduplication.

Official RTX 4060 Ti results:

- atlas latency: `14.6163 ms/forward` (threshold `≤20 ms`);
- peak CUDA memory: `15.1865 MB` (threshold `≤64 MB`);
- generic candidates: `4032 raw / 4032 unique`;
- aligned tensor error against the original deduplicated measure: `2.17e-15`;
- sorted posterior L1 error: `4.80e-16`;
- prior L1 error: `0`;
- predecessor S0.4 report tree unchanged: yes;
- full tests, ruff, and mypy: passed.

This closes only the Cartesian-atlas runtime qualification. It authorizes
preparation of a separately versioned S1a production hybrid trainer/reverse
sampler qualification. It does not mean S1a has started and does not authorize
the S2 blueprint, tensor training, oracle, relaxation, DFT, or DFPT.
