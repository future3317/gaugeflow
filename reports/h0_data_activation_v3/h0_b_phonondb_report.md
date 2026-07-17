# H0-B PhononDB derivation qualification

## Decision

`H0-B qualified` under
`h0_b_phonondb_derivation_attestation_v4_stratified`. This does not pass H0:
H0-C, H0-D and H0-E remain blocked, so H1 is not authorized.

## Root cause in v1

The frozen v1 compact cache satisfied only the row acoustic sum rule

\[
\sum_j \Phi_{ij}=0,
\]

but did not enforce the Hessian permutation identity

\[
\Phi_{ij}^{ab}=\Phi_{ji}^{ba}.
\]

At \(q=0\), the raw right action on a mass-weighted translation was therefore
small, while the left action was not. Phonopy forms a Hermitian dynamical
matrix, schematically

\[
D_H(0)=\tfrac12\left(D_{\rm raw}(0)+D_{\rm raw}(0)^\dagger\right),
\]

so one-sided ASR did not imply \(D_H(0)T=0\). The worst frozen v1 example,
`mp-12265`, consequently showed a 1.648 THz apparent acoustic frequency,
0.950 minimum translation-subspace singular value and 0.0825 dynamical
residual despite a small row-sum statistic.

## Versioned repair

The v1 cache is unchanged. `phonondb_force_constants_v2` rebuilds the full
supercell Hessian from each source displacement/force YAML, applies phonopy's
level-3 full-Hessian projection onto permutation symmetry and both row/column
ASR, and only then compresses to primitive-by-supercell storage.

Across all 10,034 materials:

- source reproduction error is at most \(3.17\times10^{-13}\);
- projected row and column ASR residuals are at most
  \(9.66\times10^{-13}\);
- projected permutation residual is exactly zero in stored float64 arithmetic;
- median projection relative L2 is \(1.43\times10^{-4}\), and the 99th
  percentile is 0.0163;
- 43 long-tail records exceed 0.05 relative L2 and remain explicitly marked in
  the index for later H2 confidence handling.

The source dielectric is retained as audit evidence. Production NAC uses the
explicit symmetric part \((\epsilon+\epsilon^T)/2\). The preliminary signed
frequency comparison at Gamma failed because square-root frequencies amplify
machine-scale acoustic eigenvalue jitter. The corrected equivalence audit uses
the dynamical matrix and nonzero-q frequencies: maximum differences are
\(1.78\times10^{-15}\) and \(1.86\times10^{-13}\) THz, respectively. The
Gamma jitter, \(1.21\times10^{-6}\) THz, remains reported but is not
misinterpreted as a NAC change.

## Bounded numerical audit

The expensive eigendecomposition audit is deterministic and stratified rather
than a second full-data sweep. It contains 1,024 materials:

- all 654 v1 acoustic failures;
- all 43 v2 projection-relative-L2 records above 0.05;
- the 64 largest raw dielectric-asymmetry records;
- deterministic hash selections balanced by NAC availability and primitive
  atom-count bins, bringing the mandatory-union size of 737 to 1,024.

The sample contains 505 NAC-available and 519 explicitly NAC-unavailable
records, 55,335 degenerate mode clusters, and 256 deterministic q/-q pairs.
There were zero load or numerical failures. Observed maxima/minima were:

| Metric | Observed | Frozen limit |
| --- | ---: | ---: |
| Gamma translation frequency | 0.001202 THz | 0.05 THz |
| translation subspace minimum singular value | 0.9999999999999867 | 0.995 |
| translation dynamical residual | 3.07e-8 | 1e-3 |
| degenerate-projector gauge error | 9.99e-16 | 1e-10 |
| q/-q frequency error | 1.37e-13 THz | 1e-8 THz |
| q/-q projector error | 5.84e-12 | 1e-8 |
| Born-charge neutrality | 1.20e-7 | 1e-5 |
| processed dielectric symmetry error | 0 | 1e-8 |
| NAC nonzero-q frequency shift | 1.86e-13 THz | 1e-8 THz |
| NAC dynamical-matrix shift | 1.78e-15 | 1e-12 |

Large external artifacts remain under
`E:/DATA/T2C-Flow/processed/phonondb_force_constants_v2` and
`E:/DATA/T2C-Flow/processed/gaugeflow_h0_v3`. The active repository stores the
builder, auditor, frozen config, tests and this compact report.

The numerical audit was run in the pinned WSL environment with one BLAS thread
per worker to avoid nested-process oversubscription:

```bash
export PYTHONPATH="$PWD:$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m scripts.audit_phonondb_h0_b \
  --data-root /mnt/e/DATA/T2C-Flow \
  --output-root /mnt/e/DATA/T2C-Flow/processed/gaugeflow_h0_v3 \
  --workers 14 --sample-size 1024 --conjugacy-sample-size 256
```
