# Stage-D response data and scale audit

This audit prepares the independent Cartesian response evaluator; it does not
train or qualify tensor-conditioned generation.

## Immutable support

- Cache SHA-256: `4f780dba78b422e7b6f3e0db338cf769c968b9865f7096f5d5add0227f737e1c`
- Graphs: `3,946`; atoms: `43,015`
- Formula/prototype-disjoint split: `3,173 / 398 / 375`
- Piezoelectric labels: `3,946`
- Dielectric, Born-charge and Gamma labels: `3,943`
- Strict internal-strain labels: `1,266` graphs
- Audited JARVIS elastic labels: `2,893` graphs, converted exactly once from GPa
  engineering Voigt through an orthonormal Kelvin basis to Cartesian
  `C_ijkl`. All non-elastic tensors are bitwise unchanged by the merge.

Reduced-composition overlap is zero across all split pairs. IDs, formulae,
prototype labels and split metadata are construction-only fields and do not
enter model batches.

## Heavy-tail finding

The train dielectric total RMS has median `7.61`, p99 about `1,107`, and maximum
about `112,401`; its anisotropic RMS maximum is about `91,636`. Ordinary RMS
normalization would therefore compress the typical response by hundreds of
times. These records can be physically meaningful soft-mode responses and are
not silently deleted.

The fitted transform uses train-only, source-local scalar statistics, a median
isotropic identity location for rank-two tensors, and the invertible radial
chart

\[
  \mathcal T(X)=\frac{\operatorname{asinh}(\|X\|_{\rm RMS})}
  {\|X\|_{\rm RMS}}X.
\]

It is O(3)-equivariant, preserves physical zero and has an analytic radial-sinh
inverse. The elastic robust scale is `23.50885 GPa`; the final normalizer
SHA-256 is
`27112e0c3f32911903ce9740942bbe50a10c3634275ab4c986946f1d3af87d35`.

## Execution closure

A three-step BF16 CUDA smoke from the selected Stage-C 30k EMA completed with
finite loss and gradients, `3.35 GiB` peak allocated memory, validation
dielectric/elastic losses `1.07464/0.71929`, and total loss `0.85943`. Before the
radial chart the corresponding diagnostic was dominated by dielectric losses
of roughly `682` per batch and `1,056` on validation. The smoke proves numerical
execution only; D0 performs the first paired mechanism selection.
