# H1a exact-quotient affine-readout audit v2

Status: **qualified in its no-training scope. H1a remains failed.**

A deterministic orthonormal Helmert basis removes the three common-translation
modes analytically before rank, projection and least-squares calculations.  In
this exact quotient, the 225 final coordinate-readout parameters span all
`30/30` physical output directions and project the fixed target with relative
residual `1.12e-15`.

The affine FP64 solution has linear MSE `5.05e-16`; applying it through the
unchanged FP32 production forward gives `5.39e-8`.  Affine-forward relative
error is `2.69e-4`, within the frozen `1e-3` tolerance appropriate to a readout
update more than three orders of magnitude larger than its initialization.  All
parameters are restored bit-exactly after the audit and no tensor path is used.

The result is a positive expressivity finding and a negative scale finding.
The quotient condition number is `3.496e7`, entropy effective rank is `2.23`,
and the minimum-norm update is `2079.20` versus initial readout norm `0.80036`.
The edge basis alone is already full rank, so another Cartesian/Fourier output
branch is not justified.  A capacity-neutral unit scaling of both existing
vector and edge equivariant bases is the next allowed mechanism; it must improve
the no-training spectrum and CUDA behavior before any memorization run.

This audit does not qualify H1a or authorize H1b, tensor conditioning, oracle
work or later Gates.  Exact values are in `result.json`.
