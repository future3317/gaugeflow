# H0-E-v3 K0 maximal-k parent occurrence

## Decision

`H0-E-v3-K0_failed_stop_before_H1a`.

The separately frozen cell-changing successor evaluated every index-2--4
maximal klassengleiche E0 embedding on the exact ordered 64-material E1a
zero-candidate panel. It found `0/64` new candidate materials versus the
preregistered minimum `3/64`. H0-E-v3, H0, H1a/H1b and H2--H6 therefore remain
stopped. This result does not alter H0-E-v1 or H0-E-v2 E1a history.

## Frozen results

| Metric | Result | Requirement | Pass |
|---|---:|---:|:---:|
| Selected rows / Alex joins | 64 / 64 | 64 / 64 | yes |
| Candidate edges evaluated | 578 | 578 | yes |
| Processing failures / nonfinite values | 0 / 0 | 0 / 0 | yes |
| New candidate materials | 0 | >= 3 | **no** |
| New candidate-material fraction | 0.0 | >= 0.046875 | **no** |
| Forward/reverse embedding-set agreement | 1.0 | 1.0 | yes |
| Path-quarantine application | 1.0 | 1.0 | yes |
| Total / p95 per-row runtime | 59.05 s / 3.682 s | diagnostic | -- |

Candidate-conditional strict-parent, StructureMatcher, site-count and full-
action-order fractions are fail-closed at zero because no occurrence survived.
The independent reverse-order rebuild reproduced all 64 rows, all 578 edges,
zero candidates and every stored decision check.

## Mechanism localization

The K0 implementation constructs the exact finite action
`G x (Z^3 / B Z^3)` in the observed child supercell. Positive controls recover
an index-2 P1 parent, an index-2 inversion-symmetric parent, and an off-diagonal
non-HNF integral embedding; the complete affine action closes and is
enumeration-order invariant.

A post-result read-only stage decomposition of all 578 real-material edges
localized the negative result before any tolerance or matcher stage:

- 265 edges have a primitive child site count not divisible by the declared
  cell index, so their ordered composition cannot be a repetition of a smaller
  primitive parent composition;
- 108 edges cannot define a species-preserving nearest assignment satisfying
  the complete parent-supercell group law;
- 205 edges define a complete assignment, but every one fails the frozen
  triangle-safe `0.4 Angstrom` orbit-defect prefilter. Their minimum, median,
  p95 and maximum defects are `1.22584`, `2.44933`, `4.27364` and
  `5.89248 Angstrom`.

No edge reached one-sided Reynolds projection, strict parent reidentification
or StructureMatcher. The nearest real edge is still more than three times the
frozen orbit-defect bound. The failure is therefore not attributable to a
near-threshold numerical choice, the quarantined SG12-to-SG71 path, or a
matcher-only implementation gap. This panel does not contain commensurate
short-period translation-restoration parents within the registered distortion
domain.

## Why multi-step t was not run

For nested parent groups `H subset M subset G`, the fixed spaces obey
`Fix(G) subset Fix(M)`, hence
`distance(x, Fix(G)) >= distance(x, Fix(M))`. A multi-step t chain cannot lower
the first maximal-t projection distance and cannot reach the 17 E1a materials
with no incoming t edge. K0 therefore tested the only new bounded E0 mechanism:
cell-changing maximal-k restoration. Its failure stops this H0-E-v3 version.

## Provenance

- Protocol freeze commit: `9ed9301762849385d55041f04b827aede3be3416`
- Exact maximal-k implementation commit: `27d238b0fa5dbf62a2bcca0188e8ccca4e3457f6`
- Standalone-runner implementation commit used by the run:
  `8cc2dd2a392753d949a64580ddec20a15ca901ac`
- Results SHA-256:
  `d8060044d244401dd10c18c675cfc0f82ce55322dcf147166e9842934739eb4d`
- Manifest SHA-256:
  `8aa795cc6fa05ab08d48a0b65005390ba2eb17daf65ae8eb89db6b8d36bf63b0`
- Independent-audit SHA-256:
  `11f8fe4ce8a7d00c52911ad8739306fd1078d10fd9ff2f2ee13b7c7034bae815`
- External artifact root:
  `E:/DATA/T2C-Flow/processed/gaugeflow_h0_v6/maximal_k_parent_occurrence_k0_v1/`
