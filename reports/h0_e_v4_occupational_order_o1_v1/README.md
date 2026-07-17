# H0-E-v4 O1-v1 held-out occupational census

## Decision

`H0-E-v4-and-H0_qualified_only_separately_frozen_H1a_may_start`.

The complete 835-material held-out remainder produced 224 distinct candidate
materials and 454 unique canonical material--embedding paths. Together with
the 125 already-qualified v1 materials and 10 distinct O0-v2 materials, the
clean-universe coverage is `359/1023 = 0.350929`, above the unchanged frozen
minimum of `0.15`. The independent reverse-order auditor rebuilt every row,
edge, candidate array and scalar with no failure.

This qualifies H0-E-v4 and, together with frozen H0-A through H0-D-v2,
qualifies H0. It authorizes only the freezing of a separate real-data H1a
protocol. It does not start H1a, H1b, H2--H6, tensor conditioning, oracle work,
relaxation, DFT or DFPT, and it does not rewrite any failed historical Gate.

## Frozen result

| Metric | Result | Requirement | Pass |
|---|---:|---:|:---:|
| Held-out rows | 835 | 835 | yes |
| Clean disjoint partition | 125 + 63 + 835 = 1,023 | exact | yes |
| O0 overlap | 0 | 0 | yes |
| Candidate / eligible edges | 13,370 / 13,370 | exact | yes |
| Processing failures / nonfinite values | 0 / 0 | 0 / 0 | yes |
| New O1 candidate materials | 224 | >= 19 | yes |
| New O1 candidate fraction | 0.268263 | >= 0.022754 | yes |
| Aggregate qualified materials | 359 | >= 154 | yes |
| Aggregate clean-universe coverage | 0.350929 | >= 0.15 | yes |
| Canonical material paths / occurrences | 454 / 454 | unique fraction 1.0 | yes |
| Forward/reverse embedding-set agreement | 1.0 | 1.0 | yes |
| Exact coloring reconstruction | 1.0 | 1.0 | yes |
| Occupational stabilizer subgroup certificate | 1.0 | 1.0 | yes |
| Stabilizer order equals child operation order | 1.0 | 1.0 | yes |
| Strict geometry-parent reidentification | 1.0 | 1.0 | yes |
| Terminal integer-element fraction | 1.0 | 1.0 | yes |
| Partial occupancies | 0 | 0 | yes |
| Distinct aggregate parent / child groups | 59 / 55 | >= 10 / >= 20 | yes |
| Maximum projected group error | 6.954e-14 A | <= 1e-7 A | yes |
| Maximum source displacement | 0.196925 A | <= 0.2 A | yes |
| Maximum source Hencky norm | 0.115615 | <= 0.15 | yes |
| Wall time, four single-thread workers | 1,162.90 s | diagnostic | -- |

The path statistic is deliberately separate from material coverage. Multiple
source-certified affine embeddings for one material contribute distinct
canonical paths but never increase the material numerator. The 454 paths are
all unique `(material_id, E0_embedding_key)` pairs and represent 3.3957% of the
13,370 evaluated material--embedding edges.

## Held-out coverage

O1 is not a second mechanism-locating sample. Starting from the immutable v1
1,024-row selection, the versioned data cleaning removes one material, the 125
v1-positive rows form one partition, and all 63 clean O0 rows form another.
Every remaining zero-candidate, nonfailure row is in O1. Selection is therefore
a full census and does not use any O0 parent group, transition or outcome.

| Stratum | Held-out rows | Candidate materials | Fraction |
|---|---:|---:|---:|
| train | 672 | 181 | 0.2693 |
| validation | 83 | 27 | 0.3253 |
| test | 80 | 16 | 0.2000 |
| triclinic | 67 | 7 | 0.1045 |
| monoclinic | 97 | 24 | 0.2474 |
| orthorhombic | 129 | 32 | 0.2481 |
| tetragonal | 135 | 51 | 0.3778 |
| trigonal | 132 | 34 | 0.2576 |
| hexagonal | 140 | 29 | 0.2071 |
| cubic | 135 | 47 | 0.3481 |
| <=4 primitive sites | 182 | 29 | 0.1593 |
| 5--8 primitive sites | 207 | 66 | 0.3188 |
| 9--16 primitive sites | 214 | 75 | 0.3505 |
| 17--32 primitive sites | 232 | 54 | 0.2328 |

The 454 occurrences contain 158 maximal-t paths and 296 maximal-k paths. Cell
indices are 158 at index 1, 230 at index 2, 22 at index 3 and 44 at index 4.
This demonstrates that occupational symmetry breaking is not an axial edge
case of the O0 panel and that both primitive-cell and cell-changing catalogue
branches are used on held-out structures.

## Independent audit

The auditor joined raw Alex rows in `test -> val -> train` order, traversed the
E0 catalogue and the selected materials in reverse order, and recomputed the
strict geometry projection plus exact integer-coloring stabilizer. It rebuilt:

- 835/835 rows and 13,370/13,370 candidate edges;
- 224 candidate and occupationally nontrivial materials;
- 454/454 occurrence sets and all stored numeric arrays/scalars;
- the aggregate material count and final Gate decision;
- zero failures.

## Provenance

- O1 protocol freeze commit: `de7624c`
- Formal runner/auditor implementation commit: `6f8a563eebdc26c368717c5093160afbc66f1cd6`
- Protocol config SHA-256:
  `854a2be60f5103dcced08e26f6b89f7005cb007b170d9c507ffb29ef7728b56c`
- Results SHA-256:
  `baf1a293171f05a8866546cebf62cacc4d92bb35d661cf0df9f5d4f7911c94c1`
- Manifest SHA-256:
  `5080228b3452601a9d72c34a8cc4e8f26c51c11584a2c082c33662459029294a`
- Independent-audit SHA-256:
  `bb2ea88de8fc2dba55a7cab55fdbc6d3992b5810f7f6b1c2a69a009642328961`
- External artifact root:
  `E:/DATA/T2C-Flow/processed/gaugeflow_h0_v8/occupational_order_o1_v1/`

