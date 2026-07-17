# H0-E-v4 O0-v2 occupational-order mechanism Gate

## Decision

`H0-E-v4-O0-v2_qualified_only_held_out_O1_protocol_may_be_frozen`.

The cleaned, frozen 63-material mechanism panel produced 10 materials with 13
qualified parent occurrences. This exceeds the preregistered minimum of three
materials. Every occurrence uses a species-free geometric carrier and an exact
integer occupational coloring; no partial occupancy, dummy physical element,
target composition input or tolerance expansion is involved.

O0-v2 is an implementation/causal qualification on a mechanism-locating
panel. It does **not** qualify H0-E or H0 and does not authorize H1a. It permits
only a separately frozen held-out O1 occurrence protocol.

## Frozen result

| Metric | Result | Requirement | Pass |
|---|---:|---:|:---:|
| Selected rows | 63 | 63 | yes |
| Material exclusions applied before enumeration | 1 | 1 | yes |
| Candidate / eligible edges | 962 / 962 | 962 / 962 | yes |
| Processing failures / nonfinite values | 0 / 0 | 0 / 0 | yes |
| New candidate materials | 10 | >= 3 | yes |
| New candidate-material fraction | 0.158730 | >= 0.047619 | yes |
| Occupationally nontrivial materials | 10 | >= 3 | yes |
| Qualified embedding occurrences | 13 | diagnostic | -- |
| Forward/reverse embedding-set agreement | 1.0 | 1.0 | yes |
| Exact coloring reconstruction | 1.0 | 1.0 | yes |
| Occupational stabilizer subgroup certificate | 1.0 | 1.0 | yes |
| Stabilizer order equals child operation order | 1.0 | 1.0 | yes |
| Strict geometry-parent reidentification | 1.0 | 1.0 | yes |
| Terminal integer-element fraction | 1.0 | 1.0 | yes |
| Partial occupancies | 0 | 0 | yes |
| Maximum projected group error | 3.161e-14 A | <= 1e-7 A | yes |
| Maximum source displacement | 0.127430 A | <= 0.2 A | yes |
| Maximum source Hencky norm | 2.115e-15 | <= 0.15 | yes |
| Total / p95 per-row runtime | 242.84 s / 12.955 s | diagnostic | -- |

The independent auditor rebuilt the panel in reverse record and split order.
It reproduced all 63 rows, 962 edges, 10 candidate materials, 13 occurrences,
every candidate array/scalar and the final decision, with no rebuild failure.

## Qualified materials

| Material | Child -> parent | Kind/index | Full action -> coloring stabilizer | Max displacement (A) |
|---|---|---|---:|---:|
| `alex<agm004705110>` | 229 -> 221 | k/4 (two distinct E0 embeddings) | 192 -> 48 | 0.083729 |
| `alex<agm001172261>` | 216 -> 227 | t/1 | 48 -> 24 | 0.000288 |
| `alex<agm003566725>` | 193 -> 191 | k/2 (two distinct E0 embeddings) | 48 -> 24 | 0.045757 |
| `alex<agm005206758>` | 38 -> 63 | t/1 | 8 -> 4 | 0.028273 |
| `alex<agm004917812>` | 160 -> 166 | t/1 | 12 -> 6 | 0.098789 |
| `alex<agm002272226>` | 216 -> 227 | t/1 | 48 -> 24 | 0.005955 |
| `alex<agm003641678>` | 123 -> 139 | k/2 | 32 -> 16 | 0.127430 |
| `alex<agm005016853>` | 119 -> 139 | t/1 | 16 -> 8 | 0.000000 |
| `alex<agm002232727>` | 198 -> 212 | t/1 | 24 -> 12 | 0.050301 |
| `alex<agm005089321>` | 194 -> 191 | k/2 (two distinct E0 embeddings) | 48 -> 24 | 0.020922 |

The repeated material/parent rows above have distinct source-certified affine
embedding keys. They do not inflate the Gate statistic, which counts distinct
materials. A future O1 protocol must retain E0 physical-path canonicalization
and report both material and path-class coverage.

## Mechanism closure

The old implementation assigned species before constructing the parent group
action. That required every parent operation to preserve terminal elements and
therefore rejected chemical ordering as a legitimate symmetry-breaking field.
O0-v2 instead uses

```text
X_geom = (L, F, G),
H_a = {g in G^B : P_g a = a},
H_child = H_a intersect H_modes intersect H_strain intersect H_residual.
```

The offline projector uses one internal equivalence class only to solve the
geometry action, strips that label from its result, and then evaluates the
full 118-class terminal coloring. Production reconstruction now requires a
`ParentGeometryCarrier` and an `OccupationalPattern`; the retired
species-copying `ParentCrystal` path no longer exists.

Positive controls cover binary symmetry breaking, uniform-color recovery of
the full group, global element-name relabeling, node-permutation conjugacy,
mode/occupation subgroup intersection, maximal-t recovery and an off-diagonal
index-two maximal-k quotient. The complete repository validation at the
implementation commit was 155 tests passed, Ruff clean, production mypy clean
and zero active redundancy findings.

## Data cleaning

`alex<agm004639609>` is excluded before candidate enumeration by
`parent_occurrence_quarantine_v2`. The immutable raw Alex row and all frozen
v1--v3 histories remain unchanged. The filtered ordered-ID hash is
`6e78ebd1c47ae94f770bc04bb3fc1c0ae89088cb4509cea6dec621cc4069f25b`.

## Provenance

- O0-v1 protocol freeze commit: `7fe67d53c3a6e2ef32f4b478a617062208566e37`
- Material-cleaning/O0-v2 protocol commit: `268c011`
- Occupational-order implementation commit: `d558776`
- Formal runner/auditor commit used by the run:
  `a77e7131df1de6eb8fc9858532d1a8f6157c6666`
- Results SHA-256:
  `a0f9178d93c65c2d5232aa778f316e6dea324b5814f3ea480dfd3e4b27e4f40c`
- Manifest SHA-256:
  `9beae4f6d8934881f0c2c24096845d84ef93bce04978ed69e55b58aa462526c3`
- Independent-audit SHA-256:
  `5257c2f5044dfa779bb8aad5d591a7a79f1a1687895013a784e9f548cdeb27d9`
- External artifact root:
  `E:/DATA/T2C-Flow/processed/gaugeflow_h0_v7/occupational_order_o0_v2/`

