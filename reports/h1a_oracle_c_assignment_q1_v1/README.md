# H1a oracle-C assignment Q1 v1

Status: **failed**.

This Gate evaluates only exact count-constrained site assignment conditioned on
oracle-labelled composition and a species-free parent carrier. It does not
qualify generated-C exposure, `p(N)`, lattice L1, joint M1, tensor conditioning,
relaxation, DFT, or DFPT.

| metric | validation | test |
|---|---:|---:|
| quotient NLL | 9.155383 | 8.210089 |
| uniform quotient NLL | 6.688477 | 5.186358 |
| model-uniform UCB95 | 4.742380 | 6.846176 |
| exact target quotient probability | 0.123245 | 0.220520 |
| unary-family probability ceiling | 0.590212 | 0.677123 |
| categorical sample retrieval | 0.129630 | 0.229980 |
| sampled orbit-aligned site accuracy | 0.534582 | 0.611207 |
| labeling-MAP target retrieval (diagnostic) | 0.185185 | 0.312500 |
| fixed-CIF site accuracy (diagnostic) | 0.433980 | 0.588306 |
| exact composition | 1.000000 | 1.000000 |
| failures | 0 | 0 |

Decision: `stop before generated-C, p(N), L1 or M1 and attribute failure to unary assignment energy, carrier identifiability or unresolved symmetry-inequivalent global coloring without adding target-derived inputs`.

## Independent failure attribution

The exact law and sampler are not the failure source. Exact target-quotient
probability and 32-draw categorical retrieval differ by only `0.006385` on
validation and `0.009461` on test. Every MAP and sampled assignment preserves
the oracle composition exactly, no evaluation or sampling failure occurs, and
the scorer is exactly constant under the parent action. The test relabeling
residual is `1.1444e-5`, narrowly above the frozen `1e-5` FP32 threshold, but
the much larger likelihood and retrieval failures remain even if this
borderline numerical check is ignored.

The checkpoint fits the training carriers: material-balanced train quotient
NLL is `2.77939` versus uniform `8.00377`, and target quotient probability is
`0.368097`. This is `99.86%` of the independently computed `0.368609`
site-signature unary-family ceiling. Generalization then reverses: validation
and test NLL are worse than uniform, and probabilities reach only
`0.123245/0.220520`.

The frozen material-disjoint split is a strong OOD axis. Validation and test
composition-partition support have zero overlap with training, exact parent
action-signature coverage is only `25.58%/13.21%`, and parent-space-group
coverage is `67.44%/69.81%`. Moreover, identical implemented site signatures
merge distinct parent action orbits in `11.63%/83.02%` of validation/test
carriers; their tighter unary-family probability ceilings are
`0.414904/0.304714`, below the looser parent-orbit ceilings printed in the raw
result. Thus Q1 rejects the present unary scorer on this OOD split. It does not
support more steps or target-derived inputs. A successor must first separate
IID assignment-law calibration from formula/prototype OOD evaluation and, if
needed, introduce a globally normalized occupation interaction that can
distinguish symmetry-inequivalent colorings.
