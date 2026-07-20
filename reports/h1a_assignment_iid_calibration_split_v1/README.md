# H1a assignment IID calibration split v1

Decision: `freeze the split manifest for a future from-scratch oracle-C IID assignment Gate; the existing Q1 split remains the distinct OOD stress panel`.

The IID calibration/test rows are drawn only from the original GaugeFlow
train partition. Original validation and test remain untouched OOD stress
panels, so one checkpoint can report IID calibration and OOD generalization
without mixing their scientific meanings.

| role | materials | carriers |
|---|---:|---:|
| iid_fit | 98 | 174 |
| iid_fit_rare | 37 | 90 |
| iid_calibration | 23 | 42 |
| iid_test | 23 | 52 |
| ood_validation | 27 | 43 |
| ood_test | 16 | 53 |

IID calibration partition support: `1.000000`.
IID test partition support: `1.000000`.
Exact input-output duplicate role overlap: `0`.

Formula/prototype overlap inside these IID roles is intentional; it tests
calibration on supported contexts. Formula/prototype-disjoint evidence is
reported only on the untouched original OOD panels.

Boundary: This split cannot qualify assignment, generated-C, p(N), L1/M1, tensor work, relaxation, DFT or DFPT.
