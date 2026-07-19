# H1a reverse-SDE-50 non-inferiority v1

Status: **completed; failed**.

This zero-training Gate used 512 validation structures with zero overlap with
the earlier 256-structure sampler panel. Atom types and lattices were fixed to
their real values. Reverse-SDE-50 and reverse-SDE-100 shared initial states and
a nested ancestral Brownian-bridge path. The structure-level bootstrap used
2,000 paired resamples.

| Metric | SDE-50 | SDE-100 | Frozen requirement |
|---|---:|---:|---:|
| empirical NN W1 (A) | 0.40174 | 0.35789 | UCB95 difference <=0.03 A |
| W1 difference UCB95 (A) | 0.05767 | reference | <=0.03 |
| valid distance fraction | 1.00000 | 1.00000 | degradation <=0.005 |
| minimum-distance q01 (A) | 0.99176 | 1.04334 | degradation <=0.03 A |
| minimum-distance q05 (A) | 1.26760 | 1.34627 | degradation <=0.03 A |
| endpoint periodic RMS (A) | 2.56643 | 2.55008 | relative degradation <=5% |
| latency (s) | 143.54 | 289.55 | ratio <=0.60 |

Both paths are finite with zero failures. SDE-50 passes latency, endpoint RMS,
valid-distance rate, and finite-state checks. It fails the W1 upper confidence
bound and both direct and bootstrap lower-tail checks. The apparent closeness
on the earlier panel does not replicate under the independent qualification.

Decision: retain reverse-SDE-100 for coordinate-only audits and stop sampler
search. This result does not evaluate free generation, element composition, or
lattice generation and does not change the failed H1a status.
