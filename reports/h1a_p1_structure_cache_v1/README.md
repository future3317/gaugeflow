# Current H1a packed structure cache

**Decision: `qualified`.**

This is a data-plane result only. It does not train or qualify the generator.

| Metric | Result |
|---|---:|
| Source rows | 675204 |
| Rebuilt rows | 675204 |
| Processing failures | 0 |
| Maximum source equivalence error (A) | 8.09606e-15 |
| Maximum float32 cache equivalence error (A) | 2.78761e-06 |
| Train/val/test rows | 540164 / 67520 / 67520 |
| Train/val/test nodes | 5161621 / 673217 / 644718 |
| Independent audit wall seconds | 140.71 |

A qualified result permits only freezing a separate H1a training protocol.
H1b and H2--H6 remain unauthorized.
