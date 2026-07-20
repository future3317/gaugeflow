# E1.2 failure attribution

E1.2 failed every frozen semantic threshold while satisfying numerical,
gradient, throughput, memory, atom-count, and failure checks.  Mean
teacher-forced NLL ratio was 0.66916, but `t=0.5` top-1/top-5 was
0.40920/0.57098, `t=0.9` top-1 was 0.07811, and reverse exact composition was
0/256.  L1 and later Gates remain prohibited.

The dedicated graph posterior did not repair the E1.1 bottleneck.  Reverse
count overlap changed only from 0.08144 to 0.08684 and mean graph count L1 from
18.59375 to 18.48438.  Oracle target counts still raise terminal site accuracy
from 0.05944 to 0.69664, so global composition inference remains primary.

## Current-state information audit

The same checkpoint was evaluated without further training:

| element time | noisy-input count overlap | graph-posterior overlap | clean-token oracle overlap | clean-token oracle exact |
|---:|---:|---:|---:|---:|
| 0.25 | 0.85913 | 0.68352 | 0.70629 | 0.08594 |
| 0.50 | 0.52528 | 0.53300 | 0.68545 | 0.06250 |
| 0.75 | 0.16442 | 0.18950 | 0.64107 | 0.05078 |
| 0.90 | 0.05210 | 0.10884 | 0.59977 | 0.02344 |

At low noise, the learned graph head discards useful count information already
present in the current categorical state.  At high noise, it improves on the
nearly uniform input but remains far from identifying the target.  Even when
clean tokens are exposed only as an oracle, the head does not exactly count
them because their histogram reaches it only through learned embeddings,
message passing, pooling, and a second compression.

The next bounded repair is therefore not a wider graph MLP.  It is an explicit
exchangeable current-token histogram, which is a permutation-invariant
sufficient statistic for the global occupancy under uniform categorical
corruption.  A time-gated residual posterior must preserve this statistic at
the clean boundary and learn only the correction needed as categorical
information is destroyed.  This remains current-state information, not target
composition leakage.
