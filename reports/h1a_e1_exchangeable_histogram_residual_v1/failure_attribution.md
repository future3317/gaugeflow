# E1.3 failure attribution

E1.3 failed the frozen semantic thresholds and does not authorize L1 or any
later Gate.  It passed numerical, gradient, throughput, memory, atom-count,
mask, and sampling-failure checks.  Mean teacher-forced NLL ratio was 0.70436;
free reverse exact composition and assignment remained 0/256.

The exchangeable histogram residual repaired exactly the implementation defect
it targeted:

| metric | E1.2 graph head | E1.3 histogram residual |
|---|---:|---:|
| `t=.25` composition overlap | 0.68352 | **0.87534** |
| `t=.25` exact composition | 0.05469 | **0.27734** |
| `t=.25` clean-token oracle exact | 0.08594 | **0.89062** |
| `t=.9` composition overlap | 0.10884 | 0.08530 |
| reverse composition overlap | 0.08684 | 0.06831 |
| reverse site accuracy | 0.05944 | 0.03396 |

At low noise the exact current-state histogram is now preserved and the model
can count a revealed clean state.  At high noise, however, the uniform
site-token path contains almost no coherent formula information.  The residual
cannot infer an unseen global species multiset reliably from a nearly uniform
independent site state; improving the low-noise boundary therefore does not
bootstrap the free reverse trajectory.

Oracle target counts still raise terminal site accuracy to 0.70089 and exact
assignment to 0.35547.  Site ranking is not the primary limitation.  The
evidence now rejects another deterministic composition head, wider local
features, more exposure, loss-weight search, or sampler tuning on this state
space.

The next scientific question is whether occupational generation must be
factorized into an explicit graph composition state and a count-constrained
site assignment state.  That proposal must first be supported by a train-split
composition-complexity audit and an exact small-state kernel/normalization
qualification.  No further E1 training is authorized in this protocol.
