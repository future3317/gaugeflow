# H1a J1 matched clock attribution

## Answer first

The matched attribution **fails** its frozen causal criterion.  Under the same
seed, five-regime mixture, 2,111-step exposure and exact 5,232,057-parameter
budget, the three-clock C2 model has essentially the same diagonal and interior
final error as the single-clock C0 control.  Their structure-paired confidence
intervals cross zero.  J1 therefore remains a successful composite intervention,
but its noisy/noisy improvement cannot be attributed specifically to named
modality clocks.

The result does not show that separate clocks are harmful.  C2 is significantly
better on clean--clean and noisy-element states and has no significant loss on
the other three regimes.  It shows that the fixed five-task training mixture,
not clock identity alone, is sufficient to obtain the earlier diagonal/interior
performance at this capacity and exposure.

## Frozen controls

All arms use Alex-MP-20 P1, seed 5705, batch 64, 2,111 steps, the exact
`13/13/13/13/12` regime allocation, identical shuffling and device-noise seeds,
BF16 learned matmuls/FP32 geometry, AdamW, EMA and coordinate target.

- C0 sees only `t_F`; its side-clock MLPs and fusion map are disconnected.
- C1 sees `t_F` and the scalar `(t_A+t_L)/2`; its second side-clock MLP and
  unused fusion columns are disconnected.
- C2 is the archived J1 model and sees `(t_F,t_A,t_L)` separately.

Every arm owns 5,232,057 parameters.  The controls therefore do not confuse
time information with nominal parameter count.  Logs verify zero gradients for
the disconnected C0/C1 dummy parameters.

## Results

Final held-out coordinate MSE on the common 256-structure/noise panel is:

| regime | C0 single | C1 side mean | C2 separate |
|---|---:|---:|---:|
| clean--clean | 0.55303 | 0.54216 | **0.52814** |
| noisy element | 0.62187 | 0.61225 | **0.59945** |
| noisy lattice | 0.62257 | **0.61921** | 0.62028 |
| diagonal | 0.67028 | 0.66935 | **0.66934** |
| interior | 0.75772 | 0.75505 | **0.75190** |

For the preregistered C2-minus-C0 structure-paired final-MSE differences:

| regime | mean | paired 95% interval |
|---|---:|---:|
| clean--clean | -0.02489 | [-0.03361, -0.01627] |
| noisy element | -0.02242 | [-0.03229, -0.01290] |
| noisy lattice | -0.00229 | [-0.01071, 0.00547] |
| diagonal | -0.00093 | [-0.00929, 0.00703] |
| interior | -0.00582 | [-0.01619, 0.00446] |

The diagonal and interior upper bounds are not below zero, so the frozen Gate
fails.  Clean C2/C0 is 0.95499, safely within the no-more-than-5% degradation
rule.  The new structure-paired adjacent-regime analysis also corrects an
earlier statistical overstatement: noisy-element minus clean, diagonal minus
noisy-lattice and interior minus diagonal are positive, but noisy-lattice minus
noisy-element remains unresolved with interval `[-0.00821, 0.05211]`.

## Interpretation and boundary

The valid statement is now:

> Five-regime teacher-forced training preserves the clean coordinate task and
> learns noisy-side coordinate regression.  Explicit named clocks help clean
> and element-only corners on this panel, but are not the identified cause of
> diagonal/interior improvement.

This does not qualify free joint generation or on-policy reverse side states.
It does not authorize more clock capacity, a hard chain, E1/L1/M1/J2, tensor
conditioning, oracle work, relaxation, DFT or DFPT.  The separately frozen
gradient-geometry audit must be interpreted before any optimizer change.
