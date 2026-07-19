# H1a nonlinear pair conversion qualification v1

Two matched `O(E C)` equivariant MLP readouts were fit for 1,500 steps at
`t=0.6`: one reads current-state pair features, while the other additionally
reads frozen probe and quotient-Tweedie topology. The generator checkpoint is
unchanged and receives zero optimizer steps.

Both readouts overfit the fitting panel and worsen held-out residual energy.
The topology readout's incremental held-out gain over the matched base is only
`0.005375`; its structure-bootstrap 95% interval
`[-0.005030, 0.005639, 0.015699]` crosses zero. The frozen decision is
`state_derived_pair_conversion_insufficient_conditional_variance`. This result
does not authorize ACF, another topology branch, H1b--H6, tensor conditioning,
oracle work, relaxation, DFT or DFPT.
