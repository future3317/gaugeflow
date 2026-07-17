# H1a wrapped-score sampler diagnostic

The analytic grid defect is real, but changing the grid does not qualify the
learned model. This diagnostic does not alter either preceding failed Gate.

For an exact single-endpoint wrapped score, 50 `uniform_log_alpha` steps leave
mean quotient fractional RMS `0.4001`, while 50 `uniform_time` steps reduce it
to `0.02483`. The shared log-alpha grid ends with a jump from `t=0.3166` to
zero and is therefore a poor discretization of linear torus variance.

On the learned three-seed checkpoints, however, the same change lowers the
fraction of samples with minimum distance at least `0.5 A`:

| Seed | log-alpha grid | uniform-time grid |
|---:|---:|---:|
| 5201 | 0.90625 | 0.890625 |
| 5202 | 0.90625 | 0.859375 |
| 5203 | 0.921875 | 0.875000 |

The time-resolved score audit explains the discrepancy. At `t=0.01`, the
prediction/target cosine is approximately `0.19`, but the prediction norm is
only `0.55%--0.83%` of the exact target norm. At `t=0.05` it remains only
`2.2%--2.5%`. At `t>=0.5` the true wrapped score is exponentially close to
zero; the reduced aggregate coordinate loss primarily measures successful
suppression of a random initial output, not a sufficiently strong low-noise
denoising field.

The supported next repair is an exactly equivalent scaled-score
parameterization: the network predicts `sigma(t) * score` and the sampler
analytically divides by `sigma(t)`. This is the torus counterpart of
epsilon-prediction. It preserves the weighted score objective and reverse
process but keeps both the low-noise target and its parameter gradient order
one. No sampler change, new loss weight, extra network, or guidance term is
supported by this audit.
