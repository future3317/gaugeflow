# H1 harmonic covariance audit

This is a deterministic operator audit, not a training or generation result.

For the degree-`l` query `q_l(gx)=rho_l(g)q_l(x)` and the score `s(R;x,e)=sum_l,m w_lm <rho_l(R)e_lm,q_l(x)>/sqrt(2l+1)`, orthogonality of `rho_l` gives `s(R;gx,he)=s(g^{-1}Rh;x,e)`. The continuous score is tested directly at the transformed nodes; it is distinct from a sampled grid posterior.

- seed: `20260715`
- continuous-score theorem status: `True`
- overall audit status: `True`

| test                                |       value | expectation                  | result   |
|:------------------------------------|------------:|:-----------------------------|:---------|
| continuous_score_covariance         | 3.37045e-08 | <= 5e-5                      | True     |
| high_symmetry_tensor_representative | 1.48407e-08 | <= 5e-7                      | True     |
| high_symmetry_score_representative  | 8.05183e-09 | <= 5e-5                      | True     |
| zero_tensor_uniform_posterior       | 0           | <= 1e-12                     | True     |
| finite_grid_identity_shift          | 0           | <= 1e-12                     | True     |
| finite_grid_left_shift_nonclosure   | 1.80675     | > 1e-4 (reported limitation) | True     |
| finite_grid_right_shift_nonclosure  | 1.77612     | > 1e-4 (reported limitation) | True     |

The positive nonidentity left/right residuals are expected: the finite Hopf QMC grid is not a group and therefore has no exact generic left/right reindexing. They are not a threshold for a generation gate; a later protocol must pre-register its grid/refinement choice.
