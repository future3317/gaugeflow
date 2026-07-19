# H1a all-pair clean-topology attribution v2

Decision: **probe_predictive_but_topology_correction_not_residual_causal**.

This is a zero-optimizer frozen-checkpoint diagnostic. It does not add a production branch or change H1a.

## Checks

- middle_topology_is_disrupted: `True`
- clean_topology_mass_is_covered: `True`
- clean_topology_oracle_helps: `True`
- clean_topology_probe_is_predictive: `True`
- probe_weighted_carrier_retains_oracle_gain: `False`

## Decision metrics

- middle_soft_jaccard: `0.504131`
- middle_hard_switch_fraction: `0.264689`
- minimum_clean_mass_coverage: `1.000000`
- oracle_middle_mean_improvement: `0.107163`
- noisy_middle_mean_improvement: `-0.003535`
- oracle_minus_noisy_middle_mean: `0.110698`
- oracle_middle_supporting_times: `3.000000`
- probe_middle_mean_explained_fraction: `0.613621`
- probe_middle_mean_auc: `0.879227`
- probe_middle_mean_improvement_over_noisy: `0.613693`
- learned_middle_mean_improvement: `-0.043911`
- learned_to_oracle_improvement_ratio: `-0.409763`
