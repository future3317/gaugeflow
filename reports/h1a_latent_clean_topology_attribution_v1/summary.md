# H1a latent clean-topology attribution v1

Decision: **audit_invalid_clean_topology_mass_not_covered**.

This is a zero-optimizer frozen-checkpoint diagnostic. It does not add a production branch or change H1a.

## Checks

- middle_topology_is_disrupted: `False`
- clean_topology_mass_is_covered: `False`
- clean_topology_oracle_helps: `False`
- clean_topology_probe_is_predictive: `False`
- probe_weighted_carrier_retains_oracle_gain: `False`

## Decision metrics

- middle_soft_jaccard: `0.103387`
- middle_hard_switch_fraction: `0.051815`
- minimum_clean_mass_coverage: `0.582605`
- oracle_middle_mean_improvement: `0.002419`
- noisy_middle_mean_improvement: `-0.000173`
- oracle_minus_noisy_middle_mean: `0.002592`
- oracle_middle_supporting_times: `0.000000`
- probe_middle_mean_explained_fraction: `0.056363`
- probe_middle_mean_auc: `0.727089`
- probe_middle_mean_improvement_over_noisy: `0.391374`
- learned_middle_mean_improvement: `-0.000410`
- learned_to_oracle_improvement_ratio: `-0.169543`
