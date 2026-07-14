# Gate A.1 conditional-to-trajectory causal audit

## Technical summary

Gate A remains **failed and unadvanced**.  The frozen 1.2 generated-target
between/within threshold was neither changed nor reinterpreted.  All four frozen methods fail the same generated-target separation control. Under the pre-specified attribution rule, the primary diagnosis is a shared conditional-injection/backbone failure rather than orbit aggregation alone.
Teacher-forced own-target ranking is positive across methods, so the additional mechanism to test next is trajectory integration/guidance rather than a wholly task-irrelevant velocity response.

This is a diagnostic-only audit of the four existing 400-step checkpoints,
the eight fixed v1 training IDs, 792 stabilizer candidates, and the original
eight-step sampling budget.  It does not train a model, activate v2, perform
relaxation/DFPT, or claim Gate A passage.

## Unified four-method result

`final_training_loss` below means the reproducible eight-draw, final-checkpoint
training-panel flow loss.  The original historical final minibatch loss was
not saved in the checkpoint and is not retroactively claimed.

| method             |   condition_shuffle_gap |   representative_velocity_error |   generated_between_within_ratio |   condition_feature_shift |   final_training_loss |   sampling_failure_count |
|:-------------------|------------------------:|--------------------------------:|---------------------------------:|--------------------------:|----------------------:|-------------------------:|
| raw_tensor         |                 0.0936  |                         0.35221 |                          1.00444 |                   2.21489 |               2.03353 |                        0 |
| direct_irrep       |                 0.10845 |                         0.21211 |                          1.00937 |                   2.0839  |               2.10048 |                        0 |
| stabilizer_pooling |                 0.1205  |                         0.11765 |                          1.00339 |                   2.05897 |               2.02035 |                        0 |
| orbit_alignment    |                 0.06634 |                         0.04523 |                          1.00664 |                   2.04028 |               2.09976 |                        0 |

The common condition-shuffle gap and feature-shift controls are nonzero, while
all generated ratios remain below the frozen `1.2` requirement.

## Common-noise counterfactual trajectories

For every method, the eight tensor conditions start from identical type-logit,
fractional-coordinate, lattice-log noise and share every subsequent sampler
operation.  `trajectory_pairwise.csv` holds all 28 target pairs at every time;
`trajectory_mean_curves.csv` and `trajectory_curves.svg` are the aggregate
curves.  The head/time summary (onset is 10% of each head's own peak) is:

| method             | quantity                       |   onset_time_at_10pct_peak |   peak_time |   peak_pairwise_rms |   terminal_pairwise_rms |   terminal_over_peak | trajectory_attenuates_after_peak   |
|:-------------------|:-------------------------------|---------------------------:|------------:|--------------------:|------------------------:|---------------------:|:-----------------------------------|
| direct_irrep       | velocity_fractional_coordinate |                          0 |       0     |              0.0306 |                  0.0113 |               0.3689 | True                               |
| direct_irrep       | velocity_lattice_log           |                          0 |       0.625 |              0.2992 |                  0.2977 |               0.9951 | False                              |
| direct_irrep       | velocity_type_logit            |                          0 |       0     |              0.095  |                  0.0612 |               0.644  | True                               |
| orbit_alignment    | velocity_fractional_coordinate |                          0 |       0.75  |              0.0298 |                  0.0285 |               0.9539 | False                              |
| orbit_alignment    | velocity_lattice_log           |                          0 |       0.875 |              0.3643 |                  0.3643 |               1      | False                              |
| orbit_alignment    | velocity_type_logit            |                          0 |       0.875 |              0.1073 |                  0.1073 |               1      | False                              |
| raw_tensor         | velocity_fractional_coordinate |                          0 |       0     |              0.0286 |                  0.0163 |               0.5706 | True                               |
| raw_tensor         | velocity_lattice_log           |                          0 |       0     |              0.3146 |                  0.2982 |               0.9479 | False                              |
| raw_tensor         | velocity_type_logit            |                          0 |       0     |              0.1002 |                  0.0596 |               0.5942 | True                               |
| stabilizer_pooling | velocity_fractional_coordinate |                          0 |       0     |              0.0268 |                  0.0197 |               0.7332 | True                               |
| stabilizer_pooling | velocity_lattice_log           |                          0 |       0     |              0.3449 |                  0.3223 |               0.9345 | False                              |
| stabilizer_pooling | velocity_type_logit            |                          0 |       0     |              0.0918 |                  0.0567 |               0.6178 | True                               |

The terminal state-distance values are:

| method             |   state_fractional_coordinate |   state_lattice_log |   state_type_logit |
|:-------------------|------------------------------:|--------------------:|-------------------:|
| direct_irrep       |                      0.015604 |            0.297747 |           0.074697 |
| orbit_alignment    |                      0.013229 |            0.281805 |           0.059508 |
| raw_tensor         |                      0.0188   |            0.306243 |           0.076009 |
| stabilizer_pooling |                      0.018842 |            0.331217 |           0.07064  |

`trajectory_dynamics_summary.csv` explicitly flags whether each trace has
fallen by at least 10% from its peak.  This separates an early conditional
velocity response from a response that survives integration as a state
difference.

## Teacher-forced own-target ranking

At the true flow interpolant, each crystal's own condition is compared with a
fixed cyclic permutation using the same base noise and time.  Positive margin
means the own condition predicts the correct flow velocity more closely.

| method             |   own_target_win_rate |   mean_margin |
|:-------------------|----------------------:|--------------:|
| direct_irrep       |                0.625  |        0.2215 |
| orbit_alignment    |                0.6528 |        0.1586 |
| raw_tensor         |                0.6944 |        0.1633 |
| stabilizer_pooling |                0.6806 |        0.1927 |

The time-resolved evidence is `teacher_forced_ranking.csv` and
`teacher_forced_ranking_curves.svg`.

## Condition representation and pooling-collapse audit

The audit exports 8-by-8 tensor-orbit and six-probe response distances, plus raw,
uniform-pooled, aligned and both offline pooling-definition embeddings.  The
current production path is `sum_k q_k phi(tensor_k)`; `phi(sum_k q_k tensor_k)`
was evaluated offline only.  No checkpoint or production method changed.

| method             | embedding                |   dimension |   effective_rank |   off_diagonal_cosine_mean |   off_diagonal_cosine_min |   off_diagonal_cosine_max |   mean_pairwise_distance |
|:-------------------|:-------------------------|------------:|-----------------:|---------------------------:|--------------------------:|--------------------------:|-------------------------:|
| raw_tensor         | raw_irrep_coordinates    |          18 |           3.6317 |                     0.0536 |                   -0.9394 |                    1      |                   5.39   |
| raw_tensor         | six_probe_response       |          18 |           3.6473 |                     0.0306 |                   -0.9054 |                    1      |                   5.2859 |
| raw_tensor         | raw_condition_embedding  |          64 |           4.3856 |                     0.4259 |                   -0.1099 |                    0.9851 |                   2.3999 |
| raw_tensor         | uniform_pooled_embedding |          64 |           4.3856 |                     0.4259 |                   -0.1099 |                    0.9851 |                   2.3999 |
| raw_tensor         | aligned_embedding        |          64 |           4.3345 |                     0.4877 |                   -0.0157 |                    0.9824 |                   2.3999 |
| raw_tensor         | phi_sum_q_tensor         |          64 |           4.3856 |                     0.4259 |                   -0.1099 |                    0.9851 |                   2.3999 |
| raw_tensor         | sum_q_phi_tensor         |          64 |           4.3345 |                     0.4877 |                   -0.0157 |                    0.9824 |                   2.3999 |
| direct_irrep       | raw_irrep_coordinates    |          18 |           3.6317 |                     0.0536 |                   -0.9394 |                    1      |                   5.39   |
| direct_irrep       | six_probe_response       |          18 |           3.6473 |                     0.0306 |                   -0.9054 |                    1      |                   5.2859 |
| direct_irrep       | raw_condition_embedding  |          64 |           4.3528 |                     0.5244 |                    0.0902 |                    0.988  |                   2.3491 |
| direct_irrep       | uniform_pooled_embedding |          64 |           4.3528 |                     0.5244 |                    0.0902 |                    0.988  |                   2.3491 |
| direct_irrep       | aligned_embedding        |          64 |           4.3093 |                     0.5779 |                    0.1671 |                    0.9849 |                   2.3491 |
| direct_irrep       | phi_sum_q_tensor         |          64 |           4.3528 |                     0.5244 |                    0.0902 |                    0.988  |                   2.3491 |
| direct_irrep       | sum_q_phi_tensor         |          64 |           4.3093 |                     0.5779 |                    0.1671 |                    0.9849 |                   2.3491 |
| stabilizer_pooling | raw_irrep_coordinates    |          18 |           3.6317 |                     0.0536 |                   -0.9394 |                    1      |                   5.39   |
| stabilizer_pooling | six_probe_response       |          18 |           3.6473 |                     0.0306 |                   -0.9054 |                    1      |                   5.2859 |
| stabilizer_pooling | raw_condition_embedding  |          64 |           4.7849 |                     0.4112 |                   -0.0433 |                    0.991  |                   3.2112 |
| stabilizer_pooling | uniform_pooled_embedding |          64 |           4.1682 |                     0.5074 |                    0.0753 |                    0.9919 |                   2.5593 |
| stabilizer_pooling | aligned_embedding        |          64 |           4.1322 |                     0.5602 |                    0.1791 |                    0.9905 |                   2.5593 |
| stabilizer_pooling | phi_sum_q_tensor         |          64 |           3.813  |                     0.5337 |                    0.0063 |                    0.996  |                   1.5035 |
| stabilizer_pooling | sum_q_phi_tensor         |          64 |           4.1322 |                     0.5602 |                    0.1791 |                    0.9905 |                   2.5593 |
| orbit_alignment    | raw_irrep_coordinates    |          18 |           3.6317 |                     0.0536 |                   -0.9394 |                    1      |                   5.39   |
| orbit_alignment    | six_probe_response       |          18 |           3.6473 |                     0.0306 |                   -0.9054 |                    1      |                   5.2859 |
| orbit_alignment    | raw_condition_embedding  |          64 |           4.2873 |                     0.4556 |                   -0.1891 |                    0.9634 |                   2.146  |
| orbit_alignment    | uniform_pooled_embedding |          64 |           3.9448 |                     0.5157 |                    0.0239 |                    0.9887 |                   1.7035 |
| orbit_alignment    | aligned_embedding        |          64 |           4.1111 |                     0.599  |                    0.1078 |                    0.9902 |                   1.4917 |
| orbit_alignment    | phi_sum_q_tensor         |          64 |           3.9227 |                     0.6728 |                    0.222  |                    0.9967 |                   0.9511 |
| orbit_alignment    | sum_q_phi_tensor         |          64 |           4.1111 |                     0.599  |                    0.1078 |                    0.9902 |                   1.4917 |

All matrices are in `condition_embedding_matrices/`; the reference-state
heatmap is `condition_embedding_heatmaps.svg`.

## Posterior diagnostics

The report distinguishes the 8-frame alignment posterior from the 792-way
crystal-automorphism posterior.  `posterior_summary.csv`,
`posterior_weights.csv`, and `posterior_pairwise_divergence.csv` report q(t),
entropy, top-mode mass, effective frame count, JSD between targets, and token
distance.  The paired mean diagnostic is:

|   time |   frame_jsd |   automorphism_jsd |   token_rms |
|-------:|------------:|-------------------:|------------:|
|  0     |    0.005231 |           0        |    0.089072 |
|  0.125 |    0.003369 |           0.006291 |    0.084319 |
|  0.25  |    0.003016 |           0.014952 |    0.087462 |
|  0.375 |    0.008663 |           0.027619 |    0.107745 |
|  0.5   |    0.028686 |           0.043761 |    0.141918 |
|  0.625 |    0.071353 |           0.056581 |    0.189349 |
|  0.75  |    0.105615 |           0.054065 |    0.24107  |
|  0.875 |    0.120993 |           0.037695 |    0.276952 |

If posterior JSD is small while tokens differ only weakly, the state produces
a shared posterior.  If posterior JSD is appreciable while token RMS is small,
the posterior differs but the downstream token is collapsing.  The saved
per-time/pair values make that distinction inspectable rather than inferred
from one scalar.

For this common-noise run, the 792-way posterior is exactly shared at t=0
(mean JSD 0) because the state is identical; its divergence and the token RMS
both grow later.  Thus the observed record is not "different posterior but the
same downstream token".  Pooling compresses the token distances (see the
embedding table), but does not alone explain the cross-method failure.

## CFG and sampler sensitivity (not protocol selection)

`sampling_sensitivity.csv` evaluates only the frozen orbit-alignment
checkpoint over the declared CFG scales and sampler steps with common random
numbers.  CFG was not trained (`condition_dropout=0.0` in this checkpoint),
so every nonzero scale is a sensitivity probe, not a valid replacement for the
frozen sampling protocol.  No row replaces the scale-0, 8-step Gate A result.

| method          | analysis                              |   cfg_scale |   sampler_steps |   within_target_distance_mean |   between_target_distance_mean |   between_within_distance_ratio |   condition_permutation_feature_shift_mean |   sampling_failure_count |
|:----------------|:--------------------------------------|------------:|----------------:|------------------------------:|-------------------------------:|--------------------------------:|-------------------------------------------:|-------------------------:|
| orbit_alignment | diagnostic_only_common_random_numbers |         0   |               4 |                       18.1325 |                        14.7163 |                         0.81159 |                                    3.24301 |                        0 |
| orbit_alignment | diagnostic_only_common_random_numbers |         0.5 |               4 |                       17.7667 |                        15.0015 |                         0.84436 |                                    4.4858  |                        0 |
| orbit_alignment | diagnostic_only_common_random_numbers |         1   |               4 |                       17.3287 |                        15.2376 |                         0.87932 |                                    5.58617 |                        0 |
| orbit_alignment | diagnostic_only_common_random_numbers |         2   |               4 |                       16.3746 |                        15.592  |                         0.95221 |                                    7.43596 |                        0 |
| orbit_alignment | diagnostic_only_common_random_numbers |         0   |               8 |                       18.1199 |                        14.7337 |                         0.81313 |                                    3.30183 |                        0 |
| orbit_alignment | diagnostic_only_common_random_numbers |         0.5 |               8 |                       17.7259 |                        15.0256 |                         0.84766 |                                    4.56213 |                        0 |
| orbit_alignment | diagnostic_only_common_random_numbers |         1   |               8 |                       17.2689 |                        15.2634 |                         0.88387 |                                    5.67109 |                        0 |
| orbit_alignment | diagnostic_only_common_random_numbers |         2   |               8 |                       16.3062 |                        15.6139 |                         0.95754 |                                    7.50471 |                        0 |
| orbit_alignment | diagnostic_only_common_random_numbers |         0   |              16 |                       18.114  |                        14.7407 |                         0.81377 |                                    3.32244 |                        0 |
| orbit_alignment | diagnostic_only_common_random_numbers |         0.5 |              16 |                       17.7126 |                        15.0355 |                         0.84886 |                                    4.59336 |                        0 |
| orbit_alignment | diagnostic_only_common_random_numbers |         1   |              16 |                       17.2556 |                        15.2718 |                         0.88503 |                                    5.69841 |                        0 |
| orbit_alignment | diagnostic_only_common_random_numbers |         2   |              16 |                       16.288  |                        15.6197 |                         0.95897 |                                    7.52102 |                        0 |

## TensorOrbit-JARVIS-v2 activation audit

The separate v2 candidate remains inactive.  Formula groups are disjoint,
ID/cache joins and response strata revalidate, explicit zero tensors are kept,
and the `StructureMatcher` prefilter has zero candidate cross-split formula
groups.  The activation audit status is `candidate_not_active_audit_complete`.  See
`tensororbit_jarvis_v2_activation_report.md` and
`artifacts/tensororbit_jarvis_v2_activation_audit/activation_protocol.json`.

## Limits and next decision

This audit establishes a causal diagnosis only for the small frozen training
panel and the existing checkpoints.  It is not orbit-tensor fidelity evidence
from a qualified external oracle, not a generalization result, and not a
physical validation.  Gate A must remain unresolved; any proposed method
change needs a new versioned protocol and a separate small test before a
larger experiment.

## Artifact index

- Unified metrics: `/mnt/e/CODE/T2C-Flow/gaugeflow_perf_audit/reports/gate_a1_causal_audit/unified_four_method_metrics.csv`
- Trajectories: `/mnt/e/CODE/T2C-Flow/gaugeflow_perf_audit/reports/gate_a1_causal_audit/trajectory_pairwise.csv` and `/mnt/e/CODE/T2C-Flow/gaugeflow_perf_audit/reports/gate_a1_causal_audit/trajectory_curves.svg`
- Teacher-forced ranking: `/mnt/e/CODE/T2C-Flow/gaugeflow_perf_audit/reports/gate_a1_causal_audit/teacher_forced_ranking.csv`
- Embeddings: `/mnt/e/CODE/T2C-Flow/gaugeflow_perf_audit/reports/gate_a1_causal_audit/condition_embedding_matrices`
- Posterior: `/mnt/e/CODE/T2C-Flow/gaugeflow_perf_audit/reports/gate_a1_causal_audit/posterior_summary.csv`
- Sensitivity: `/mnt/e/CODE/T2C-Flow/gaugeflow_perf_audit/reports/gate_a1_causal_audit/sampling_sensitivity.csv`
