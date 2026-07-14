# Gate A3 early-branching two-target mechanism screen

## Status

`two_target_not_passed`. This is a two-target direct-irrep mechanism test only. It does not
change Gate A v1/A1/A2, start S2, launch a 4/8-target extension, or claim Gate
A passage.

## Fixed results

| variant                           |   checkpoint_step |   final_training_loss |   identification_training_loss |   early_own_target_retrieval_accuracy |   all_time_own_target_retrieval_accuracy |   mean_own_target_margin |   mean_own_flow_error |   mean_all_negative_flow_error | own_not_worse_than_all_negatives   | common_noise_early_branch_pass   |   generated_between_within_ratio |   generated_nearest_centroid_accuracy |   sampling_failure_count |   decoded_endpoint_retrieval_accuracy |   common_noise_terminal_type_logit_rms | common_noise_argmax_composition_equal   | continuous_control_without_discrete_branch_change   | eligible_for_expansion   | two_target_pass   | checkpoint_sha256                                                |
|:----------------------------------|------------------:|----------------------:|-------------------------------:|--------------------------------------:|-----------------------------------------:|-------------------------:|----------------------:|-------------------------------:|:-----------------------------------|:---------------------------------|---------------------------------:|--------------------------------------:|-------------------------:|--------------------------------------:|---------------------------------------:|:----------------------------------------|:----------------------------------------------------|:-------------------------|:------------------|:-----------------------------------------------------------------|
| fm_only                           |               200 |               1.78484 |                        0       |                                   0.8 |                                   0.6875 |                  0.14592 |               1.97026 |                        2.11617 | True                               | True                             |                          1.01973 |                                 0.875 |                        0 |                                 0.5   |                                0.09526 | False                                   | False                                               | False                    | False             | 5c117cba2c88879f42c613d9ecdec0dd50e204c9df490494e355c0ed2e3c736a |
| fm_only                           |               400 |               1.76992 |                        0       |                                   0.6 |                                   0.5625 |                  0.04784 |               1.92384 |                        1.97168 | True                               | True                             |                          1.01222 |                                 0.75  |                        0 |                                 0.375 |                                0.06456 | False                                   | False                                               | False                    | False             | 7708f9f417d23b3a9289ab667e9965bf33698c8faa93ba22986ba1355f933b24 |
| early_all_negative_identification |               200 |               1.81622 |                        0.06543 |                                   0.8 |                                   0.6875 |                  0.16369 |               1.9736  |                        2.13729 | True                               | True                             |                          1.02022 |                                 0.875 |                        0 |                                 0.5   |                                0.10247 | False                                   | False                                               | True                     | False             | 7776dc3787c0f3a2695096b9b16ad77a852eea4c1cda6c820890a8490c0e3673 |
| early_all_negative_identification |               400 |               1.86943 |                        0.18963 |                                   0.7 |                                   0.5    |                  0.04641 |               1.94154 |                        1.98794 | True                               | True                             |                          1.01288 |                                 0.75  |                        0 |                                 0.375 |                                0.07635 | False                                   | False                                               | True                     | False             | 42eaf9b3d4f99d191ba5986a13eebfe583157583209c19f65f6090d7f52a0146 |

## Decoded-state boundary

`decoded_state_audit.csv` records argmax atom types, composition, lattice
shape/volume, a permutation-invariant fractional pair-distance spectrum,
nearest-neighbor topology, and endpoint retrieval. A true value in
`continuous_control_without_discrete_branch_change` means that a terminal
continuous type-logit difference did not change the matched-noise argmax
composition; it is not evidence of a discrete generative branch.

## Advancement boundary

Only `early_all_negative_identification` can satisfy this protocol's criteria.
Even a passing two-target result requires a new versioned protocol before a
4-target or 8-target screen. A failure requires review of the probability path,
atom-type manifold, decoder, and flow-target definition rather than another
conditional-module search.
