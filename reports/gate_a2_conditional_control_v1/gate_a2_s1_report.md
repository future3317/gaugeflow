# Gate A2 S1 conditional-control screen

## Status

`s1_direct_irrep_not_passed`.  This report evaluates only the pre-registered direct-irrep S1
mechanism screen at 400 and 800 steps.  It does not change Gate A v1, activate
v2, run S2, or claim Gate A passage.

## Fixed-combination results

| variant                         |   checkpoint_step | conditional_control   |   condition_dropout |   counterfactual_weight |   final_training_loss |   counterfactual_training_loss |   condition_shuffle_gap |   generated_between_within_ratio |   own_target_win_rate |   mean_own_target_margin |   mean_own_flow_error |   mean_wrong_condition_flow_error | own_not_worse_than_wrong   | common_noise_terminal_pass   |   sampling_failure_count_cfg0 | all_residual_heads_recorded   | s1_pass   | checkpoint_sha256                                                |
|:--------------------------------|------------------:|:----------------------|--------------------:|------------------------:|----------------------:|-------------------------------:|------------------------:|---------------------------------:|----------------------:|-------------------------:|----------------------:|----------------------------------:|:---------------------------|:-----------------------------|------------------------------:|:------------------------------|:----------|:-----------------------------------------------------------------|
| original_injection              |               400 | original_injection    |                 0   |                    0    |               1.86492 |                        0       |                 0.09533 |                          1.00027 |               0.61111 |                  0.20326 |               1.95772 |                           2.16098 | True                       | True                         |                             0 | True                          | False     | 2baaac58eaf55ee9f01349bf7d9130efab260cbd8e81c4981e48b89403832834 |
| original_injection              |               800 | original_injection    |                 0   |                    0    |               1.77164 |                        0       |                 0.11691 |                          1.00683 |               0.63889 |                  0.22056 |               1.83729 |                           2.05785 | True                       | True                         |                             0 | True                          | False     | 217af5e3821bc86a0797aa8845fccebf6d843aab0f270263849eb936fe7ce3ef |
| residual_field                  |               400 | residual_field        |                 0   |                    0    |               1.8379  |                        0       |                 0.08681 |                          0.99711 |               0.54167 |                  0.16686 |               1.94284 |                           2.1097  | True                       | True                         |                             0 | True                          | False     | b331901d34588336e8fff9352ef477464056bd90e754a6f61c673073dea867b3 |
| residual_field                  |               800 | residual_field        |                 0   |                    0    |               1.7249  |                        0       |                 0.14112 |                          1.00381 |               0.55556 |                  0.2415  |               1.82725 |                           2.06874 | True                       | True                         |                             0 | True                          | False     | 1c93a8faa7501276795ca37f58fd0450cf7815bc5dbc948cad2d04a76bd9efb7 |
| residual_counterfactual         |               400 | residual_field        |                 0   |                    0.25 |               2.01706 |                        0.65453 |                 0.11585 |                          0.99807 |               0.59722 |                  0.23218 |               1.94606 |                           2.17824 | True                       | True                         |                             0 | True                          | False     | 03a26f45d0689ff61c3c522be296d4d16a11af0cd785a86af2e137e0e1ba5712 |
| residual_counterfactual         |               800 | residual_field        |                 0   |                    0.25 |               1.90162 |                        0.50975 |                 0.22069 |                          1.00685 |               0.56944 |                  0.41504 |               1.82796 |                           2.243   | True                       | True                         |                             0 | True                          | False     | 23e30106323086724bd0377df4accc6640f9b93140d8f1c6cab0084d503b7507 |
| residual_counterfactual_dropout |               400 | residual_field        |                 0.1 |                    0.25 |               1.65356 |                        0.6702  |                 0.13694 |                          0.99759 |               0.56944 |                  0.24784 |               1.9283  |                           2.17614 | True                       | True                         |                             0 | True                          | False     | 27bb207c46ee7a020a7723237b591caae7c798262092293f7016519816665ac0 |
| residual_counterfactual_dropout |               800 | residual_field        |                 0.1 |                    0.25 |               1.63154 |                        0.60728 |                 0.12979 |                          0.99828 |               0.63889 |                  0.24681 |               1.82642 |                           2.07323 | True                       | True                         |                             0 | True                          | False     | a4ead5336c84c3cabb7ee246c0e35128b69db11c1eec6f8c73793829a39202e1 |

## Interpretation boundary

The main sampling result is always CFG=0.  CFG=1 appears only for the
graphwise-dropout variant as a pre-registered supplement, never as a
replacement result.  `velocity_component_curves.csv` records base velocity,
conditional residual, g(t)-weighted residual, and residual/base ratio for all
three heads.  `common_noise_trajectory.csv` records the matched-noise terminal
state path; `teacher_forced_ranking.csv` retains every target/time comparison.

S2 remains locked.  It may be started only by a separate command after a
direct-irrep S1 result passes every registered criterion and the selected
conditional-control backbone/loss is explicitly frozen.
