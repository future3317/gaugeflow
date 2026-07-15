# P5-D0.6 Quotient Rollout-Corrected Flow Map

Passed: `False`. Attribution: `rollout_or_free_running_failure`. No subsequent gate is automatically authorized.

The model predicts finite quotient maps for `(s,u)` with Fourier interval features and FiLM in every message block. Training is exactly L_on + L_corr with a detached first rollout; sampling composes finite maps directly. No velocity Euler update, endpoint bridge coefficient, tensor, harmonic, or unseen source appears in this protocol.

|   model_seed |   fixed_source_count |   time_grid_count |   all_time_flow_map_mse |   teacher_forced_translation_aligned_rms |   free_running_translation_aligned_rms |   sampling_failures | passed   | attribution                     |
|-------------:|---------------------:|------------------:|------------------------:|-----------------------------------------:|---------------------------------------:|--------------------:|:---------|:--------------------------------|
|         5201 |                   64 |                33 |              3.6427e-05 |                                0.0445661 |                               0.232736 |                   0 | False    | rollout_or_free_running_failure |
