# P5-D0.5 endpoint-bridge metric coordinate-flow qualification

Passed: `False`. Attribution: `endpoint_residual_fit_failure`. No subsequent gate is automatically authorized.

The model predicts the translation-quotient residual to the endpoint, not a source-ambiguous raw terminal velocity. The sampler applies the bounded exact bridge contraction. This run uses only the same 64 fixed sources.

|   model_seed |   fixed_source_count |   time_grid_count |   all_time_grid_endpoint_residual_mse |   teacher_forced_translation_aligned_rms |   free_running_translation_aligned_rms |   sampling_failures | passed   | attribution                   |
|-------------:|---------------------:|------------------:|--------------------------------------:|-----------------------------------------:|---------------------------------------:|--------------------:|:---------|:------------------------------|
|         5201 |                   64 |                33 |                            0.00113232 |                                0.0313744 |                               0.175085 |                   0 | False    | endpoint_residual_fit_failure |
