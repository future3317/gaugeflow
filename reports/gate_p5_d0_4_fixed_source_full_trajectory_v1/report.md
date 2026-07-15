# P5-D0.4 fixed-source full-trajectory coordinate-flow learning test

Passed: `False`. Attribution: `time_conditioning_or_vector_field_expression_failure`. No subsequent gate is authorized by this run.

The 64 D0.3 source noises are fixed. Each optimization update draws a new independent Uniform[0,1] time for every source. No unseen source, tensor, endpoint ID, CFG, or harmonic condition is evaluated.

|   model_seed |   fixed_source_count |   time_grid_count |   all_time_grid_velocity_mse |   teacher_forced_translation_aligned_rms |   free_running_translation_aligned_rms |   sampling_failures | passed   | attribution                                          |
|-------------:|---------------------:|------------------:|-----------------------------:|-----------------------------------------:|---------------------------------------:|--------------------:|:---------|:-----------------------------------------------------|
|         5201 |                   64 |                33 |                    0.0087122 |                                0.0422809 |                                0.19957 |                   0 | False    | time_conditioning_or_vector_field_expression_failure |
