# P5-D0.3 translation-quotient metric coordinate-flow qualification

Fixed-batch qualification passed: `True`. Attribution: `source_coupling_generalization_failure`. P5-D1 allowed: `False`.

This authorized one-endpoint test uses no tensor, endpoint ID, CFG, or harmonic input. It does not modify historical D0/D0.1/D0.2/P5 evidence and does not authorize a subsequent Gate.

|   model_seed |   fixed_batch_velocity_mse |   fixed_batch_translation_aligned_endpoint_rms |   fixed_batch_absolute_origin_rms_diagnostic |   unseen_velocity_mse |   unseen_translation_aligned_endpoint_rms |   unseen_absolute_origin_rms_diagnostic |   free_running_translation_aligned_endpoint_rms |   free_running_absolute_origin_rms_diagnostic |   sampling_failures | passed   | attribution                            |
|-------------:|---------------------------:|-----------------------------------------------:|---------------------------------------------:|----------------------:|------------------------------------------:|----------------------------------------:|------------------------------------------------:|----------------------------------------------:|--------------------:|:---------|:---------------------------------------|
|         5201 |                8.95226e-05 |                                     0.00482586 |                                     0.144908 |              0.200939 |                                  0.166554 |                                0.220714 |                                        0.210714 |                                      0.212558 |                   0 | True     | source_coupling_generalization_failure |
