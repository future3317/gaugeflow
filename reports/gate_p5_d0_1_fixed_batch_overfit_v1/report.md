# P5-D0.1 fixed-batch unconditional coordinate-flow overfit audit

Fixed training batch overfit: `False`. Attribution: `model_or_loss_cannot_memorize`. P5-D1 allowed: `False`.

The 64 train `(source noise, t)` pairs were generated before model construction, hashed, and repeated unchanged for exactly 5,000 updates. No tensor, condition mask, endpoint ID, harmonic module, CFG, or resampling is used.

|   model_seed |   fixed_batch_velocity_mse |   fixed_batch_endpoint_rms | fixed_train_batch_overfit_passed   |   unseen_teacher_forced_endpoint_rms |   free_running_endpoint_rms | failure_attribution           |
|-------------:|---------------------------:|---------------------------:|:-----------------------------------|-------------------------------------:|----------------------------:|:------------------------------|
|         5201 |                  0.0497715 |                   0.108541 | False                              |                             0.176281 |                   0.0850646 | model_or_loss_cannot_memorize |

The P5-D0 result remains immutable. This audit ends here and does not authorize P5-D1, P3, P4, P6, oracle, real tensor, relaxation, DFT, or DFPT.
