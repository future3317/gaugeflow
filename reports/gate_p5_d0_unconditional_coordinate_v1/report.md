# P5-D0 single-endpoint unconditional coordinate-flow qualification

Status: `not_passed_unconditional_coordinate_substrate`. Failure attribution: `training_fit_failure`. P5-D1 allowed: `False`.

This model receives only the current flow state, graph index, and time. It receives no tensor, condition mask/null token, endpoint ID, harmonic alignment/grid, CFG, learned oracle, or real-material response.

- Analytic teacher SO(3) equivariance error: `2.98023224e-07`
- Every pre-registered seed passes: `False`

|   seed |   final_flow_loss |   periodic_coordinate_rms |   analytic_teacher_target_orbit_error |   analytic_teacher_target_orbit_relative_error |   unique_endpoint_retrieval |   teacher_forced_endpoint_rms |   teacher_forced_coordinate_velocity_mse |   sampling_failures |
|-------:|------------------:|--------------------------:|--------------------------------------:|-----------------------------------------------:|----------------------------:|------------------------------:|-----------------------------------------:|--------------------:|
|   5201 |         0.0863978 |                  0.299286 |                               1.00924 |                                       0.818402 |                           0 |                      0.139389 |                                0.0806189 |                   0 |
|   5202 |         0.0428109 |                  0.298142 |                               1.23731 |                                       1.00335  |                           0 |                      0.135491 |                                0.0726723 |                   0 |
|   5203 |         0.0862262 |                  0.318114 |                               1.17961 |                                       0.956559 |                           0 |                      0.146098 |                                0.0858893 |                   0 |

The frozen P5 conditional negative result is not modified. This report does not activate P5-D1 unless its manifest says `p5_d1_allowed: true`; it does not authorize P3, P4, oracle, real tensor, relaxation, DFT, or DFPT.
