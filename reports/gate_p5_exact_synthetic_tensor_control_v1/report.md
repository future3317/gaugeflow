# P5 exact synthetic tensor-control gate

Status: `not_passed_exact_synthetic_control`. The property is recomputed analytically from generated coordinates; no learned oracle, real piezo label, relaxation, DFT, or DFPT is used.

- Teacher SO(3) equivariance error: `2.98023224e-07`
- Two-target finite-grid orbit distance: `1.04069185e+00`
- Every pre-registered seed passes: `False`

|   seed |   final_flow_loss |   exact_teacher_target_retrieval |   target_orbit_distance_mean |   target_orbit_distance_other_mean |   between_exact_property |   within_exact_property |   between_within_exact_property_ratio |   common_noise_representative_coordinate_rms |   sampling_failures |
|-------:|------------------:|---------------------------------:|-----------------------------:|-----------------------------------:|-------------------------:|------------------------:|--------------------------------------:|---------------------------------------------:|--------------------:|
|   5101 |         0.0711403 |                              0   |                     1.83873  |                           1.56109  |                  2.34101 |               0.0986099 |                               23.7401 |                                   0.0025553  |                   0 |
|   5102 |         0.074532  |                              0.5 |                     0.643554 |                           0.783069 |                  2.38876 |               0.127411  |                               18.7485 |                                   0.00258529 |                   0 |
|   5103 |         0.0847335 |                              0.5 |                     0.973611 |                           0.999244 |                  2.41932 |               0.114658  |                               21.1003 |                                   0.00368775 |                   0 |

This is only a coordinate-substrate test with fixed atom types and lattice metric. It cannot be used as evidence for joint crystal generation or real piezoelectric control.
