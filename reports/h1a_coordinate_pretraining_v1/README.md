# H1a coordinate-only pretraining v1

Status: **frozen, not run**.

Exact quotient scores close generic, real, and high-symmetry endpoints, and no
alternative score integrator improves the failed joint checkpoint.  The prior
gradient audit instead measured that element and lattice objectives dominate
the useful low-noise coordinate gradient under global clipping.  This protocol
therefore trains the unchanged 4.47M-parameter model from scratch for exactly
one full train pass using only the Rao--Blackwell coordinate DSM objective.

This is a representation/optimization qualification, not a generative result:
element, volume and shape heads are deliberately not optimized.  Passing
requires held-out coordinate loss, teacher-forced endpoint estimation, and
100-step oracle-context rollout to pass simultaneously.  It permits only a
separately frozen joint initialization experiment.  It cannot qualify H1a or
start H1b.
