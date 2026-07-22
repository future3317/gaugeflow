# Stage-B MatPES lattice-domain audit

The first formal Stage-B-v1 launch stopped after one optimizer update before
writing a learned checkpoint. Rank 1 encountered an explicit-geometry/chart
metric mismatch on its sixth physical batch. The frozen v1 protocol file SHA
is `74a453bdae70b7803a8f5861d78db18b60260538ed2f877a4d0d10f89625058f`;
its failed launch is not treated as physical-learning evidence.

The triggering indexed row has a lattice metric condition number of about
`1.08e5` and a `0.197 Angstrom` thin direction. A complete read-only scan then
found 1,380 of 749,866 rows outside the declared ordered-bulk numerical domain:
the minimum lattice width was below `0.5 Angstrom`, the metric condition number
exceeded `1e4`, or both. These rows are excluded at index construction rather
than patched during training; immutable source JSONL files are unchanged.

The audit also exposed a separate implementation defect. Reconstructing an
FP32 SPD metric solely to compare it with the same source lattice is
batch-dependent for anisotropic cells: 199 full-index batched round trips
crossed the `2e-5` internal check, including 127 rows inside the retained
physical domain. The clean physical entry point already derives its lattice
state from the supplied lattice. It now uses that source lattice directly for
periodic geometry and uses the derived chart only as a learned lattice context.
The generative lattice path is unchanged.

Stage-B-v1 must not resume. A separately hash-bound data-cleaned successor must
rebuild the MatPES index, train-only normalizer and aligned TensorNet feature
cache before another formal run.
