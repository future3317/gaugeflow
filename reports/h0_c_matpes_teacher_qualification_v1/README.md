# H0-C-v1 frozen failure

`h0_c_matpes_teacher_qualification_v1` is frozen as failed. This result is not
overwritten by later protocols and does not authorize H0-D or H1.

Both pinned teachers completed the deterministic 512-structure held-out sample
with zero inference failures, finite outputs, and energy/force/stress MAEs below
their frozen limits. TensorNet also passed the 32-structure translation,
proper-rotation, atom-relabeling and unimodular-cell audit. M3GNet failed all
four transformation classes and exceeded every rigid/cell invariance envelope.

The checkpoint is explicitly a `Potential v3` PyG artifact. MatGL 2.0.9 treats
M3GNet as a DGL model and cannot load the artifact without changing backend
semantics, so a runtime downgrade is not a valid repair. The intended MatGL
4.0.3 runtime remains frozen for this result.

The complete generated JSON and Parquet evidence remains in the versioned data
root; their hashes and the decision-critical metrics are frozen in
`result_manifest.json`. H0-C-v2 preserves the exact v1 sample and thresholds and
changes only the independent disagreement architecture.
