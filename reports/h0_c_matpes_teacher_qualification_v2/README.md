# H0-C-v2 qualification

`h0_c_matpes_teacher_qualification_v2` passed every frozen check. H0-C is
qualified, but H1 remains unauthorized because H0-D and H0-E are still open.

The protocol preserves the exact v1 512-structure held-out sample,
32-structure invariance sample, runtime, primary TensorNet checkpoint and all
thresholds. It changes only the independent teacher from the failed M3GNet
artifact to QET trained on the same MatPES-PBE-2025.2 dataset. The superficially
named CHGNet 2025.2.10 candidate was rejected before inference because its
pinned model card declares MatPES-PBE-2024.11 rather than the matched dataset.

TensorNet and QET completed all predictions without failure or non-finite
values. Energy, force and stress MAEs passed, disagreement remained nonzero,
and translation, proper rotation, reverse atom relabeling and unimodular cell
basis changes stayed within the frozen FP32 envelopes. Runtime identity,
checkpoint protocol/config binding and immediate file rehashes also passed.

These models are not independent DFT validation and must never be used as
reverse-sampling guidance. They are authorized only for frozen offline labels,
uncertainty filtering, representation extraction and auxiliary PES losses.
