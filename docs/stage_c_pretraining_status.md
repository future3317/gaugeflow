# Stage-C continued-pretraining status

Updated: 2026-07-22. This is the active status record for post-A1 structural
scale-up; historical mechanism screens remain in `research_iteration_history.md`.

## Purpose

Stage-C continues the qualified 34.28M GaugeFlow-base from the Stage-B-v1.1
checkpoint. It expands structural coverage without discarding either the
physical representation learned from MatPES or the Alex-MP-20 generative
substrate. The fixed objective is

\[
0.4\mathcal L_{\rm LeMat\ structure}
+0.3\mathcal L_{\rm MatPES\ physical}
+0.3\mathcal L_{\rm Alex\ structure}.
\]

LeMat contributes geometry-only product-space denoising. MatPES is the sole
energy/force/stress/teacher-feature stream. Alex is replayed to preserve the
declared benchmark distribution. No tensor condition, RL, relaxation, DFT, or
DFPT objective is active.

## Incoming evidence

Stage-B-v1.1 is complete:

| Metric | Result |
|---|---:|
| Physical composite calibration loss | `19.6127 -> 0.5929` |
| PBE teacher-feature cosine | `0.8996` |
| A1 retention: exact composition | `1.0` |
| A1 retention: sampling failures | `0` |
| A1 retention: NN-W1 / volume-W1 | `0.5444 / 0.0722` |

This qualifies physical-representation transfer and bounded A1 retention. It
does not imply tensor conditioning, stability, relaxation retention, or
materials-discovery capability.

## Active run

- Protocol: `configs/gates/stage_c_lemat_continued_pretraining_v1.json`
- Seed: `5705`
- Optimizer updates: `50,000`
- Three fixed roles: LeMat structure / MatPES physical / Alex structure
- Global batch per role: `64`
- Checkpoints: every `10,000` updates
- Devices at launch: RTX 4090 GPUs `1,3,4`
- Run directory: `/home/workspace/lrh/DATA/T2C-Flow/runs/stage_c_lemat_continued_pretraining_v1`

The run presents 3.2M examples from each stream. Because LeMat is sampled with
equal PBE/PBEsol/SCAN source weight, this is an expected-exposure budget rather
than one raw, without-replacement LeMat epoch.

## Execution contract

One rank owns each role and its full model replica. After local backward passes,
the three weighted gradients are bucketed, summed once, and every rank applies
the same AdamW and EMA update. LeMat rows are materialized source-locally and
prefetched to a dedicated CUDA stream. Prefetch never crosses a checkpoint
boundary; a four-step interrupted-resume comparison found zero mismatches over
2,245 tensor leaves and 696 scalar leaves.

## Pending evaluation

No Stage-C learning result is reported until a declared checkpoint is evaluated
on all three panels:

1. LeMat held-out geometry denoising;
2. MatPES held-out normalized energy, force, Kelvin stress, force cosine, and
   PBE node-feature cosine by functional;
3. the unchanged A1-v1.1 512-reference/512-free-sample retention panel,
   including validity, exact composition, positive lattice, and periodic
   distance checks.
