# Stage-C continued-pretraining status

Updated: 2026-07-23. This is the completed status record for post-A1 structural
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

## Completed run

- Protocol: `configs/gates/stage_c_lemat_continued_pretraining_v2.json`
- Seed: `5705`
- Optimizer updates: `50,000`
- Three fixed roles: LeMat structure / MatPES physical / Alex structure
- Global batch per role: `64`
- Checkpoints: every `5,000` updates
- Devices at launch: RTX 4090 GPUs `1,3,4`
- Run directory: `/home/workspace/lrh/DATA/T2C-Flow/runs/stage_c_lemat_continued_pretraining_v2`

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

Stage-C-v1 stopped after a source row declared eight sites but stored 15
positions and species. A complete geometry-boundary rebuild found and removed
two malformed OQMD records from LeMat-v4. The 20k checkpoint was migrated
without changing model, optimizer, EMA, MatPES/Alex cursors or objective RNG
states; only the LeMat stream was deterministically re-based on the clean
support. A three-GPU one-step resume smoke passed before v2 continued.

## Complete trajectory and selection

| Metric | Stage-B | Stage-C 10k | Stage-C 20k | Stage-C 30k | Stage-C 40k | Stage-C 50k |
|---|---:|---:|---:|---:|---:|---:|
| Physical composite | `0.5929` | `0.3871` | `0.3254` | `0.2908` | `0.2652` | `0.2505` |
| Teacher cosine | `0.8996` | `0.9171` | `0.9264` | `0.9323` | `0.9364` | `0.9394` |
| LeMat macro loss | -- | -- | `1.5714` | `1.5317` | `1.5039` | `1.4863` |
| NN-W1 | `0.5444` | `0.5533` | `0.5628` | `0.5656` | `0.5785` | `0.5723` |
| Volume-W1 | `0.0722` | `0.0624` | `0.0676` | `0.0680` | `0.0711` | `0.0676` |

Physical representation and LeMat denoising improve through 50k. Generative
hard validity remains perfect with zero failures. NN-W1 does not improve
monotonically: it worsens through 40k and partially recovers at 50k. This is a
physical-transfer versus local-geometry-retention trade-off rather than
sampler collapse.

## Evaluation contract

The archived 10k--40k mid-training evaluator covered the complete MatPES
calibration split and unchanged A1-v1.1 512-reference/512-free-sample retention
panel. Those two panels remain valid, but the script had not implemented the
LeMat held-out structure panel declared by the Stage-C plan. They are therefore
not relabelled as complete three-panel evaluations.

The final selection protocol `stage_c_checkpoint_selection_v1` closes that
interface before the 50k result. Every v2 candidate is evaluated on:

1. a fixed functional-balanced LeMat-v4 calibration panel: all 500 PBEsol
   calibration rows plus 500 target-independent matched PBE and SCAN rows,
   with paired rows and noise across checkpoints;
2. complete MatPES calibration: normalized energy, force, Kelvin stress, force
   cosine, and PBE node-feature cosine by functional;
3. the unchanged A1-v1.1 512-reference/512-free-sample retention panel,
   including validity, exact composition, positive lattice, and periodic
   distance checks.

Candidate selection first enforced hard closure, removed Pareto-dominated
checkpoints, and then minimizes maximum min--max-normalized regret over physical
composite, LeMat structure loss, NN-W1 and volume-W1. This is an operational
checkpoint choice declared after the 40k diagnostic, not a new learning Gate.

All four candidates are eligible and 40k is Pareto dominated. The frontier is
20k/30k/50k. Their maximum normalized regrets are `1.0 / 0.539119 / 0.608750`,
so the declared rule selects **Stage-C 30k** (global step 40,523; checkpoint
SHA-256 `8807877bbdcc61090a431dc5cd146ed62bf545b2a65425ff8bb16c8d0d317bf9`).
The 50k checkpoint remains the completed trajectory endpoint and supplies the
best LeMat/physical/volume objectives, but it is not the operational base.
Complete evidence is archived in `reports/stage_c_checkpoint_selection_v1/`.
