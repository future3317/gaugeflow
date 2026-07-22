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

## Mid-training evidence

| Metric | Stage-B | Stage-C 10k | Stage-C 20k | Stage-C 30k | Stage-C 40k |
|---|---:|---:|---:|---:|---:|
| Physical composite | `0.5929` | `0.3871` | `0.3254` | `0.2908` | `0.2652` |
| Energy RMSE | `0.1143` | `0.0955` | `0.0859` | `0.0801` | `0.0757` |
| Force RMSE | `0.3882` | `0.3318` | `0.3053` | `0.2898` | `0.2748` |
| Stress RMSE | `0.5733` | `0.4301` | `0.3888` | `0.3643` | `0.3469` |
| Teacher cosine | `0.8996` | `0.9171` | `0.9264` | `0.9323` | `0.9364` |
| NN-W1 | `0.5444` | `0.5533` | `0.5628` | `0.5656` | `0.5785` |
| Volume-W1 | `0.0722` | `0.0624` | `0.0676` | `0.0680` | `0.0711` |

Physical representation improves monotonically. Generative validity remains
perfect with zero failures, while the monotone NN-W1 increase is tracked as a
physical-transfer versus generative-retention trade-off. At 30k, all 512
samples still have exact composition, finite positive lattices, valid minimum
distance and zero failures; this remains true at 40k. The larger 30k-to-40k
NN-W1 increase (`+0.0128`) makes the trade-off material: selection will use the
physical/retention Pareto frontier, not the last optimizer step. The next full
diagnostic is the final Stage-C 50k checkpoint (global step 60,523).

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

Candidate selection first enforces hard closure, removes Pareto-dominated
checkpoints, and then minimizes maximum min--max-normalized regret over physical
composite, LeMat structure loss, NN-W1 and volume-W1. This is an operational
checkpoint choice declared after the 40k diagnostic, not a new learning Gate.
