# GaugeFlow v1 Freeze Boundary

Date: 2026-07-24

## Freeze Decision

GaugeFlow v1 is frozen as a partially qualified tensor-free generator plus a
blocked Stage-E tensor-conditioning stack.  The frozen v1 evidence must remain
available as historical baseline and negative-control material.  A-v2 work must
not overwrite these claims, checkpoints, random streams, panels or reports.

This freeze does not mean the whole project failed.  It means the current v1
route has reached a clear boundary: local Stage-E interface bugs were repaired,
but the generated-state coverage problem remains.

## Active Code Boundary

Active branch:

```text
codex/assignment-closure
```

Latest documentation/evidence commit at freeze time:

```text
0bfcd77e docs: record Stage-E orderless partial exposure diagnostic
```

Key Stage-E repair commits:

```text
838364d9 fix: preserve composition context in Stage-E lattice exposure
7b8f222d fix: scale Stage-E lattice shape residual
50a1c93d fix: expose partial assignment state in Stage-E lattice adapter training
b747613c fix: retain clean lattice exposure carrier in Stage-E adapter training
```

## Frozen Checkpoints

```text
Stage-C base:
  /home/workspace/lrh/DATA/T2C-Flow/runs/stage_c_lemat_continued_pretraining_v2/
    checkpoint_step_00040523.pt
  SHA-256:
    8807877bbdcc61090a431dc5cd146ed62bf545b2a65425ff8bb16c8d0d317bf9

Stage-D independent response evaluator:
  /home/workspace/lrh/DATA/T2C-Flow/runs/stage_d_response_training_v1/
    best_checkpoint.pt

Stage-E E3 tensor adapter:
  /home/workspace/lrh/DATA/T2C-Flow/runs/stage_e_e3_adapter_trust_region_30k_v1/
    checkpoint.pt

Composition law p(C|N):
  /home/workspace/lrh/DATA/T2C-Flow/runs/h1a_e1_absolute_likelihood_v1/
    seed_5705/step_001900.pt
```

Stage-E lattice exposure adapters:

```text
Historical old adapter, negative control:
  /home/workspace/lrh/DATA/T2C-Flow/runs/
    stage_e_lattice_generated_exposure_jarvis_v1/adapter.pt

Counts-fixed clean-exposure adapter:
  /home/workspace/lrh/DATA/T2C-Flow/runs/
    stage_e_lattice_generated_exposure_jarvis_countsfix_v1/adapter.pt

Orderless-partial diagnostic adapters, not production candidates:
  /home/workspace/lrh/DATA/T2C-Flow/runs/
    stage_e_lattice_generated_exposure_jarvis_orderless_partial_v1/adapter.pt
    stage_e_lattice_generated_exposure_jarvis_orderless_partial_v2/adapter.pt
```

## What v1 Qualifies

- Tensor-free GaugeFlow-base A1-v1.1 software/runtime and free-generation panel
  under its own protocol.
- Explicit `p(C|N)` absolute-likelihood composition law.
- Supported-IID exact-count assignment under its own IID carrier contract.
- Stage-C 30k/global 40523 as the operational tensor-free base.
- Stage-D as an independent response evaluator.
- Stage-E E0/E3 teacher-forced tensor sensitivity and software interfaces.

These claims do not imply tensor-conditioned generation has passed.

## Stage-E Causal Findings

1. Missing `composition_counts` was a real interface bug.  It explains the old
   lattice exposure adapter's volume drift.
2. Full lattice shape residual was too strong.  `shape_scale=0.25` repaired the
   counts-fixed adapter's `oracle_ca` volume/NN decoupling in smoke32.
3. The lattice head does not directly read coordinate message blocks.  In the
   full hybrid forward, the lattice readout consumes the composition/lattice
   graph context, not the coordinate edge encoder output.
4. `oracle_c` still fails because the joint sampler asks the lattice head to
   operate on partial/MASK orderless assignment states, while the clean
   exposure adapter was trained on full clean element tokens.
5. Training the same adapter on orderless partial/MASK states improves
   `oracle_c` only slightly and damages `oracle_ca` volume retention.  The
   single shared adapter cannot be promoted.

## Current Smoke32 Comparison

Conditioned role only:

| adapter | shape scale | oracle_ca volume-W1 | oracle_ca NN-W1 | oracle_c volume-W1 | oracle_c NN-W1 | free NN-W1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| counts-fixed clean exposure | 0.25 | 0.079249 | 0.528087 | 0.075922 | 0.701007 | 0.329158 |
| orderless partial v2 | 0.25 | 0.157782 | 0.541020 | 0.086297 | 0.667458 | 0.319875 |
| orderless partial v2 | 0.00 | 0.149005 | 0.639191 | 0.086886 | 0.673520 | 0.321570 |

Target 123 remains pathological in `oracle_c`:

```text
counts-fixed clean exposure, scale 0.25:
  condition = 12.483171

orderless partial v2, scale 0.25:
  condition = 12.530057

orderless partial v2, scale 0.00:
  condition = 11.041600
```

## Negative Claims

v1 does not qualify:

- Stage-E tensor-conditioned generation;
- `oracle_c` or `free` generated-side tensor conditioning;
- Stage-F reward post-training;
- relaxation, DFT or DFPT validation;
- discovery of new piezoelectric materials.

The partial/MASK exposure adapter is a diagnostic artifact.  It must not replace
the counts-fixed clean-exposure adapter in reports except as an explicitly
labelled negative-control comparison.

## Next Allowed Direction

Follow the A-v2 plan only after this freeze boundary is cited:

1. Write the A-v2 generated-state data contract.
2. Validate the on-policy/replay cache provenance on a small 34M carrier.
3. Confirm exact composition, assignment provenance, lattice shape/volume
   telemetry, coordinate validity and generated-state stratification.
4. Only then run capacity competition or larger multi-GPU training.

Do not launch Stage-F, do not rerun A/B/C v1 to overwrite history, and do not
claim Stage-E pass from any scalar adapter dose.
