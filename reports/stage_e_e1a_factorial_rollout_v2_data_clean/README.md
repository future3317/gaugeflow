# Stage-E1a factorial rollout v2 (data-support clean panel)

This is a separately identified diagnostic continuation of the immutable v1
factorial rollout.  It excludes the six validation records whose complete
composition is a single noble-gas token (all pure Kr); the exclusion rule and
indices are recorded in `data_quality_audit.md`.  No checkpoint, lattice,
coordinate or terminal sample was repaired.  The v1 result remains unchanged.

Protocol: `configs/gates/stage_e_e1a_factorial_rollout_v2_data_clean.json`
(SHA-256 `a1dcadadc72786acc7eb1bbf2312df59c5849c850ced3cf3ed1c640f7f18c9db`),
with 256 selected structures, 50 reverse-SDE steps, 384 orbit frames and
common seeds.  This is a localization diagnostic, not a qualification gate.

The archived full-run JSON was written by the pre-schema-fix evaluator and is
therefore v1-compatible at the top-level schema field even though its protocol
is v2.  The evaluator now derives the schema from the protocol; an 8-sample
v2 smoke rerun is archived as `schema_smoke_fixed.json` and reports
`gaugeflow.stage_e_e1a_result.v2`.  No metric from the full run was changed.

## Result

All 12 arm/role combinations completed with zero sampling failures and finite
positive lattice determinants.  The aggregate metrics are:

| arm | base orbit | E3-v2 orbit | conditioned - base | base volume W1 | E3 volume W1 |
|---|---:|---:|---:|---:|---:|
| `oracle_cal` | 1.18895 | 1.22523 | +0.03628 | 0 | 0 |
| `oracle_ca` | 352.49091 | 2641.72339 | +2289.23218 | 61.12725 | 64.31692 |
| `oracle_c` | 1.25292 | 1.26444 | +0.01152 | 0.05601 | 0.05772 |
| `free` | 1.26829 | 1.26776 | -0.00053 | 0.35987 | 0.34279 |

The clean panel removes the original pure-Kr support outliers, but the
`oracle_ca` failure persists and is no longer attributable to that data rule.
The first localized gap is therefore the generated-lattice side state (or its
50-step reverse exposure), not the coordinate-only path: `oracle_cal` remains
finite and stable, while `oracle_ca` develops extreme volume tails.  The
`oracle_c` and `free` arms do not provide useful tensor target separation, so
the E3 adapter still has no efficacy qualification.

## Decision boundary

This result does not authorize tensor-conditioned generation, F/RL,
relaxation, DFT/DFPT, or a free-crystal claim.  Before changing the model, the
next diagnostic should be a lattice-only NFE/exposure audit against the
qualified L1 100-step protocol, with per-structure tail reporting.  A lattice
fix must preserve the P1 chart, exact composition context and Stage-C null
branch; terminal volume clipping would be an invalid repair.
