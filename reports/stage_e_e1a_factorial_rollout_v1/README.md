# Stage-E1a factorial rollout v1

This is the frozen 256-structure diagnostic run using the Stage-C checkpoint,
the repaired E3-v2 adapter, and the independent Stage-D response evaluator.
It is a localization experiment, not a qualification gate.

Protocol: `configs/gates/stage_e_e1a_factorial_rollout_v1.json` (SHA-256
`6490635e734209ec3e56a21867b0c4fd00232cf14f6d54c70415ca0722cefb8c`), with
50 reverse-SDE steps, 384 atlas frames, common seeds, and four progressively
less constrained arms (`oracle_cal`, `oracle_ca`, `oracle_c`, `free`).

## Result

All 12 roles completed with zero sampling failures and finite positive lattice
outputs. The arm summaries are:

| arm | base orbit error | E3-v2 orbit error | conditioned - base | base volume W1 | E3 volume W1 |
|---|---:|---:|---:|---:|---:|
| `oracle_cal` | 1.25658 | 1.28583 | +0.02924 | 0 | 0 |
| `oracle_ca` | 1747.62256 | 2.29049 | -1745.33191 | 3.487e26 | 3.498e26 |
| `oracle_c` | 1.21007 | 1.22972 | +0.01966 | 26.3881 | 30.3458 |
| `free` | 1.22917 | 1.22946 | +0.00029 | 0.33465 | 0.32005 |

The `oracle_cal` arm is numerically stable, so the coordinate-only path is not
the first failure. The `oracle_ca` lattice-only arm contains the first clear
failure: four validation records produce extreme volumes, while the
coordinate-only and free arms remain finite. The extreme records are all
single-token Kr compositions (token 35, atomic number 36), a data-support
outlier in the piezoelectric response cache. The original v1 panel therefore
cannot be interpreted as a pure learned-interface failure.

`oracle_c` and `free` show no useful target separation for E3-v2. Thus this
run establishes an interface/data diagnostic, not tensor-conditioned efficacy.
It does not authorize F/RL, relaxation, DFT/DFPT, or a free-generation claim.
The six pure noble-gas validation records are handled only by the separately
versioned v2 data-clean diagnostic; v1 remains immutable.
