# Stage-C checkpoint selection

Stage-C-v2 completed all 50,000 declared continuation updates at global step
60,523. The terminal checkpoint is finite, has SHA-256
`ab884d80a3b46e384c6169f0af6b99bc15d9b34c915267e141f6962655fe6986`,
and the three-rank training process exited without traceback, non-finite loss,
CUDA, NCCL, or out-of-memory errors.

The operational checkpoint was selected by the protocol frozen after the 40k
diagnostic and before the 50k result. All candidates use the same EMA, paired
LeMat rows/noise, complete MatPES calibration split, and unchanged 512-sample
A1-v1.1 retention panel.

| Stage-C step | LeMat macro loss | Physical composite | NN-W1 | Volume-W1 |
| ---: | ---: | ---: | ---: | ---: |
| 20k | 1.571414 | 0.325350 | **0.562817** | 0.067634 |
| 30k | 1.531680 | 0.290835 | 0.565613 | 0.068019 |
| 40k | 1.503874 | 0.265180 | 0.578456 | 0.071122 |
| 50k | **1.486348** | **0.250460** | 0.572337 | **0.067552** |

Every candidate is hard-eligible: exact composition, finite positive lattice,
minimum-distance validity and formula uniqueness are one; terminal masks and
sampling failures are zero. The 40k candidate is Pareto dominated. Among the
20k, 30k, and 50k frontier, min--max-normalized maximum regrets are 1.0,
0.539119, and 0.608750 respectively. The declared Pareto-minimax rule therefore
selects **Stage-C 30k**, global step 40,523, checkpoint SHA-256
`8807877bbdcc61090a431dc5cd146ed62bf545b2a65425ff8bb16c8d0d317bf9`.

The selection is not an early-stop reinterpretation: 50k provides the best
LeMat, physical, and volume objectives, while 30k is the lowest worst-case
regret across physical transfer and local-geometry retention. It remains a
tensor-free GaugeFlow-base checkpoint. It does not qualify tensor conditioning,
RL, relaxation, DFT, or DFPT.

Machine-readable evidence:

- `candidate_20000.json` through `candidate_50000.json`: complete three-panel
  evaluations;
- `selection.json`: eligibility, Pareto frontier, normalized regrets, and the
  selected checkpoint;
- `configs/gates/stage_c_checkpoint_selection_v1.json`: immutable selection
  contract.
