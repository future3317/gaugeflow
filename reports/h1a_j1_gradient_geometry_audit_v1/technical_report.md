# H1a J1 zero-step gradient-geometry audit

## Answer first

The audit finds **large gradient norms without persistent cross-regime
conflict**.  Median pre-clip scale is 0.2661, above the frozen severe-clipping
boundary 0.2.  All ten regime-pair median cosines are positive; no pair is
negative in 75% of the eight fixed microbatches.  The decision is therefore to
retain the current global clipping policy.  No blockwise clipping, AGC, target
RMS normalization or clip-threshold change is authorized.

The combined matched-attribution/gradient figure is stored once at
`../h1a_j1_matched_clock_attribution_v1/j1_matched_gradient_summary.{png,pdf}`.

## Protocol

The raw (not EMA) C2 training state at step 2,111 is evaluated on 64 fixed
validation structures as eight 8-structure microbatches.  For each batch, all
five regimes share structures, coordinate times and random draws.  Each
coordinate loss is backpropagated independently; no optimizer step is taken.
The audit records the full pre-clip gradient, norm, cosine, hypothetical clip
scale and an exhaustive parameter-module partition.

## Results

| regime | mean gradient norm | mean clip scale |
|---|---:|---:|
| clean--clean | 4.5965 | 0.2575 |
| noisy element | 5.5922 | 0.2295 |
| noisy lattice | 4.0415 | 0.2832 |
| diagonal | 4.2369 | 0.2619 |
| interior | 3.8036 | 0.3038 |

Across all 40 regime/microbatch gradients, clip scale has median 0.2661,
interquartile range `[0.2014, 0.3251]`, and range `[0.0728, 0.4905]`.
The smallest median cosine is clean--clean versus interior (0.2207); its
negative fraction is only 0.125.  Other median cosines range from 0.4018 to
0.7495.  Lattice-noisy and interior gradients are not larger than the other
regimes and do not oppose the backbone direction.

Mean gradient-energy shares are 69.48% coordinate readout, 18.22% input/time
embeddings, 11.52% dynamic edge/angular, 0.57% base message blocks, 0.19% time
fusion, and 0.014% inactive heads/atlas numerical residue.  The fusion clock is
small because it is a compact upstream transform, not because global clipping
leaves it a conflicting residual.

## Decision boundary

The observed 97.4% training clip frequency is real, but frequency alone does
not establish harmful clipping under AdamW.  This audit specifically rejects
the proposed conflict-based rationale for changing optimization.  It cannot
qualify free joint generation and does not authorize E1/L1/M1/J2, H1b--H6,
tensor/oracle work, relaxation, DFT or DFPT.
