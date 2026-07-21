# Assignment two-GPU software qualification v1

Status: **PASS**.

The selected 3,885,568-parameter assignment scorer was exercised on 12,799
real, non-Gold Alex-MP-20 training structures with an intentionally uneven
final batch.  Global reveal orders were sampled once and sharded without
padding, and graph/reveal-path gradients were accumulated before one DDP
synchronization.

The one-GPU and two-GPU gradients have cosine `0.9999999999999571`; their AdamW
updates have cosine `0.9999999999593552` and relative L2 error `9.43e-6`.  Four
continuous updates and a serialized `2 + resume + 2` trajectory agree exactly
(`0.0` maximum parameter error).  The 100-update real-data smoke covers all
12,799 structures exactly once, reduces NLL by a factor of `0.49725`, and has
zero Gold leakage and finite gradients throughout.

On two NVIDIA GeForce RTX 4090 cards (physical devices 1 and 2), the run
processed `312.40 graphs/s` with `4,392.31 MiB` peak allocated memory per rank.
All frozen checks passed.  This authorizes one exact full-Alex masked-occupation
pretraining pass; it does not qualify the parent-conditioned assignment law.

