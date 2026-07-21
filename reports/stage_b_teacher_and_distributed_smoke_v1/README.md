# Stage-B teacher-cache and two-GPU smoke

This bounded preflight used the qualified 34.28M A1 EMA backbone and the pinned
MatPES-PBE TensorNet teacher. It is software evidence, not a trained Stage-B
result.

The teacher contract is type-matched and per atom: TensorNet
`feature_dict['readout']` has shape `[N,128]`. The original untrained placeholder
projected a graph mean, which would have erased the local environments that the
physical transfer is intended to learn. The production target, mask, head and
loss now remain node-resolved; the loss is a graph-equal mean of per-node cosine
distance. A 256-row real cache contains 1,723 atoms, is finite, and opens through
the memory-mapped reader in about 4.1 ms. A real four-graph backward pass covers
all 53 atoms and has nonzero finite gradients.

The two-GPU smoke uses rank-sharded MatPES/Alex batches, exact local/global loss
fractions, globally summed gradients, one rank-0 AdamW/EMA owner, and parameter
broadcast after each update. Both ranks finish with the same full-state digest.
Interrupted/repeated execution reproduces metrics, model, AdamW and EMA state
exactly. The first resume attempt correctly failed because portable CPU EMA
shadows were not migrated back to the live CUDA model; the general EMA loader
was repaired rather than adding a Stage-B-specific fallback.

The next software step is the production data-cursor runner and full PBE cache.
Formal physical training remains blocked until those artifacts and the final
implementation commit are inserted into the frozen protocol.
