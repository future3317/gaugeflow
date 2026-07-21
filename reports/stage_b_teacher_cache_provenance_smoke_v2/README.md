# Stage-B teacher-cache provenance smoke v2

The resumable cache builder previously checked an existing part only by its
row interval. That was sufficient for the fresh 256-row smoke but unsafe for a
long interrupted build: an old part from another teacher or feature contract
could have been reused silently.

Shard schema v2 binds every part to the Stage-B protocol, MatPES index,
qualified teacher report, teacher checkpoint manifest, hashes of the actual
TensorNet model files, functional, feature dimension, and graph/node batch
contract. A one-row real PBE extraction passed on CPU. Reusing its work part
under the identical contract reproduced the final feature hash. Changing only
`graphs_per_batch` caused a fail-closed provenance error before an output cache
was written.

This is software evidence only. The complete PBE cache remains pending and no
physical model weights were trained.
