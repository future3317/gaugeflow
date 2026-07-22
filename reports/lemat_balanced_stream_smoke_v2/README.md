# LeMat full-index block-local stream smoke

This is a CPU-only Stage-C data-plane check, not training evidence and not
authorization for full LeMat continuation. It uses the qualified 5,068,754-row
LeMat v2 index and its 4,563,032-row train split.

The sampler preserves the requested uniform PBE/PBEsol/SCAN mixture while
randomizing parquet row-group blocks and then rows within each block. Across
1,000 deterministic global batches of 64, the largest source-balance error was
0.00366146. A batch touched 3.081 parquet blocks on average and at most 5,
instead of scattering individual examples across dozens of blocks. Both rank
shards emitted exactly 32,000 examples and checkpoint restoration reproduced
the next batch exactly.

Block partition identity is hashed into the stream checkpoint. Changing the
parquet block assignment therefore fails restoration instead of silently
changing the continued-pretraining sample order.
