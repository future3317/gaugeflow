# LeMat source-balanced stream bounded smoke

This is a CPU-only training-interface check, not Stage-C learning evidence and
not authorization for full LeMat continuation. It uses the previously audited
17,975-row bounded LeMat index and its 16,196-row train split.

The raw train subset is strongly imbalanced across functionals (`13,826` PBE,
`665` PBEsol and `1,705` SCAN). Over 1,000 deterministic global batches of 64,
the balanced stream produced fractions `0.32967 / 0.33353 / 0.33680`, with
maximum absolute deviation `0.00366` from the requested uniform mixture. Two
rank shards contained exactly 32,000 examples each, shared identical global
source cursors and reproduced the next batch exactly after checkpoint restore.

The stream traverses each source without replacement inside a source epoch and
wraps sources independently. Consequently, balancing changes exposure without
padding distributed batches or treating physical labels that are absent as
numeric zero targets.
