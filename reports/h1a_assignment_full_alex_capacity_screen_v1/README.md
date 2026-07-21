# Full-Alex assignment capacity screen v1

Status: **FAIL (frozen)**.

The screen was launched once from implementation commit `8b31ae5df0cca173b1330063010d020672f87333`
on physical GPU 3, an NVIDIA GeForce RTX 4090.  The first width (`hidden_dim=256`,
3,885,568 parameters) exhausted CUDA memory during the first backward pass before
an optimizer update.  The implementation materialized two reveal paths by
duplicating the complete 128-graph carrier, reaching 21.41 GiB process memory and
then failing a 3.41 GiB allocation.

This is an execution-path failure, not evidence about representation capacity.
The thresholds, data panel, and candidate widths were not interpreted.  The
successor protocol must preserve the same 256/384/512 capacity candidates and
selection rule while replacing carrier duplication with sequential path and
graph-microbatch gradient accumulation.

The archived runtime log has SHA-256
`513c07e5f373374033ece0aa8de58749ef7cd71f57c3651401b7afb013b9387a`.

