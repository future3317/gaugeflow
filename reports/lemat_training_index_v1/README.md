# Full LeMat continued-pretraining index v1

The full `N<=20` LeMat index was built on the server from 19 immutable parquet
shards after excluding the qualified Alex validation/test ID union. It contains
5,068,754 structures and 4,878,239 fingerprint/ID split groups, with
4,563,032 / 252,475 / 253,247 train/calibration/test rows and zero invalid
index rows. The index tensor SHA-256 is recorded in `result.json`.

Of 135,040 normalized Alex benchmark IDs, 129,152 occurred among otherwise
eligible LeMat records and were excluded. The remaining IDs were absent or
outside the `N<=20` scope. Both the exclusion file SHA-256 and canonicalized
content SHA-256 are bound into the qualified index manifest.

The full train split is highly imbalanced (`4,222,763` PBE, `9,014` PBEsol,
`331,255` SCAN). The deterministic balanced rank stream was therefore checked
again on this full index. PBE/PBEsol/SCAN random access passed, two ranks
received equal counts, and checkpoint resume was exact. This remains a data
qualification artifact rather than Stage-C training evidence.
