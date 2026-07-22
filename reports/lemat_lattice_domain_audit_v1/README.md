# LeMat lattice-domain audit v1

This read-only audit evaluated every one of the 5,068,754 rows selected by the
qualified LeMat v2 index under the same target-independent bulk-domain bounds
used for MatPES: positive volume, minimum metric width at least 0.5 Angstrom,
and metric condition at most 10,000.

Two train rows failed. One had width 0.35095 Angstrom and condition 10,877.69;
the other had width 0.49230 Angstrom. No calibration or test row failed, and
the audit found no nonpositive-volume row. The raw parquet files remain
immutable. LeMat index schema v2 therefore excludes
these two rows at the processed-data boundary and records every reason count.
There is no runtime fallback or relaxed numerical tolerance.

The rebuilt 5,068,752-row v3 index then passed the same exhaustive audit with
zero failures. Its observed minimum width is 0.50969 Angstrom and maximum
metric condition is 7,706.28.

This result qualifies only the data-domain correction needed before a bounded
Stage-C execution smoke. It does not authorize LeMat continued pretraining
before Stage-B-v1.1 passes its frozen physical and generation-retention Gate.
