# Alex benchmark exclusion v1

This manifest freezes the audit-only ID exclusion used when constructing the
full LeMat continued-pretraining index. The qualified packed Alex validation
and test indices each contain 67,520 rows. Their normalized union contains
135,040 unique IDs, so the two benchmark splits have no material-ID overlap.

The full ID list is a processed-data artifact on the server rather than a
repository payload. Its SHA-256 is recorded in `manifest.json`. IDs are used
only while building the LeMat index and are never added to a model batch.
