# Stage-C LeMat geometry-boundary repair

Stage-C-v1 stopped reproducibly after its 20k checkpoint because the LeMat-v3
index admitted a source row whose nested geometry disagreed with `nsites`.
The model, optimizer and numerical losses were finite; this was a data-boundary
failure, not early stopping or model divergence.

An exact replay of the rank-0 balanced stream located `oqmd-2964825` at
checkpoint-relative batch 4,320. It declares eight sites but stores 15 species
and 15 Cartesian positions. A complete rebuild then inspected periodicity,
dimension flags, Cartesian shape/finiteness, species count and element symbols
for every index candidate. It found one additional malformed row,
`oqmd-2969647`, with the same positions-versus-`nsites` defect.

The v4 index excludes both records before training and stores their IDs,
locations, reasons and evidence hash. It is qualified with train/calibration/
test counts `4,563,028 / 252,475 / 253,247`; v3 had
`4,563,030 / 252,475 / 253,247`. Raw parquet files and the archived v3 index
remain unchanged. No runtime parser fallback was added.

The complete Stage-C-v1 20k checkpoint was migrated to v2 as follows:

- model, optimizer, EMA, MatPES/Alex cursors and every objective RNG state are
  byte-preserved;
- the rank-0 LeMat balanced stream is deterministically advanced to batch
  20,000 over the clean v4 support;
- unused LeMat stream instances remain at batch zero;
- the migration source hash and new data hashes are part of checkpoint
  metadata;
- future checkpoints are written every 5,000 Stage-C steps.

A three-GPU one-step resume smoke advanced global step 30,523 to 30,524 with
finite LeMat/MatPES/Alex losses (`1.72114 / 0.378884 / 1.32161`) and finite
gradient norm `2.03921`. Formal Stage-C-v2 then resumed from the migrated 20k
checkpoint on GPUs 1, 3 and 4.

Evidence files in this directory contain the exact horizon failure, v3/v4
manifests, malformed-row catalogue, migrated checkpoint sidecar and resume
smoke metric.
