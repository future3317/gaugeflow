# LeMat training index v3

The production LeMat index contains 5,068,752 immutable parquet references
after benchmark-overlap removal, the `N<=20` restriction, and a
target-independent lattice-domain filter. The train/calibration/test counts are
4,563,030 / 252,475 / 253,247. Exactly two train rows were removed for a metric
width below 0.5 Angstrom; one of those also exceeded metric condition 10,000.
No nonpositive-volume row was present.

The index remains source balanced at sampling time rather than duplicating rare
functionals on disk. Raw LeMat parquet artifacts are unchanged. The manifest
binds every source hash and the compact 55.8 MB index tensor.

