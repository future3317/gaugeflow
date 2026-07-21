# LeMat training-interface bounded smoke

This CPU-only smoke validates the data contract needed for later LeMat
continued pretraining. It deliberately reads only one row group from each of
the 19 immutable parquet shards and is not a Stage-C qualification or an
authorization to train on LeMat.

The active parser treats the lattice as a row basis, converts Cartesian site
positions to wrapped fractional coordinates, divides the uncorrected total
VASP energy by the site count, and converts the symmetrized full stress from
compressive-positive kbar to tensile-positive Kelvin GPa. Missing or malformed
force arrays disable only the force loss. Rows with
`cross_compatibility=false` remain valid structure-prior examples but their
energy, force, and stress targets are masked under the default
`compatible_only` policy.

Splits are grouped by `entalpic_fingerprint`, not by individual parquet row,
so cross-functional structural duplicates cannot cross the IID boundary. The
known `alex<agm...>` wrapper is normalized only for explicit Alex validation
and test exclusion; IDs and fingerprints remain audit metadata and do not
enter model batches.

The bounded index contains 17,975 structures with at most 20 sites and zero
invalid index rows. Random access was exercised for PBE, PBEsol, and SCAN in
all three splits. Full indexing, Alex benchmark-overlap exclusion, a
row-group-aware distributed sampler, and Stage-C training remain pending.
