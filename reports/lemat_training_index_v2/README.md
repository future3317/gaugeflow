# LeMat native-fingerprint overlap expansion v2

The v2 builder first collects every LeMat `entalpic_fingerprint` associated
with the hash-bound Alex validation/test ID union, then excludes all eligible
rows sharing those fingerprints even when their immutable IDs differ.

It found 129,302 benchmark-associated fingerprints across the full inventory.
Among eligible `N<=20` rows, all 129,152 exclusions were direct ID matches and
zero additional rows were cross-ID fingerprint matches. The selected index
tensor is therefore byte-identical to v1 (`6ab73b...0fc1`). This closes the
provider-native structural-fingerprint envelope without a quadratic matcher.

This result does not assert that the provider fingerprint is a mathematical
complete invariant of periodic structures. A broader independent
StructureMatcher stress panel can still be reported, but there are no native
fingerprint collision candidates requiring adjudication before software
integration.
