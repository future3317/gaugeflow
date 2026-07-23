# Data-support audit for the clean factorial panel

The original Stage-D validation cache contains 398 structures.  A scan of the
complete element composition found six pure noble-gas rows, all krypton
(dense token 35, atomic number 36), at validation-local indices
`98, 103, 104, 105, 379, 390`.  Their atom counts are `2, 2, 9, 4, 2, 4`.
They carry physical-zero/missing piezo response labels and are outside the
support needed to test a tensor-conditioned solid-crystal interface.

The v2 diagnostic excludes exactly these rows before the seeded panel draw.
It does not clip volume, replace a lattice, repair coordinates, or alter any
checkpoint.  The original v1 result remains archived unchanged; v2 is a
separately identified data-support diagnostic and cannot be compared as if it
were the same validation panel.
