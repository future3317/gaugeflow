# Explicit composition-state data audit

The qualified H1a cache supports a sparse exact composition representation.
The train split contains 540,164 graphs and 5,161,621 atoms, with mean/max atom
count 9.5557/20 and 76 active elements.  Its number of distinct species is:

| species per graph | train graphs |
|---:|---:|
| 1 | 678 |
| 2 | 33,094 |
| 3 | 276,663 |
| 4 | 228,329 |
| 5 | 1,340 |
| 6 | 59 |
| 7 | 1 |

Thus 0.997408 of training graphs contain at most four species and the maximum
is seven, satisfying the frozen 0.95 / 10 bounds.  Validation and test maxima
are both six.  An exact state serialized as increasing atomic-number tokens
and positive counts therefore needs at most seven short species--count pairs,
not a combinatorial enumeration of all 118-dimensional histograms.

This is a data-representation qualification only.  It does not qualify a
composition generator, restart E1 training, or authorize L1/M1, tensor work,
relaxation, DFT, or DFPT.  The next permissible action is an exact synthetic
normalization and sampling-closure test for the proposed sparse composition
kernel.
