# Full-Alex assignment capacity screen v2

Status: **PASS**.

The screen compared the unchanged geometry-aware remaining-count scorer at
hidden widths 256, 384, and 512 (3,885,568 / 8,531,712 / 14,980,096 parameters).
Every candidate saw the same 12,800 non-Gold fit structures for 100 updates and
the same fixed-order 1,024-structure validation panel.  Two reveal paths were
accumulated sequentially over 32-graph microbatches, so the mathematical
objective and 128-graph optimizer batch were unchanged from the failed v1
screen without materializing duplicate carriers.

The 256-width model achieved the best final validation NLL (`5.011470`), ahead
of width 384 (`5.652535`) and width 512 (`6.093660`).  It also used 4,835.66 MiB
and processed 328.46 graphs/s on physical GPU 3, an NVIDIA GeForce RTX 4090.
All candidates had finite updates and decreasing validation NLL, and Gold
assignment leakage was zero.  The preregistered smallest-within-1%-of-best rule
therefore selects **hidden width 256** for DDP software qualification and the
one-pass full-Alex masked-occupation pretraining run.

This result selects only assignment representation capacity.  It is not a
parent-conditioned assignment Gate and says nothing about the appropriate size
of the later joint GaugeFlow-base backbone.

