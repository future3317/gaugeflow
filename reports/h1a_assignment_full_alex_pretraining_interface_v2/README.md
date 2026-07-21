# Full-Alex assignment pretraining interface v2

Status: **FAIL (frozen)**. No model training started.

The exact dual-lattice replacement solved the scientific and numerical defect
found by v1:

- all 540,164 train graphs completed with zero candidate-cap failure;
- the 2,048-structure float64 sphere-decoder panel had maximum distance error
  `1.5249e-6`;
- all Gold and IID calibration/test material IDs were excluded;
- feature compilation reached `3533.59 graphs/s` and `172.60 MiB` on an RTX
  4090, with zero nonfinite output.

The protocol nevertheless failed its separately frozen
`refined_graphs <= 2000` check: 49,692 graphs contained at least one pair whose
dual integer box extended outside the initial 27 images. This is 9.20% of
graphs but only 2.29% of the 55,358,736 directed pairs. The worst exact box had
450 candidates and the measured throughput/memory checks passed by wide
margins.

The failed check mixed two non-equivalent quantities: v1 counted graphs not
certified by an isotropic shell-4 lower bound, whereas v2 counted every graph
that used any exact dual-box refinement. Refinement incidence is an internal
execution-path diagnostic, not a correctness or resource measure. The v2
result remains failed and unchanged. A successor protocol may retain the
observed incidence in its report while qualifying the interface only by exact
reference agreement, zero fail-closed events, leakage, finite outputs,
throughput and peak memory.
