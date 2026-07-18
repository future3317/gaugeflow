# H1a symmetric-endpoint reverse-kernel audit v1

Status: **completed; analytic symmetry closure passed, no kernel selected**.

Generic synthetic and 64 real six-site endpoints close under every analytic
candidate.  This no-training audit returns to the historical InN/BN four-site
panel, whose coordinates and lattices are pinned in the protocol and traced to
the SHA-256-fixed processed JARVIS source.

The primary endpoint metric minimizes only over permutations that preserve
atomic species.  Fixed-CIF row RMS remains diagnostic, while minimization over
all 24 permutations is reported solely to identify species-invalid branch
changes.  This follows the A11 conclusion that arbitrary CIF row labels are not
physical, without silently quotienting an In site with an N site.

All methods, thresholds, trajectories and random streams were frozen before
the first trajectory result.  No model is trained or changed.

## Result and corrected attribution

For both InN and BN, both terminal initializations, and all four score-only
integrators, the 200-step fixed-CIF and type-preserving quotient recovery are
`1.0`; species-invalid and fully unrecovered branch fractions are `0`.  The
endpoint-aware wrapped reference also passes.  Results are stable from 100 to
200 steps and the final translation posterior has a large, finite modal
margin.

Together with the generic and real-endpoint audits, this falsifies the earlier
informal claim that the exact quotient sampler intrinsically loses roughly
20% of four-site trajectories at a torus cut locus.  The archived RMS values
near `0.433/0.612` are consistent with measuring representatives before
removing the shared translation gauge.  They are not evidence for a physical
endpoint failure.

No analytic candidate can be preferred by endpoint closure: all pass.  The
remaining H1a failure must be tested with the learned seed-5601 score on
rollout states.  Production sampling remains unchanged pending that separate
diagnostic.
