# E1 species-law and co-occurrence attribution

## Outcome

This zero-training audit does **not** pass E1 and does not authorize site
assignment.  It shows that the failed `0.775688 <= 0.75` check is an indirect
and poorly isolated measure of the learned species law, rather than evidence
that the factorized composition model collapsed.

The source screen divided the complete composition NLL by its random-
initialization value.  That complete NLL contains a fixed train-only integer-
partition prior whose mean calibration contribution is `1.617298` nats per
graph at both step 0 and step 2005.  Only the species term is optimized.  For
that term alone the final/initial ratio is `0.750885`, missing `0.75` by only
`0.000885`; including the unchanged support term moves the reported ratio to
`0.775688`.

The random denominator is also not a neutral reference.  At initialization,
the network is `0.178444` nats per decision worse than an exactly uniform law
over the history-dependent legal categorical support.  The final model is
`0.901800` nats per decision better than that uniform law.  A threshold on the
ratio to one random draw therefore mixes learning with arbitrary initial-logit
scale and an untrained additive likelihood term.

## Frozen protocol

The audit reproduces the exact seed-6605 train-internal 95/5 split and reads
the frozen seed-5705 step-0 and step-2005 checkpoints.  It takes zero optimizer
steps.  For every calibration composition it holds the true node count and
integer partition fixed, then audits only

```text
p_theta(distinct elements | integer partition, node count).
```

The comparison baseline is fit-split element frequency conditioned on node
count, count-rank slot, and current stoichiometric count.  It is renormalized
over the same exact history-dependent distinct-element and equal-count-tie
support as the neural model.  This is a deliberately weaker baseline than the
model because it does not read the full partition or recurrent element
history.

## Conditional likelihood

| metric | result |
|---|---:|
| calibration graphs | `27,009` |
| active element decisions | `91,008` |
| mean species per graph | `3.36954` |
| initial species NLL / decision | `4.33633` |
| final species NLL / decision | `3.25609` |
| species-only final/initial ratio | `0.750885` |
| legal-uniform NLL / decision | `4.15789` |
| count-slot empirical NLL / decision | `3.63994` |
| final minus empirical | `-0.383854` |

The neural law is substantially better than the lower-context empirical
baseline and captures `1.74x` its uniform-to-baseline headroom.  This does not
mean the model exceeds the unknown Bayes optimum: it means the full partition
and recurrent history contain useful information omitted from that baseline.

The group table still identifies small rare-stratum weaknesses.  Single-
species compositions have only 42 calibration decisions and end at `4.6880`
nats per decision; count 16 has 11 decisions and remains `0.9575` nats above
the empirical baseline.  Element-frequency Q1 has a final/initial ratio of
`0.91584`.  These strata are too small to explain the population-level miss,
but they must remain visible in any future Gate.

## Fixed-partition co-occurrence

The new `sample_species_given_partition` API is the same exact conditional
kernel used by free composition sampling; it does not add a target-
composition runtime input.  Its numerical test checks that the selected
partition is preserved and that sampled and teacher-forced conditional log
probabilities close.

| metric | step 0 | step 2005 |
|---|---:|---:|
| element-count JSD | `0.07422` | `0.001281` |
| element-presence JSD | `0.06012` | `0.001105` |
| pair-distribution JSD | `0.15772` | `0.010461` |
| pair-probability RMSE | `0.001886` | `0.000451` |
| covariance cosine | `0.30113` | `0.93787` |
| covariance relative Frobenius error | `1.03626` | `0.35141` |
| frequent-pair recall (1,423 pairs) | `1.0` | `1.0` |

All four preregistered diagnostic checks pass.  The remaining covariance
relative error of `0.35141` is not hidden or promoted into a pass criterion;
it shows that rarer higher-order deviations remain even though normalized
pair mass and common pairs are well calibrated.

## Decision and next boundary

The correct attribution is
`initial_ratio_criterion_is_too_indirect_for_species_law_calibration`.
The historical E1 screen remains failed exactly as recorded.  It must not be
retroactively re-scored after seeing this audit.

A future versioned E1 qualification should replace random-initialization total
NLL ratio with an absolute conditional-species likelihood relative to a legal
empirical reference, fixed-partition element-pair calibration, explicit rare-
stratum floors, and the existing exact-count/zero-failure checks.  Its split,
checkpoint and thresholds must be frozen independently before it runs.  Until
that happens, count-constrained site assignment, L1/M1, tensor/oracle work,
relaxation, DFT and DFPT remain blocked.

Machine-readable evidence is in `result.json`; group attribution and the 25
largest pair errors are in `conditional_nll_groups.csv` and
`cooccurrence_top_errors.csv`.
