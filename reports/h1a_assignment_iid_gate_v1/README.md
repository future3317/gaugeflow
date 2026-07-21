# H1a assignment IID Gate v1 — frozen FAIL

This directory preserves the first formally preregistered IID oracle-composition
assignment run.  It was executed once from implementation commit
`885ff40158c36d2f9779fff810688fd90b2f0aca` with seed 5705 on an RTX 4090.  The
checkpoint SHA-256 is
`410a407ead5d9407efb3dbed49c3f79dc61351183f13c529612d85683805c160`.

The Gate failed and is not rewritten by later diagnostics.  Exact composition,
finite gradients, zero sampling failures, IID-test retrieval lift, IID-test
orbit-aligned accuracy, and the small-N exact subset checks passed.  The
calibration paired bootstrap, IID-test relative ELBO-NLL reduction, IID-test
paired bootstrap, and FP32 node-relabel check failed.

Key frozen observations are:

- calibration order-ELBO NLL: 5.68645 versus uniform 9.68801; paired
  material-bootstrap UCB95 of the difference: +1.71626;
- IID-test order-ELBO NLL: 8.29439 versus uniform 8.39346; relative reduction:
  1.18%; paired material-bootstrap UCB95: +1.45001;
- IID-test retrieval: 0.30288 versus uniform expectation 0.04180;
- IID-test orbit-aligned site accuracy: 0.69464;
- exact composition: 1.0 and sampling failures: 0;
- small-N exact quotient probability lift: +0.34201;
- maximum FP32 relabel logit residual: 7.534e-4 versus the frozen 2e-5 limit.

The JSON retains the historical field name `quotient_lower_bound_nll`.  A
post-run audit determined that the large-N quantity is a Monte Carlo estimate
of a uniform-order Jensen lower bound, not an exact quotient marginal.  Its
exponentiated finite-sample estimate can exceed one after adding the orbit-size
term.  The archived numbers are therefore interpreted as order-ELBO evidence;
the exact subset-DP panel remains the true small-N quotient-probability check.

Read-only localization also found that the nominal IID panels mixed two
scientifically different regimes.  Test carriers whose target-free
`embedding_key` occurred in fit had mean model-minus-uniform NLL -1.2245,
whereas unseen carrier signatures had +2.4332.  The latter 16 carriers drive
the aggregate likelihood failure.  A successor protocol must preregister a
supported-carrier IID calibration Gate and retain unseen carrier topology as a
separate stress panel.  This diagnosis does not retroactively pass v1.

The archived `read_only_audit.json` adds three checks without updating model
parameters.  At 256 reveal-order samples per IID-test carrier, the mean path
log-probability standard deviation is 3.1861 and the maximum is 8.1021; the
corresponding Monte Carlo standard errors are 0.1991 and 0.5064.  Re-evaluating
the historically exponentiated overflow rows with 1,024 orders removes every
value above one, confirming finite-order Monte Carlo error rather than a
normalized-probability implementation failure.  The input edge features are
exactly relabel consistent.  Strict FP32 matmul reduces the maximum raw-logit
residual to `5.52e-6`, FP64 gives `1.62e-14`, while the fixed-path objective
retains a small reduction-order residual.  The successor therefore uses strict
FP32 and deterministic sorted segment reductions rather than widening the
threshold.

Boundary: this result does not authorize generated composition, `p(N)`,
lattice, coordinates, joint generation, tensor conditioning, relaxation, DFT,
or DFPT.
