# Current-state paper figures v1

These figures are regenerated from versioned report artifacts rather than
copied from superseded experiment panels.

- `base_probability_factorization.*` defines the current
  `p(B,N,C,A,L,F)` interface and makes no qualification claim.
- `composition_assignment_evidence.*` combines the qualified absolute-
  likelihood composition E1, failed legacy unary Q1, geometry-complete
  expressivity audit, and no-training orderless Q0. Each panel retains its own
  protocol boundary.
- `conditional_coordinate_qualification.*` visualizes the passed one-pass
  clean-`A,L` coordinate Gate. It does not claim near-prior or generated-side-
  state closure.
- `gaugeflow_training_roadmap.*` is the planned A0--F dependency graph, not
  completed evidence.

Regenerate from the repository root with:

```powershell
python scripts/plot_paper_state_figures.py `
  --reports-root reports `
  --output-dir reports/paper_current_state_figures_v1
```

`manifest.json` records every input and output SHA-256 digest. The manuscript
uses the PDF vector exports; PNG copies are retained for quick visual review.
