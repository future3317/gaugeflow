# GaugeFlow vNext execution contract lock

The controlling specification is `../CODEX_IMPLEMENTATION_SPEC.md`, SHA-256
`3bdac52ba00a14c40e8bb6f9de732d16d8a91eb5f81e1a9a2e9b2334e8dd952b`
(2026-07-15). The package keeps its historical name `gaugeflow`; every new
implementation is isolated under `gaugeflow.vnext`.

The gate order is Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, then a manually
unlocked Q10. A failed or blocked gate prevents every later gate. Real tensor
training, learned-oracle use, relaxation, DFT, and DFPT are prohibited unless
all stated predecessors pass and Q10 receives a separate human authorization.

Q0 is a diagnostic audit, not a scientific qualification. To avoid calling it
a pass, its terminal states are `complete` or `blocked`; Q1 accepts only
`complete`. This is the conservative resolution of the specification's generic
`pass|fail|blocked` example and its explicit Q0 instruction, "do not set a pass
conclusion."

At contract lock time, the required historical P5-C0 checkpoint does not
exist. The frozen runner saved metrics and couplings but no model weights, and
no D0.4--D0.8 checkpoint exists elsewhere in the repository workspace. Q0 may
compute checkpoint-independent diagnostics, but must report `blocked` if the
learned-Jacobian and learned-solver diagnostics cannot be completed. It may not
retrain or reconstruct weights from metrics.

The original run did report `blocked` and remains immutable. A subsequent code
review identified that permanently coupling the independent regular-affine Q1
qualification to an unrecoverable artifact deadlocks the research program.
The versioned amendment `vnext_protocol_amendment_q0_1.md` therefore adds a
partial-legacy execution status with corrected diagnostic semantics and a
separate P0 release checklist. Only the versioned Q1v2 protocol may consume
that authorization; this paragraph does not edit or reinterpret original Q0.

Equation (28) in the supplied Markdown contains a form-feed typo before
`rac1N`; implementations use the unambiguous horizontal projection
`s_i - (1/N) sum_j s_j`.
