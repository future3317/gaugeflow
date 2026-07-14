# Gate A6 discrete atom-type substrate

A6 replaces only the failed continuous atom-type path. It uses an absorbing non-chemical mask and an endpoint-posterior discrete flow sampler; it does not modify A4/A5 or start tensor conditioning.

## Analytic discrete-path closure

Exact-posterior terminal atom accuracy: `1.000`; terminal masks: `0`.

## Fixed endpoint-ID type result

Composition accuracy: `0.750` (minimum `0.95`); atom accuracy: `0.469` (minimum `0.95`).
Common-noise terminal discrete difference: `0.750` (minimum `0.25`); masks: `0`; failures: `0`.

A passing A6 qualifies only the type substrate and requires a separate geometry protocol; it never passes Gate A or authorizes tensor-conditioned training.
