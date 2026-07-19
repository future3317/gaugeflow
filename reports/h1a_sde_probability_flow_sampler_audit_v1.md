# H1a SDE–probability-flow sampler audit v1

Status: **completed; probability-flow candidates failed the frozen quality
non-inferiority checks**.

## Decision encoded in the implementation

GaugeFlow remains a hybrid diffusion model. Atom types retain the absorbing
categorical reverse process, coordinates retain the wrapped Brownian path on
the translation quotient, and lattice variables retain the VP process in the
log-volume/trace-free log-metric chart. The new sampler interface changes the
continuous solver, not the training objective or probability path.

For coordinate variance `v`, a reverse step from `v_from` to `v_to` now uses

```text
reverse SDE:      x_to = x_from + (v_from-v_to) score(x_from) + bridge noise
probability flow: x_to = x_from + 0.5 (v_from-v_to) score(x_from)
```

The former `stochastic=False` path incorrectly removed noise while retaining
the full reverse-SDE drift. For lattice VP states, the former deterministic
path returned a DDPM posterior mean. It is replaced by the DDIM transport

```text
epsilon_hat = (x_t - alpha_t x0_hat) / sigma_t
x_s         = alpha_s x0_hat + sigma_s epsilon_hat.
```

Because atom types remain categorical stochastic, the combined method is a
**hybrid sampler with deterministic continuous probability-flow transport**,
not a pure ODE over the complete crystal state.

## Quotient convention

The feedback suggested wrapping coordinates after every solver stage. That is
not adopted because it conflicts with GaugeFlow's qualified quotient contract.
The reverse trajectory remains on a translation-horizontal universal-cover
lift. Periodic wrapping is used inside geometry construction, horizontal mean
projection is applied after every update, and `wrap01` is used only for the
terminal decoded structure. Per-stage wrapping would reintroduce cut-locus
image switches into the numerical trajectory.

## Common-random-number interface

The sampler now separates three random streams: continuous initialization,
categorical transitions, and continuous SDE noise. A reusable continuous prior
state can be injected into both solver modes. Consequently SDE and
probability-flow runs can share the exact same initial coordinates and lattice
latents while the categorical stream remains independent of whether the
continuous solver consumes Brownian noise.

## Scientific scope correction

The active two-pass checkpoint at step 16,882 was trained with the
`coordinate` objective. Its element and lattice heads are not a qualified
joint generator. Running the full reverse sampler and interpreting composition
or lattice differences would therefore be invalid. The frozen protocol keeps
clean atom types and lattices fixed and evaluates only coordinate sampling;
composition and lattice are reported as exact controlled invariants, not as
qualified generated metrics. A full-hybrid SDE–ODE comparison must wait for a
separately qualified jointly trained checkpoint.

## Frozen CUDA result

The committed protocol was executed once on the RTX 4060 Ti using the fixed
256-graph validation panel, the step-16,882 EMA checkpoint, common continuous
initial states, and no optimizer step.

| continuous solver | NFE | normalized NN W1 | distance >=0.5 A | latency (s) | graphs/s |
|---|---:|---:|---:|---:|---:|
| reverse SDE | 25 | 0.75170 | 1.00000 | 44.28 | 5.78 |
| reverse SDE | 50 | 0.56892 | 0.99609 | 82.54 | 3.10 |
| reverse SDE | 100 | **0.56626** | **1.00000** | 170.98 | 1.50 |
| probability flow | 25 | 1.02007 | 0.98438 | 41.82 | 6.12 |
| probability flow | 50 | 0.79848 | 0.99609 | 85.78 | 2.98 |
| probability flow | 100 | 0.67502 | 0.99609 | 170.26 | 1.50 |

All six paths produced finite states with zero failures. Peak CUDA allocation
was approximately 120 MiB for every path. Composition and lattice identity are
one by construction and remain controls rather than generated-quality claims.

The 25-NFE probability-flow path meets the latency requirement at `0.2446` of
the reverse-SDE-100 latency, but fails both nearest-neighbour Wasserstein and
minimum-distance non-inferiority. The 50-NFE path meets latency (`0.5017`) and
minimum-distance requirements but still fails nearest-neighbour Wasserstein:
its normalized W1 is `0.79848`, versus `0.56626` for the reference and a frozen
maximum additive degradation of `0.05`.

## Decision

Do not promote probability flow to the production fast sampler. Removing
continuous reverse stochasticity makes the generated local-distance
distribution worse under the current learned score, and even 100-NFE
probability flow remains worse than 100-NFE reverse SDE.

The observed reverse-SDE-50 value (`0.56892`) is close to reverse-SDE-100 while
using about half the latency, but it was not a preregistered candidate in this
Gate. It is recorded as a follow-up hypothesis only and cannot silently replace
the production NFE without a new held-out qualification.
