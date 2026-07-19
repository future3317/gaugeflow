# H1a SDE–probability-flow sampler audit v1

Status: **implemented and preregistered; not run**.

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

The formal 25/50/100-NFE CUDA sampling comparison has not been started because
sampling and training are paused. No result or Gate status is claimed.
