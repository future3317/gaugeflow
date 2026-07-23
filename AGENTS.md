# GaugeFlow contributor instructions

These instructions apply to the entire repository.

## One active baseline

The only runtime is the revised hybrid-diffusion implementation under
`gaugeflow.production`. Do not restore retired continuous-logit flows,
harmonic/Hopf conditioners, FlowMM dependencies, exploratory Gate runners, or
checkpoint compatibility fallbacks. Historical reproduction belongs to Git
tags and `docs/research_iteration_history.md`, not to active dispatch code.

GaugeFlow and PiezoJet remain separate projects. GaugeFlow may consume
versioned data/oracle artifacts but must not import PiezoJet modules.

## Current boundary

- Mathematical/state-space and Cartesian-atlas runtime qualification passed.
- The tensor-free trainer, EMA/checkpoint recovery, and joint reverse sampler
  passed a bounded CUDA software closure; this is not real-data evidence.
- H0 data/catalogue qualification passed. The current representation uses a
  species-free parent carrier plus exact occupational ordering, low-index
  supercells, OPD displacement modes, Kelvin strain, and a bounded residual.
- The complete 675,204-row H1a P1 cache is built and independently qualified.
- Real-data H1a is frozen as failed: joint training learned coarse chemistry
  and lattice statistics but not the train-reference local-geometry
  distribution; one-pass coordinate-only pretraining also missed its frozen
  validation and low-noise endpoint thresholds.
- The joint H1a run did use all 540,164 training structures for 20,000 updates
  (1,280,000 graph presentations, about 2.37 passes). Later 1/4/16/64-state
  panels are mechanism audits, not substitutes for that full-data run.
- Corrected tangent and exact Helmert-quotient audits show full physical rank
  `30/30` but severe anisotropy: condition numbers are `2.3e7--3.5e7`, effective
  rank is about `2.2`, and a one-state exact readout needs an update of
  `2079.20` from an initial norm of `0.80036`. The current attribution is
  optimization geometry plus state-dependent feature learning, not a missing
  coordinate direction, corrupt cache, or probability-path closure failure.
- No external pretrained weights or failed H1a checkpoint initialize the active
  model. "Coordinate-only pretraining" denotes a from-scratch auxiliary
  objective on the qualified train split, not transfer learning.
- The corrected Cartesian-tangent baseline completed one exact pass over all
  540,164 train structures at seed 5705 but failed final/initial validation
  (`0.70396 > 0.5`) and the low-noise endpoint (`0.04207 A > 0.04 A`).
- A zero-step readout-span audit rules out the final global linear head as the
  main limitation. At step 8441 a train-fitted minimum-norm head changes train
  loss by only `-5.23%` and validation loss by `+3.46%`; a validation-label
  oracle head explains only `49.61% < 75%`. All designs are rank `80/80`, but
  validation effective rank is `21.32`. The classification is
  `backbone_span_limited`. Production now retains one compact factorized
  Cartesian angular-moment mechanism: 64-dimensional persistent scalar edge
  state, eight first/second-moment channels, one concatenated linear-time
  segment reduction, and no explicit triplet index. It adds 466,944 parameters
  and is qualified at `489.10 graphs/s / 182.86 MiB` on the RTX 4060 Ti.
- Its exact-one-pass run improves validation ratio to `0.63864`. A causal audit
  then qualifies the equivalent chart `v_tilde=V^(-1/3)v_r`, which reduces the
  ratio to `0.58940` while preserving the fractional path and sampler. The
  low-noise endpoint is `0.040084 A`; the run still fails.
- A degree-three STF extension improves the ratio only to `0.57240` and the
  low-noise endpoint to `0.03938 A`, at lower training throughput. It is not in
  active runtime. The only active operator is the vectorized degree-one/two
  implementation; do not restore cubic, harmonic, or compatibility branches.
- Refreshing persistent edges from current layer state improves the ratio to
  `0.54417` but still fails. Explicit shell-complete TopK triplets (`0.56794`),
  unbalanced induced R=8 (`0.54583`), and fixed-six-iteration balanced induced
  R=8 (`0.53314`) all fail. The balanced branch is causally used but remains
  non-specialized: maximum slot mass `0.19579`, minimum representation rank
  `1.351`, and maximum inter-slot cosine `0.99974`. TopK, induced/R16, matched
  initialization, their active dispatch, and their experiment runners are
  removed. Do not restore them or search slot rank/balance iterations.
- H1b and H2--H6, tensor conditioning, oracle work, relaxation, DFT, and DFPT
  have not started.
- The frozen-checkpoint middle-noise reciprocal attribution is complete and all
  three preregistered checks fail: endpoint retrieval is `0.40315 < 0.75`, the
  low/high normalized reciprocal-residual ratio is `1.05348 < 1.15` with
  `0/5` supporting times, and the frozen low-k probe improves held-out MSE by
  only `0.002257` with low-minus-high `0.002939 < 0.03`. The decision is
  `do_not_implement_reciprocal_carrier`. Do not revive either reciprocal
  residual or add a new reciprocal production carrier.
- The independent Bridge audit reaches the same NO-GO on the earlier
  volume-normalized checkpoint: middle-noise held-out low-frequency explained
  fraction `-0.001368`, low-minus-random `0.000695`, low-minus-graph-token
  `-0.001368`, and low-shell excess over the permutation null `0.007755`.
  Its source hash and relationship to the main audit are recorded in
  `bridge_no_go_synthesis.md`. Do not rerun either low-k protocol.
- The complete all-pair clean-topology v2 audit covers `1.0` of the clean
  coordination mass. Middle-noise clean/noisy soft Jaccard is `0.50413`, hard
  switch fraction is `0.26469`, and the clean oracle improves residual energy
  by `0.10716` while the noisy control gives `-0.00354`. The frozen probe is
  predictive (`AUC=0.87923`, explained fraction `0.61362`) but its direct
  plug-in carrier worsens the residual by `0.04391`. The decision is
  `probe_predictive_but_topology_correction_not_residual_causal`; do not add
  the frozen probe or linear topology carrier to production.
- The unchanged dynamic coordinate model completed one separately frozen
  from-scratch two-pass exposure curve at seed 5705. Validation ratios at
  `0/0.25/0.5/1/2` passes are
  `1.00000/0.73837/0.63348/0.54371/0.49103`. One-to-two-pass improvement is
  `0.096876`, between the preregistered plateau (`<=0.05`) and undertraining
  (`>=0.10`) boundaries, so the decision is `ambiguous`. The one-pass point
  reproduces the archived `0.54417`; H1a remains failed. Do not add another
  pass or seed, and do not use the post-hoc crossing of ratio `0.5` to claim a
  Gate pass.
- The zero-training coordinate sampler comparison is complete on the same
  two-pass checkpoint and a fixed 256-graph validation panel. Probability flow
  at 25/50 NFE meets the latency checks but fails nearest-neighbour Wasserstein
  non-inferiority (`1.02007/0.79848` versus reverse-SDE-100 `0.56626`); 25 NFE
  also degrades the >=0.5 A minimum-distance fraction. All paths are finite
  with zero failures. Retain reverse SDE; do not promote deterministic
  probability flow. Reverse-SDE-50 (`0.56892`, about half the latency) is a
  post-hoc follow-up hypothesis, not a qualified production replacement.
- The independent 512-structure nested-Brownian qualification rejects that
  reverse-SDE-50 hypothesis: its W1-difference UCB95 is `0.05767 A > 0.03 A`
  and its 1%/5% minimum-distance lower tails degrade, despite a `0.4957`
  latency ratio, finite states, zero failures, and non-inferior endpoint RMS.
  Retain reverse-SDE-100 for coordinate-only audits and stop sampler search.
- The exposure-conditioned exact all-pair topology audit is complete with a
  frozen `mixed` decision. The two-pass middle clean-oracle gain is `0.09293`,
  retention versus 0.25 pass is `0.6640`, and all three middle-time bootstrap
  lower bounds remain positive. The effect is time-localized: gain falls to
  `0.04099` at `t=0.4` but remains `0.14203` at `t=0.6`. Probe explained
  fraction rises to `0.65384`, while the fixed learned carrier remains harmful
  (`-0.04325`). This neither authorizes a full ACF branch nor supports more
  exposure as a sufficient repair.
- The subsequent zero-training quotient-Tweedie audit also closes the simple
  staged self-conditioning hypothesis. At `t=0.6` its topology MSE improves
  `31.27%` over the noisy field, but AUC is `0.77003 < 0.8`; more importantly,
  inserting it through the shared clean-oracle Cartesian carrier worsens the
  held-out coordinate residual by `4.95%`, with the full structure-bootstrap
  95% interval below zero. The stronger frozen linear probe is likewise
  non-causal. Do not add ACF, a one-step Tweedie topology branch, or the old
  linear carrier to production.
- Variant-specific optimal ridge carriers remain non-causal, and a matched
  nonlinear pair-to-vector MLP yields only `+0.00537` held-out topology
  increment with a structure-bootstrap interval crossing zero while both
  readouts overfit. Do not add another deterministic topology conversion.
- A separate production audit found that historical coordinate-only training
  corrupted elements and lattice although conditional rollouts fixed both to
  their true values. The repaired clean-side-information contract passes its
  frozen 2,111-step screen: validation ratio `0.49382` versus historical
  `0.73837`, `t=.6` explained fraction `0.39070` versus `0.13024`, rollout RMS
  `0.07684/0.12153 A` from `t=.1/.2`, and zero failures. Future coordinate-only
  training/evaluation must set and verify
  `coordinate_clean_side_information=true`; joint generation must not use it.
  This does not change historical H1a or authorize ACF, H1b-H6, tensor/oracle
  work, relaxation, DFT or DFPT.
- The unchanged repaired task has now completed one exact pass over all 540,164
  training structures and passes every frozen check: validation ratio
  `0.33219`, t=.6 explained fraction `0.63509`, t=.005/.1 endpoint RMS
  `0.03756/0.04919 A`, rollout RMS `0.05123/0.07039 A` from t=.1/.2, and zero
  failures. This qualifies only `p(F|A,L,N)`, where `A` is the clean per-node
  element-token list, not composition counts alone. It is the conditional-coordinate
  substrate. Historical free joint H1a remains failed. Do not rerun retired
  local/topology branches on the old mismatched task.
- The same clean-side coordinate law is now strictly requalified in the current
  unified separate-clock backbone at seed 5705. Its one-pass validation ratio
  is `0.29798`, t=.6 explained fraction is `0.64857`, t=.005/.1 endpoint RMS is
  `0.03626/0.04751 A`, and reverse-SDE-100 rollout RMS from t=.1/.2 is
  `0.05177/0.07040 A`, with zero failures. This authorizes only the frozen
  generated-assignment/generated-lattice exposure panel. It does not qualify a
  full-from-prior coordinate trajectory, free joint H1a, capacity scaling,
  tensor conditioning, relaxation, DFT, or DFPT.
- The separately frozen 18-material supported-IID generated-side exposure Gate
  now passes. Clean-A/clean-L, generated-A/clean-L, clean-A/generated-L, and
  generated-A/generated-L full reverse-SDE-100 coordinate rollouts have
  normalized nearest-neighbour W1 `0.38042/0.40121/0.40345/0.41566`; assignment,
  lattice, and joint additive degradations are `0.02079/0.02303/0.03525`. Every
  arm has minimum-distance validity `1.0` and zero failures, assignment counts
  are exact, and lattice permutation residual is `1.01e-6`. This authorizes the
  preregistered GaugeFlow-base capacity screen only. The evidence is bounded to
  oracle composition and supported parent actions; generated composition,
  unseen-action closure, free joint M1, tensor work, relaxation, DFT, and DFPT
  remain blocked.
- J0 confirms that this qualified field materially uses both side modalities:
  at coordinate time 0.5, controlled element and lattice corruption increase
  score MSE by `5.335x` and `5.163x`, and both give `9.939x`. J1 therefore ran
  one seed-5705, 2,111-step independent-clock attribution in the same backbone.
  Its clean/noisy-element/noisy-lattice/diagonal/interior validation ratios are
  `0.47273/0.51407/0.56107/0.57304/0.64015`; clean retention and diagonal
  improvement both pass their frozen bounds. All three clock embeddings have
  finite nonzero gradients. This supports a unified multimodal hybrid
  diffusion but does not isolate clock identity from the changed task mixture
  and 3.9% capacity increase, and does not qualify free joint H1a. The next
  parameter-matched C0/C1/C2 comparison is now complete and fails its specific
  clock-attribution criterion: C2-minus-C0 diagonal/interior intervals cross
  zero, although C2 significantly improves clean and element-only corners.
  J1 remains a successful composite task-mixture intervention, not proof that
  separate clocks caused its noisy/noisy gain.
- The zero-step gradient-geometry audit finds no persistent regime conflict:
  median clip scale is `0.2661 > 0.2`, every pairwise median cosine is positive,
  and no pair reaches the frozen 75% negative fraction. Retain global clipping;
  do not add blockwise clipping, AGC or target-RMS normalization.
- E1 element-only reverse qualification is complete through four bounded
  mechanisms, all at seed 5705 and the same 2,111-update budget. The absorbing
  path reaches free site accuracy `0.03843` and exact composition `0/256`;
  uniform D3PM raises free site accuracy only to `0.06175`, with composition
  overlap `0.08144` and exact composition `0/256`; a graph-composition head
  gives `0.05944/0.08684/0` on those metrics; and an exchangeable current-token
  histogram residual gives `0.03396/0.06831/0`. The latter correctly repairs
  low-noise counting (`t=.25` overlap `0.87534`, exact composition `0.27734`,
  clean-token-oracle exact `0.89062`) but cannot create a coherent formula at
  high noise (`t=.9` overlap `0.08530`). Oracle target counts still raise site
  accuracy to about `0.70` and exact assignment to `0.27--0.36`. Therefore do
  not add another composition head, local feature, training exposure, loss
  search or sampler search to this independent-site state space.
- The qualified H1a cache contains 540,164 train graphs with at most 20 atoms,
  76 active elements, `99.7408%` of graphs containing at most four species, and
  a maximum of seven. This authorizes only a no-training exact sparse
  composition-state kernel qualification, followed by a separately frozen
  composition-only Gate if the kernel and composition-identifiability audits
  pass. It does not authorize L1/M1, tensor/oracle work, relaxation, DFT or
  DFPT. Target composition remains prohibited as a model input.
- The subsequent composition audit found an evaluation-contract issue rather
  than corrupt data: the qualified H0 matcher envelope contains reduced
  anonymous stoichiometry, so train and formula/prototype-disjoint validation
  have exactly disjoint integer-partition support at every populated node
  count (conditional TV `1.0`). Preserve that split for OOD novelty/coverage,
  but never use its marginal TV as an IID calibration Gate or train on its
  validation rows.
- The only active explicit-composition implementation is the exact
  stoichiometry-first law. It enumerates 1,840 integer partitions for `N<=20`,
  fits a train-only smoothed `p0(lambda|N)`, and generates distinct elements in
  decreasing-count order with increasing-token tie breaking. A shared
  count-position encoder and current-count query replace the retired
  per-partition lookup and interleaved count autoregressor. Q2 passes exact
  normalization, finite gradients, FP32/BF16 and CUDA performance on the
  archived RTX 4060 Ti qualification (`3.55 ms` teacher forcing, `15.17 ms`
  sampling, `50.63 MiB` for 256 graphs).
- The archived one-pass random-initialization-ratio screen remains failed at
  `0.77569 > 0.75`; it is not rewritten. Its read-only attribution showed that
  the statistic mixed the learned species law with an unchanged partition
  prior and arbitrary initial-logit scale. A separately preregistered
  absolute-likelihood Gate has now run on independent
  fit/calibration/test rows (`486340/26912/26912`) and qualifies only
  `p(C|N)`. Test conditional-species NLL is `3.26541`, versus `3.642995` for
  the legal train-only empirical baseline and `4.159026` for the legal uniform
  law; the structure-paired model-minus-empirical bootstrap 95% upper bound is
  `-0.35872`. Test pair JSD/RMSE/recall are
  `0.009112/0.000473/1.0`; atom-count preservation is `1.0`, invalid
  compositions and sampling failures are zero, and supported-element recall is
  `1.0`. This run used one exact pass, seed 5705, and an RTX 4090; its
  throughput (`13503.63 graphs/s`) and memory (`53.53 MiB`) must not be quoted
  as RTX 4060 Ti performance.
- The species-free occupational carrier audit has now passed all eight frozen
  checks on 454 candidates (`358/43/53` train/val/test), at most 20 atoms, five
  observed species and 1,053 mixed-radix DP states. Median uniform target
  quotient probability is `0.00015873`, and every carrier breaks at least one
  parent action orbit occupationally. In `41.8502%` of catalogues, distinct
  crystallographic operations induce the same finite-site permutation; the
  audit quotients to the faithful image `G_parent -> S_N` and checks group
  closure. Duplicate operation multiplicity must never weight assignment
  likelihood.
- This audit authorizes only a separately versioned oracle-C count-constrained
  assignment Q1. It does not qualify `p(N)`, site assignment itself, L1/M1,
  free joint H1a, tensor/oracle work, relaxation, DFT or DFPT. Oracle-C and
  generated-C assignment results must remain explicitly separated.
- The separately frozen oracle-C assignment Q1 has now run once at seed 5705
  from commit `4fa6093` and fails. Validation/test exact target-quotient
  probabilities are `0.12324/0.22052 < 0.25`, sampled orbit-aligned site
  accuracies are `0.53458/0.61121 < 0.8`, and model-minus-uniform quotient-NLL
  UCB95 values are `4.74238/6.84618 > 0`. Exact and sampled compositions remain
  exact, failures are zero, and sample retrieval agrees with exact probability.
  A read-only checkpoint audit gives train quotient NLL `2.77939` versus
  uniform `8.00377` and reaches `99.86%` of the implemented site-signature
  unary ceiling, ruling out failure to fit the training carriers. The frozen
  OOD split has zero train support for validation/test composition partitions
  and only `25.58%/13.21%` exact action-signature coverage. Q1 therefore rejects
  the present unary scorer under this OOD contract; do not add steps,
  target-derived occupation fields, generated-C, `p(N)`, L1/M1, tensor work,
  relaxation, DFT or DFPT.
- The assignment-specific IID split is now independently frozen without
  consuming the OOD panels. It partitions only original-train materials into
  `98/37/23/23` IID-fit/rare-fit/calibration/test materials
  (`174/90/42/52` carriers); original validation/test remain untouched OOD
  panels with `43/53` carriers. Composition-partition fit support is exactly
  one in both IID panels, exact input-output duplicate overlap is zero, and
  target-free action-signature fit support is `0.8333/0.6731`. IID and OOD
  evidence must remain separately labelled.
- Two zero-training global-coloring audits close the action-only pair route.
  Exact carrier-specific pair-orbit IDs resolve `93.66%` of exactly enumerated
  unary collisions, but they are only a mathematical upper bound: independent
  orbit IDs are not a shared relabeling-invariant input. Aggregating the
  target-free pair descriptors resolves only `3.93%`, and retaining exact
  orbitals as a shared unordered DeepSet resolves only `4.23%`; their mean
  target ceilings are `0.35765/0.36468`. Do not train either representation or
  treat the carrier-specific upper bound as a production result.
- The matched carrier-interface audit found that archived O1 serialization
  retained site-resolved geometry for only `158/454` carriers and omitted the
  expanded supercell fields for all 296 index-2--4 paths. The separately
  versioned geometry-complete compiler now passes on all 454 unchanged O1
  occurrences: candidate/HNF/node/action/target/relabel closure are all `1.0`,
  the index-1/2/3/4 counts are `158/230/22/44`, maximum periodic alignment
  error is `4.61e-14 A`, and failures/nonfinite values are zero. Carrier and
  target fields are structurally disjoint. This repairs the offline interface,
  not failed Q1; it authorizes only the frozen geometry-aware zero-training
  expressivity audit. Assignment training remains blocked until that audit
  passes and an exactly normalized successor law is separately qualified.
- The geometry-aware zero-training successor audit now passes its frozen
  aggregate criteria on all 454 carriers. Expanded-geometry unary signatures
  alone resolve `47.36%`; among the remaining 239 exact collision classes, the
  transferable complete two-point distance descriptor resolves `87.87%` with
  mean target ceiling `0.93933`. Exact enumeration coverage is `1.0`, and node
  relabeling, `GL(3,Z)` basis, and target-orbit containment have zero failures.
  The stratified result is important: OOD validation/test pair resolution is
  `0.9565/1.0`, while IID test is only `0.6364` with ceiling `0.81818`.
  Geometry is therefore a necessary material repair but a static pair-energy
  histogram is not the production law. The next permitted work is a bounded
  software qualification of a count-exact, permutation-equivariant
  remaining-count autoregressive law with target-independent reveal-order
  marginalization and auditable normalization; no assignment training is yet
  authorized.
- The ensuing no-training remaining-count Q0 now passes every frozen
  mathematical and CUDA check. Complete-distribution normalization and
  subset-DP/brute-force errors are `4.44e-16/1.04e-17`; FP64/FP32 node
  equivariance errors are `1.11e-15/4.77e-7`; residual-stabilizer error is
  `2.98e-7`; exact-count sampling is `1.0`; and BF16 output cosine is
  `0.999974`. On an RTX 4090 the explicitly no-grad 64-graph forward is
  `5.07 ms / 99.05 MiB`. An initial software attempt incorrectly retained
  simultaneous FP32 and BF16 autograd graphs and reported `720.06 MiB`; its
  result is preserved, the `512 MiB` threshold was not changed, and training
  memory remains a separate future metric.
- Q0 authorizes exactly one separately frozen, single-seed IID
  oracle-composition assignment training Gate. The original validation/test
  panels remain untouched OOD stress panels. It still does not authorize
  generated composition, `p(N)`, L1/M1, free joint H1a, tensor/oracle work,
  relaxation, DFT or DFPT.
- The supported-IID exact-count assignment Gate has now passed. Calibration/test
  reveal-order Monte Carlo ELBO reductions are `0.70939/0.85290`, orbit-aligned accuracies are
  `0.93864/0.94080`, exact composition is `1.0`, and failures are zero. This
  qualifies only oracle-composition assignment on supported IID carriers;
  it is not an exact marginal likelihood over every `N<=20` carrier. Exact
  subset-DP evidence is limited to the frozen small-N audit subset. Unseen-action
  and formula/prototype-disjoint panels remain OOD stress failures.
- The explicit train-only empirical node-count law has passed its IID Gate:
  test NLL `2.41760` versus uniform `2.99573`, JSD `1.00e-4`, integer W1
  `0.02438`, and zero invalid samples/failures. Formula/prototype-disjoint
  node-count results remain separate OOD evidence.
- The coordinate-free P1 lattice Gate has passed on 4,096 validation structures.
  Aggregate teacher volume/shape MSE ratios are `0.05823/0.60711`; free-running
  volume/density/shape normalized W1 values are `0.09057/0.02201/0.49417`;
  all 4,096 lattices are finite with positive volume and failures are zero.
  Lattice training and reverse sampling do not accept coordinates, build edges,
  or call the tensor atlas. This qualifies only `p(L|C,N,P1)` with clean
  composition; shape W1 is close to its frozen `0.50` bound.
- Future successor probability/relabel metrics must cast frozen FP32 model
  scores to FP64 before exact DP evaluation. This resolves the archived
  `1.1444e-5` reduction-order residual without weakening its frozen threshold;
  it does not alter the failed Q1 result.
- The final substrate is one heterogeneous product-space reverse field over
  `(A,F,L)`, not three modules assembled after E1/L1/M1. The five J1 regimes are
  a finite task-path measure over `(t_A,t_F,t_L)`; their regime index is audit
  metadata, never model input. Joint, conditional, staged and alternating
  generation are sampler paths of this field. The tensor orbit is a shared
  quotient-valued condition, not a fourth diffused state. The exact family has
  a nested-corruption tower identity, but no finite-model consistency loss or
  information-coordinate ablation is authorized before E1/L1/M1 qualification.
- J2 is authorized only after these attribution/gradient results are recorded,
  separately frozen E1 element and L1 lattice reverse heads qualify, and joint
  M1 training qualifies.
  Generated side states must be on-policy at the same reverse-clock time. The
  J1 coordinate-only checkpoint leaves those heads untrained and must not be
  used to fabricate them. Do not choose a hard chain, start free joint
  training, or enter later Gates before these prerequisites are satisfied.

GaugeFlow-base has completed bounded supported-IID *component* qualification:
exact-count assignment, explicit `p(N)`, lattice L1 and the four-arm
generated-side coordinate exposure Gate have passed separately. The equal-exposure
capacity screen then compared 34.28M/57.68M/97.58M parameters for exactly
540,164 graph presentations at effective batch 64. All three candidates were
eligible; the frozen minimum-sufficient rule selected 34.28M. Its validation
ratio is `0.269575`, t=.6 explained fraction `0.729079`, and clean-side
conditional-rollout NN-W1 `0.148713`, with valid-distance fraction `1.0` and
sampling failures `0` at `238.26 graphs/s`. This rollout fixes ground-truth
atom types, lattice and node count; it is not free joint generation. The 98M
candidate improved conditional-rollout NN-W1 to `0.131120` and t=.6
explained fraction to `0.771143`, but remained inside the frozen quality
margins while reducing throughput to `69.88 graphs/s`; it is not the production
default. The production `joint` trainer and reverse sampler now implement the
qualified stoichiometry-first `p(C|N)` law and the orderless exact-count
remaining-count assignment path; legacy independent site-token checkpoints
are rejected as A1. The production-size integration Gate has nonzero
assignment loss/gradients and exact composition closure. A deterministic
three-step RTX 4090 execution smoke also passes exact interrupted-resume
equality with `0` mismatches and `14.87 GiB` peak memory. This authorizes only
the separately frozen one-pass 34M A1 run. That run is now complete and passes
the corrected v1.1 512-reference/512-free-sample Gate: final NN-W1 is
`0.555003`, volume-W1 `0.073341`, valid-distance/exact-composition/
positive-lattice/formula-uniqueness fractions are all `1.0`, element JSD is
`0.047493`, node-count JSD to the declared train-only prior is `0.003924`, and
masks/failures are zero. The final checkpoint SHA-256 is
`7c8fb7afc3aee6d4723d700b59f2a0523da25e897a46de8e9d2c7e5db824b6da`.
The v1.1 result SHA-256 is
`68ceb2b7806bcc933b9f080a4ac894e592fcfb1a9e05a78398c8d1e311a39713`.
The original v1 failure remains frozen: it compared samples from train-only
`p(N)` with the formula/prototype-disjoint validation count marginal and gave
checkpoint-invariant JSD `0.363640`. v1.1 changes only that reference object;
all non-count metrics reproduce bit-for-bit. The displaced validation JSD is
an OOD diagnostic, not the sampling-law closure metric. This pass qualifies
only flexible-carrier tensor-free GaugeFlow-base. H1b and H2--H6 remain
prohibited. Do not add seeds or steps to rescue completed protocols or revive
failed reciprocal/topology branches.

The reciprocal, clean-topology, exposure-conditioned, quotient-Tweedie,
variant-specific carrier and nonlinear pair-conversion audits are complete.
They reject another deterministic local/global feature branch. The matched
clean-side-information screen and exact-one-pass qualification identify the
coordinate task contract as the material repair. The conditional coordinate
substrate and the bounded tensor-free free joint A1 generator are qualified.
No result yet authorizes OOD parent generation, H1b-H6, tensor condition,
oracle, relaxation, DFT or DFPT.

Stage-B-v1.1 physical-representation pretraining is complete. The six immutable MatPES PBE/r2SCAN artifacts contain `433189/386544`
rows and `387129/362737` rows at `N<=20`. The qualified byte-offset index has
`749866` functional rows, `387697` unique IDs, zero invalid rows and an
ID-grouped `674709/37054/38103` train/calibration/test split. Cohesive energy,
forces and stress are complete in this domain; formation energy is partial and
must not be a fallback target. IDs never enter model batches. The active
preparation uses explicit cohesive-energy targets, vectorized collation and
train-only functional scalar/Kelvin normalization that preserves Cartesian
covariance. Physical and Alex replay losses have one optimizer owner; the
clean physical path shares the production message encoder but skips all
generation terminal heads.

The Stage-B-v1.1 run reduces composite physical calibration loss from
`19.6127` to `0.5929` and reaches PBE teacher-feature cosine `0.8996`.
On the unchanged A1 512-sample retention panel it preserves exact composition
and zero sampling failures, with NN-W1 `0.5444` and volume-W1 `0.0722`.
The qualified TensorNet feature contract is per atom (`N x 128`), not per
graph; production therefore uses a type-matched node projection and a
graph-equal mean of per-node cosine distances. A real 256-row cache smoke
covers 1,723 atoms with finite memory-mapped FP16 storage, and a real A1
backward pass has nonzero finite feature gradients on all 53 selected atoms.
The two-RTX-4090 execution smoke uses rank-sharded MatPES/Alex batches,
globally weighted/summed gradients, one rank-0 AdamW/EMA owner and parameter
broadcast. It passes exact model replication and interrupted-resume equality
for model, optimizer, EMA, per-rank generator state and metrics. This remains
software evidence only. The production runner now emits one shared global
permutation as no-padding rank shards: all 674,709 MatPES train rows are seen
exactly once, while Alex replay has its own deterministic wrapped stream.
Checkpoint state contains both per-rank cursors plus CPU, CUDA and explicit
noise-generator states. Masked physical heads use globally reduced
label-bearing graph counts, so the PBE-only feature loss remains unbiased when
a rank contains r2SCAN-only graphs. The bound evaluator reports normalized
energy/force/Kelvin-stress and force cosine separately for PBE/r2SCAN, PBE
node-feature cosine, and the unchanged A1-v1.1 free-generation retention
panel. The physical cache, CUDA runner resume smoke, and formal Stage-B-v1.1
run are complete. These results qualify physical-representation transfer and
A1 retention only; tensor conditioning, relaxation, DFT, and DFPT remain
blocked.

Stage-C preparation is isolated from the hash-bound Stage-B runtime.  The
LeMat dataset now exposes split-local functional groups to a deterministic
balanced stream in a separate module.  On the 16,196-row bounded train split,
1,000 global batches of 64 produced PBE/PBEsol/SCAN fractions
`0.32967/0.33353/0.33680` despite the raw `13,826/665/1,705` imbalance.  Two
rank shards contained 32,000 examples each and resumed exactly.  This is only
data-stream software evidence; it neither authorizes full LeMat continuation
nor changes the frozen Stage-B implementation hashes.

The LeMat build now uses a hash-bound audit-only exclusion generated from the
qualified packed Alex validation/test indices.  Each split has 67,520 rows;
their normalized union has 135,040 unique IDs and is absent from model
batches.  The full LeMat index must bind the exclusion-list SHA-256 rather
than relying on a dataset-name claim of benchmark isolation.

The resulting full qualified LeMat `N<=20` index contains 5,068,754 rows and
4,878,239 split groups.  Train/calibration/test counts are
4,563,032/252,475/253,247 with zero invalid rows; 129,152 eligible records
overlapped the 135,040-ID Alex benchmark exclusion and were removed.  The
train functional counts are 4,222,763 PBE, 9,014 PBEsol and 331,255 SCAN, so
raw row-frequency sampling is not an acceptable Stage-C default.

A second-pass native-fingerprint expansion collected 129,302 LeMat
`entalpic_fingerprint` values associated with Alex benchmark IDs.  All 129,152
eligible exclusions remained direct ID matches and zero additional rows were
cross-ID fingerprint matches; the selected index tensor is byte-identical to
v1.  This closes the provider-native fingerprint envelope without claiming
that it is a complete invariant of every periodic representation.

The Stage-C core now has one dataset-neutral structure batch/objective and one
atomic three-stream cursor.  LeMat and Alex contribute the unchanged
product-space denoising loss, while MatPES contributes the globally masked
physical loss.  Each local structure mean is weighted by its exact rank graph
fraction before gradient summation; physical heads retain their global
label-bearing denominators.  LeMat, MatPES and Alex cursors restore together
or fail closed. The executable runner and CUDA resume smoke are complete.
Stage-C-v1 reached its complete 20k checkpoint and then reproducibly rejected a
malformed LeMat row (`oqmd-2964825`: `nsites=8`, 15 positions/species). The v3
index had not inspected nested geometry lengths. A complete v4 rebuild checks
periodicity, positions, species and element symbols and removes exactly two
malformed OQMD records with hash-bound evidence; train count is 4,563,028.
Stage-C-v2 preserves the 20k model, optimizer, EMA, MatPES/Alex cursors and RNG
states and deterministically re-bases only the LeMat stream. Its one-step
three-GPU resume smoke passes, and formal training continues with checkpoints
every 5,000 steps. The 10k/20k physical composite losses are
`0.38709/0.32535`; NN-W1 values are `0.55329/0.56282`, with exact composition,
positive lattices, valid distances and zero failures retained.
The same fixed diagnostic at 30k/40k gives physical composite
`0.29083/0.26518`, teacher-feature cosine `0.93226/0.93639`, and NN-W1
`0.56561/0.57846`. All 512-sample hard-validity fractions remain one and
failures remain zero. This is a physical-transfer versus local-geometry
retention trade-off, not sampler collapse; the final checkpoint is selected
from the declared Pareto panel rather than by optimizer step alone.
The archived 10k--40k evaluator covered complete MatPES calibration and A1
retention but omitted the planned LeMat held-out structure panel. Its existing
numbers remain valid two-panel diagnostics. The separately declared final
selection adds a paired functional-balanced LeMat-v4 panel (500 graphs per
functional, exhausting the rare PBEsol calibration support) and applies one
transparent Pareto-minimax operational rule across v2 20k--50k checkpoints.
Stage-C-v2 has now completed all 50,000 declared updates at global step 60,523
with finite terminal metrics and a clean three-rank exit. The 50k three-panel
objectives are LeMat macro loss `1.486348`, physical composite `0.250460`,
NN-W1 `0.572337`, and volume-W1 `0.067552`; hard validity remains one and
failures remain zero. All 20k--50k candidates are eligible, 40k is Pareto
dominated, and the frozen Pareto-minimax rule selects Stage-C 30k (global step
40,523; SHA-256
`8807877bbdcc61090a431dc5cd146ed62bf545b2a65425ff8bb16c8d0d317bf9`).
The 50k checkpoint is the completed trajectory endpoint, not the operational
base. This remains tensor-free evidence and does not authorize E/F, tensor
conditioning, RL, relaxation, DFT, or DFPT.

Stage-D preparation now uses one immutable formula/prototype-disjoint response
cache with 3,946 graphs and 43,015 atoms (3,173/398/375). Label coverage is
3,946 piezoelectric, 3,943 dielectric/Born/Gamma, 2,893 audited JARVIS elastic,
and 1,266 strict internal-strain graphs. Elasticity follows the single GPa
engineering-Voigt -> Kelvin -> Cartesian C_ijkl conversion; all pre-existing
non-elastic cache tensors remain bitwise unchanged. Heavy-tailed tensor targets
use a train-only invertible O(3)-covariant radial-asinh chart rather than sample
deletion or ordinary RMS. The final cache/normalizer SHA-256 values are
`4f780dba78b422e7b6f3e0db338cf769c968b9865f7096f5d5add0227f737e1c` and
`27112e0c3f32911903ce9740942bbe50a10c3634275ab4c986946f1d3af87d35`.
The paired 2,000-step response-probe D0 screen selected the full Cartesian
baseline: the auxiliary worsened probe error by 0.0423% instead of improving it
by the required 5%, while tensor and other-task retention passed. Do not tune
the auxiliary weight or add it to formal D. D0 is mechanism selection, not
predictive qualification and not E/F authorization.

Formal Stage D then trained the independent full-Cartesian multi-task response
model to step 7,500 and selected the validation-best step 4,500 checkpoint
(SHA-256
`67dd8e8a4624fe87b6df2bc2580adfe04b777dfbad001102e7ecb2f6059a8497`).
Validation/test total losses are `0.284270/0.256640`; test piezoelectric and
response-probe losses are `0.249202/0.294194`, and dielectric, elastic, Born,
Gamma, and internal-strain losses are
`0.522682/0.059267/0.106917/0.329163/0.272606`. This qualifies the frozen D
checkpoint as the independent evaluator for E/F mechanism work; it is not a
claim of tensor-conditioned generation, relaxation, or DFT/DFPT closure.

Stage-E E0 uses paired forwards on the same noisy state, clocks, and random
draw while changing only the representative of one tensor orbit. The typed
loss compares categorical JS, Cartesian coordinate tangents, volume/shape
charts, and the marginalized response field; raw atlas candidate IDs are never
compared. The orbit-mimic arm improves validation fine loss
`2.067480 -> 1.818112`, reduces typed orbit residual by 54.4%
(`4.379e-4 -> 1.997e-4`), raises posterior information about ninefold, and
increases target-swap separation `0.103512 -> 0.238860`. Soft retention and
three exact-null repairs lose this conditional Pareto point. Keep only two
explicit roles: C-30k owns null/unconditional sampling, while the selected E0
checkpoint (SHA-256
`19392da08eb5d92ef3a4e7a799359983a62c6fd59a572d9f2d14475b68676b32`)
requires a supplied tensor condition and must never act as a null fallback.
E0 remains a mechanism screen; F stays blocked pending the paired rollout with
the independent Stage-D evaluator.

That 64-target, 50-step paired rollout has now completed and failed. The frozen
Stage-D normalized tensor-orbit RMSE is `1.066727` for C-30k and `1.403886` for
the condition-required E0 checkpoint; the paired conditioned-minus-base 95%
interval is `[-0.003943, 0.769720]`. NN-W1 worsens
`0.248767 -> 0.366399`, and the valid-distance fraction falls
`1.0 -> 0.984375 < 0.99`; volume-W1 slightly improves and both paths retain
zero failures and positive finite lattices. Therefore Stage E is not qualified
and F0 must not run. The next E implementation must condition the missing
composition/lattice side-state interfaces and include generated-side exposure;
do not attempt to rescue this result by increasing only the mimic weight.

The E3-v2 implementation repair is now complete.  The Cartesian invariant
encoder computes higher-order contractions on a unit-norm tensor and keeps a
bounded log-magnitude feature; lattice-only and hybrid paths share a
geometry-free tensor condition token; and the centered adapter is exactly zero
at initialization while retaining immediate adapter gradients.  The targeted
server suite is `42 passed`, with ruff and mypy clean.  Its one-seed mechanism
screen is finite (`condition_retention=0.00258868`, `null_retention=2.95e-10`,
`orbit_mimic=3.43e-6`), but target-swap separation remains `-0.00246`, so this
is a software/interface repair, not tensor-conditioning efficacy.

The frozen 256-structure E1a factorial diagnostic then completed with zero
sampling failures.  `oracle_cal` is stable (base/E3 orbit error
`1.25658/1.28583`); `oracle_ca` is the first large jump (base/E3 orbit error
`1747.62256/2.29049`, volume W1 `3.487e26/3.498e26`); `oracle_c` has volume W1
`26.3881/30.3458`; and `free` has volume W1 `0.33465/0.32005`.  The four
extreme lattice rows are pure krypton (token 35) physical-zero response
records.  The original panel is immutable; a v2 data-clean protocol explicitly
quarantines the six pure noble-gas validation rows and repeats the factorial.
Neither v1 nor the E3 mechanism screen qualifies tensor-conditioned generation
or authorizes F/RL.

The separately archived v2 data-clean factorial keeps the six pure noble-gas
rows out and still completes all 12 roles with zero failures.  `oracle_cal`
remains finite (base/E3 orbit `1.18895/1.22523`), but `oracle_ca` still has the
first generated-side jump (base/E3 orbit `352.49091/2641.72339`, volume W1
`61.12725/64.31692`); `oracle_c` and `free` have orbit errors
`1.25292/1.26444` and `1.26829/1.26776` with no useful target separation.
Thus noble-gas support explains the original v1 outliers but not the remaining
lattice-only exposure failure.  The full v2 JSON was produced by a pre-schema-
fix evaluator and retains a v1-compatible top-level schema; the corrected
writer is verified by the archived v2 schema smoke.  Do not rewrite the full
run or treat either diagnostic as tensor efficacy.

The first clean production integration exposed a Cartesian index-type defect.
The reverse sampler adds a tangent drift to fractional coordinates, so the
only active chart is `v_r=v_f L` and `v_f=v_r L^-1`; the retired `L^T`
covector pullback must never return as a fallback.  After deterministic linear
reductions and a fixed FP32 geometry path, the no-training CUDA qualification
passed at `516.03 graphs/s`, with exact repeat determinism, output cosine
`0.999806`, and loss-gradient cosine `0.997593`.  The authorized one-pass
learning experiment nevertheless failed as recorded above, so work remains
inside H1a diagnosis and no joint initialization is allowed.

## Required environment

Windows is the local editing and Git host. WSL is not required and should
remain stopped unless a separately declared local Linux-compatibility test
needs it. Reported CUDA training, sampling and performance qualification run on
the laboratory server:

```text
/home/workspace/lrh/T2C-Flow/gaugeflow
/home/workspace/lrh/DATA
/home/workspace/lrh/miniconda3/envs/gaugeflow/bin/python
torch 2.5.1+cu124
CUDA 12.4
6 x NVIDIA GeForce RTX 4090
```

```bash
cd /home/workspace/lrh/T2C-Flow/gaugeflow
export PYTHONPATH="$PWD/src"
PY=/home/workspace/lrh/miniconda3/envs/gaugeflow/bin/python
$PY -m pytest -q
$PY -m ruff check
$PY -m mypy src/gaugeflow/production
```

Every reported performance number must retain its actual device. Archived RTX
4060 Ti qualifications remain valid for their stated scope; new RTX 4090
throughput or memory is not a 4060 Ti replacement. Do not use the Windows
CPU-only torch environment for reported numerical or performance results.

## Active data contract

- Large data stay under `E:/DATA/T2C-Flow`; do not copy them into Git.
- The active Alex split contains all 675,204 rows with exact counts
  540,164/67,520/67,520 and no cross-split formula, prototype,
  matcher-envelope, or connected-component overlap.
- H1a cache construction may apply only a certified Niggli `GL(3,Z)` basis
  change. It must fail closed; no unreduced fallback is allowed.
- IDs, source rows, split labels, formulas, prototypes, space groups, structure
  hashes, and Niggli transforms are audit metadata, never denoiser inputs.
- Future tensor validation/test uses source-verified TensorOrbit-JARVIS-v2 only
  after its Gate is authorized.

## Data cleaning

Confirmed corrupt or task-domain-incompatible records are removed at a
versioned dataset boundary, with ID, evidence, and hashes recorded. Never delete
raw source rows, rewrite an archived result, or add a model fallback for bad
data. Do not remove a scientifically valid hard example merely because the
model learns it poorly.

The current parent-occurrence exclusion list is
`configs/data_quality/parent_occurrence_quarantine_v2.json`. Its exclusions are
task-scoped: they do not automatically filter the P1 structure corpus.

## Physics and leakage rules

- The tensor condition and current noisy/generated state may be model inputs.
  Never pass a paired target CIF/lattice/graph, material ID, target space group,
  target stabilizer, endpoint token, or target species mapping to the denoiser.
- Distinguish a physical zero tensor from a missing/null condition.
- A polar rank-three tensor orbit is an `SO(3)` object. Improper `O(3)`
  operations enter crystal compatibility only with explicit parity.
- Descriptor-frame ambiguity groups are not automatically physical stabilizers.
- The Cartesian atlas is a state-dependent finite prior, not a Haar quadrature
  claim. Preserve candidate multiplicity correction, duplicate/order
  invariance, smooth stratum transitions, and the proven 4,032-candidate generic
  path.

## Hierarchical method rules

- The exact space group is a parent prior, not a mandatory terminal symmetry.
- The parent geometry is species-free. Terminal integer elements are a separate
  occupational field with exact stabilizer
  `H_occ={g: a[pi_g(i)] = a[i] for all i}`.
- The child group intersects supercell-preserving parent operations,
  occupational stabilizer, and active OPD stabilizers.
- The current scientific domain is ordered, stoichiometric, commensurate
  crystals with `det(B)<=4`, at most two active OPD branches, and a registered
  bounded residual. Do not represent defects, disorder, partial occupancy,
  large supercells, or thermal ensembles through the residual.
- Use the Cartesian atlas only after a concrete geometry exists. Pre-geometry
  categorical choices use orbit invariants and reachable-child compatibility.

## Implementation rules

- Prefer physically/mathematically equivalent compact representations and
  vectorization: node permutation plus 3x3 rotation instead of dense 3N actions,
  Kelvin coordinates for symmetric strain, shared factorizations for periodic
  CVP, packed stabilizers, and batched reductions.
- Equivalent acceleration must preserve the acceptance set and be covered by a
  reference test. Do not replace exact crystallographic certification by a
  learned, metric-only, or tolerance-widened shortcut.
- Do not add a method until a small diagnostic identifies the mechanism it is
  intended to repair.
- One-off builders/auditors may be archived after their current artifact is
  qualified; production readers and model code remain active.

## Required validation

At minimum run:

```bash
$PY -m pytest -q
$PY -m ruff check
$PY -m mypy src/gaugeflow/production
$PY scripts/audit_code_redundancy.py
```

Atlas/runtime changes additionally require a no-write CUDA smoke for candidate
counts, finite outputs, reference equivalence, latency, and peak memory.

## Stage-C execution baseline

Stage-B-v1.1 is qualified by the archived MatPES physical and 512-sample A1
retention evaluation.  Stage-C uses three fixed additive roles in rank order:
LeMat geometry-only structure replay, MatPES physical transfer, and Alex
generative replay.  Their gradients are summed once before identical local
AdamW/EMA updates.  Do not restore the former one-owner optimizer, per-parameter
all-reduce, full-model broadcast, or ordinary three-rank data-parallel path.

LeMat replay must not parse or expose energy, force, or stress fields; MatPES is
the sole Stage-C physical-label stream.  Keep row-group/source-local batch
materialization, pinned asynchronous transfer, bounded gradient buckets, and
the frozen role order.  Any execution change must retain exact batch content,
the additive joint gradient, rank-to-rank parameter identity, and interrupted
checkpoint equivalence before a long continuation is authorized.
