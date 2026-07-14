# GaugeFlow performance, data-pipeline, and scientific-equivalence audit

## Technical summary

GaugeFlow was running on CUDA; the apparent CPU-only behavior came from Python
graph loops, repeated e3nn basis construction, thousands of small kernel
launches, synchronizations, and host copies that starved the GPU. On the frozen
eight-crystal Gate A panel, the corrected resident-batch implementation now
runs `orbit_alignment` at 0.0220 s/optimizer step and `stabilizer_pooling` at
0.0160 s/step outside profiler overhead. These are approximately 232x and 234x
faster than the old measured paths, while `orbit_alignment` now evaluates all
792 candidates rather than the old accidental top-24 truncation.

The audit does **not** establish Gate A. After rerunning all four 400-step
checkpoints from scratch, every conditioning path was sensitive to condition
shuffling and GaugeFlow achieved a representative velocity error of 0.0452
versus 0.3522 for raw tensor conditioning. However, its generated-target
between/within distance ratio was only 1.0066 against the pre-registered 1.2
minimum. The oracle-free supporting status therefore failed. External oracle
orbit error and the registered physical audit are also still absent.

Separately, TensorOrbit-JARVIS v1 is complete and internally valid but is not
formula-disjoint: 165 reduced-formula groups affecting 672 of 4,998 rows cross
splits, and 56 cross-split near-duplicate structure pairs were found. This is a
high-severity generalization-benchmark risk, not the cause of the eight-record
training-panel failure.

## Scope and frozen definitions

- Protocol: `configs/gate_a_v1.json`
- Panel: 8 fixed JARVIS IDs, 2--6 atoms, including one exact physical zero
- Training: 400 steps, batch 8, hidden size 64, 2 layers, 8 orbit frames
- Seed: `20260714`
- Methods: raw tensor, Cartesian direct-irrep, uniform stabilizer pooling, and
  coherent orbit alignment
- Integer proposal catalogue: all 792 frozen candidate IDs; no top-k, random
  sampling, or candidate-budget reduction
- Hardware: Ryzen 7 7700, RTX 4060 Ti 16 GiB, PyTorch 2.5.1+cu124 under WSL2

## The GPU was active but underfed

The old `stabilizer_pooling` run used one logical CPU near 96%, showed 14--44%
intermittent GPU utilization with frequent 0% samples, occupied about 8 GiB
RSS, and exposed about 712 MiB through `nvidia-smi`. Formal profiling measured
3.738 s/step for pooling and 5.092 s/step for alignment. Data wait was about
1 ms and the 792-candidate catalogue took only 0.356 s cold, rejecting both
data loading and catalogue enumeration as primary bottlenecks.

The main hotspot was constant representation work: 24 calls to
`piezo_to_irreps` spent 13.16 s rebuilding the same e3nn
`ReducedTensorProducts` basis. Twenty old pooling steps launched roughly
45,300 CUDA kernels, performed about 5,900 asynchronous copies, and entered
the tensor-orbit loop once per graph per step. The GPU executed kernels, but
their duration was shorter than ordinary `nvidia-smi` refresh intervals.

After optimization, explicit CUDA-event measurements were 11.8--22.0 ms per
step in the unified benchmark. The process remains near one CPU core because
the model and panel are tiny. Low continuous GPU utilization is expected for
this diagnostic and is not evidence of CPU fallback.

## Unified after benchmark

Ten warm-up and twenty measured optimizer steps used the same resident CUDA
batch and complete candidate definition.

| method | sec/step | projected 400-step time | slowdown vs direct | torch peak VRAM | sampling graphs/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw tensor | 0.0118 | 4.7 s | 0.77x | 19.5 MiB | 193.1 |
| direct irrep | 0.0153 | 6.1 s | 1.00x | 19.5 MiB | 162.0 |
| stabilizer pooling | 0.0160 | 6.4 s | 1.05x | 19.5 MiB | 139.6 |
| orbit alignment | 0.0220 | 8.8 s | 1.44x | 34.5 MiB | 62.4 |

The incremental alignment cost relative to direct-irrep is about 8.45 microseconds
per candidate per optimizer step when the aggregate difference is divided by
792. The preprocessing cache took 50.28 s to build once and is amortized across
runs. Formal profiler overhead produced 0.0273 and 0.0533 s/step for pooling
and alignment respectively; those numbers are retained separately so profiler
instrumentation is not confused with ordinary throughput.

`torch.compile` was tested only as an optional optimization. Direct-irrep
steady-state steps improved from 0.0153 to 0.0124 s in a short run, but two lazy
compile warm-up steps cost 28.4 s, VRAM rose to 263 MiB, and a new sampling
shape triggered an 8.56 s compilation. It is therefore not enabled: dynamic
PyG/sampling shapes make it counterproductive for Gate A, and no method was
changed to accommodate compilation.

## Changes classified by scientific status

### Strictly equivalent optimizations

- Cached the fixed Cartesian/irrep change-of-basis and replaced repeated e3nn
  object construction with batched matrix multiplication.
- Vectorized graph/frame/edge dimensions and integer-candidate filtering.
- Registered periodic translations and other fixed tensors as buffers.
- Batched all 792 candidates and used chunking solely to cap peak memory; every
  candidate retains the same score and normalization.
- Precomputed fixed condition orbits and probes; state-derived automorphism
  evidence and local bond queries remain dynamic.
- Made the repeated Gate A panel a resident GPU batch and added a versioned
  CIF/Niggli/tensor preprocessing cache.

Regression tests cover candidate ID identity, cached/uncached condition token,
velocity, flow loss, tensor and parameter gradients, and one optimizer update.
The batched posterior matches an unbatched reference within FP32 tolerances.

### Scientific-definition and numerical corrections

- `stabilizer_pooling` is now genuinely uniform and state-independent instead
  of reusing the alignment posterior.
- `orbit_alignment` now scores all 792 proposals instead of top-24.
- The SVD polar factor was replaced with seven scaled Newton steps. Forward
  rotations agree with the proper SVD factor within about 7.2e-7, while the
  Newton path has finite lattice gradients at repeated singular values where
  SVD backward produced NaNs.
- The direct-irrep baseline uses exact Cartesian tensor contractions with bond
  geometry and no spherical harmonics; raw component concatenation remains a
  separate deliberately non-equivariant control.

These corrections change the old forward definition or repair undefined
backward behavior. Old checkpoints are not scientifically interchangeable,
which is why all four Gate A runs were restarted.

### Not promoted

- No top-k, candidate deduplication, random candidate sampling, AMP on polar or
  posterior operations, learned replacement posterior, or `torch.compile`
  dependency entered the active method.
- Full generated cell-basis consistency remains a Gate C outcome rather than
  a claimed unit-test result. Low-level periodic Cartesian edge geometry and
  tracked Niggli quotient round trips are covered and pass.

## Data quality and leakage

All 4,998 IDs join one-to-one across CSV, CIF, target cache, and split
manifest. There are no duplicate IDs, missing targets, non-finite tensors, or
Reynolds failures. Exact-zero targets number 1,853/221/223 in train/val/test.
The 77 apparent Niggli non-idempotences are alternate equivalent boundary-cell
representatives; the tracked lattice/fractional quotient round trip remains at
machine precision. Three FP32 Voigt round trips reach 1.79e-7 and pass the
declared 2e-7 tolerance.

The serious defect is the v1 split. It has 165 cross-split reduced-formula
groups covering 672 rows and 56 StructureMatcher near-duplicate pairs. V1 was
left unchanged. An inactive, formula-disjoint 4,000/499/499 candidate is stored
at `artifacts/tensororbit_jarvis_formula_grouped_candidate_v2/splits.json`
with candidate hash
`e2f5c08014b9c62836523d85e80e79e498fb9c7ba2bfc2273564e17327a12e5e`.
Activating it requires a new protocol version.

## Gate A result and current blockers

| supporting check | result | threshold/status |
| --- | ---: | --- |
| all methods condition-shuffle gap | pass | median gap >= 0.02 |
| GaugeFlow representative velocity error | 0.0452 | pass, <= 0.15 |
| error ratio vs raw tensor | 0.128 | pass, <= 0.5 |
| generated target between/within ratio | 1.0066 | **fail**, required >= 1.2 |
| condition permutation feature shift | 2.0403 | pass, required >= 0.02 |

The immediate Gate A blocker is thus model-level generated target separation,
not data-loader throughput or CUDA placement. The v1 split leakage is a second,
independent blocker for future generalization benchmarks. Full Gate A also
lacks a qualified frozen tensor-oracle ensemble, the training-set orbit tensor
error distribution, and the registered physical micro-audit; this audit does
not claim that those experiments ran.

## Recommended next steps

1. Keep Gate A frozen and diagnose why a condition-sensitive velocity field
   produces weakly separated samples. Inspect per-target trajectories and
   conditioning-gradient magnitudes before altering the method.
2. Add the qualified external oracle ensemble so training-panel orbit tensor
   error can distinguish “features move” from “the requested physical tensor
   is controlled.”
3. Do not launch 4,000/499/499 on v1. Review and formally activate the v2 split
   only through a new versioned protocol.
4. Do not start Gate B/C/D, relaxation, DFT, or DFPT until Gate A's failed
   target-separation control is resolved and rerun without threshold changes.

## Evidence inventory

- Before profiler: `reports/performance_audit_before.md` and
  `reports/profiler_before/summary.json`
- After profiler: `reports/profiler_after/summary.json` and top-20 tables
- Unified throughput: `reports/performance_benchmark_after.json`
- Optional compile check: `reports/torch_compile_direct_irrep.json`
- Gate A supporting diagnostics: `reports/gate_a_supporting_after.json`
- Data audit: `reports/data_quality_audit.md`, row CSV, structural-match CSV,
  and manifest
- Preprocessing manifest:
  `artifacts/tensororbit_jarvis_v1_preprocessed_v1.pt.manifest.json`

Large profiler traces and the four diagnostic checkpoints are retained locally
but intentionally excluded from source control. No Gate A pass, full benchmark,
relaxation, DFT, DFPT, or discovery claim is made.
