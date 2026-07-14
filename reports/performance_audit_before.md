# GaugeFlow performance audit: before optimization

## Scope and frozen protocol

This is the optimization-before baseline for `configs/gate_a_v1.json`. The
eight material IDs, 792 finite-order integer candidates, 400-step Gate A
budget, seed `20260714`, hidden size 64, two message layers, and eight SO(3)
frames were not changed. The already-running `stabilizer_pooling` process was
allowed to finish naturally.

Hardware/software:

- AMD Ryzen 7 7700, 8 cores / 16 logical CPUs
- NVIDIA GeForce RTX 4060 Ti, 16 GiB, driver 591.86
- WSL2 Ubuntu 22.04, Python 3.10
- PyTorch 2.5.1+cu124, CUDA 12.4

The isolated audit ran in `E:/CODE/T2C-Flow/gaugeflow_perf_audit` on branch
`perf-audit-20260714`. It did not edit the main worktree used by the original
run.

## End-to-end observed baseline

The original 400-step `stabilizer_pooling` run completed in approximately
1 h 23 min and wrote `outputs/gate_a_v1/checkpoints/stabilizer_pooling.pt`.
Resource samples during that run showed:

- one logical CPU near 96% continuously;
- GPU utilization fluctuating between 14% and 44%, with frequent 0% samples;
- process RSS rising to about 8.0 GiB;
- `nvidia-smi` process/context memory near 712 MiB.

This is a CPU-dispatch-bound GPU workload: `--device cuda` was honored, but
thousands of tiny operations, Python graph loops, repeated representation
construction, scalar synchronizations, and host/device copies starved the GPU.
It was not a CPU-only training invocation.

## Formal profiler baseline

Each method used 10 warm-up steps and 20 profiled optimizer steps on the exact
eight-record panel. Data were already parsed/Niggli-reduced in memory, matching
the Gate A training script.

| metric | stabilizer_pooling | orbit_alignment |
| --- | ---: | ---: |
| mean sec/step | 3.738 | 5.092 |
| median sec/step | 3.708 | 4.642 |
| min--max sec/step | 3.419--4.221 | 4.344--13.385 |
| mean data wait | 1.11 ms | 0.93 ms |
| peak CPU RSS | 7,021.6 MiB | 7,671.6 MiB |
| peak torch CUDA allocated | 19.61 MiB | 21.84 MiB |
| peak torch CUDA reserved | 24 MiB | 26 MiB |
| profiled copy-like calls / 20 steps | 50,900 | 50,980 |

Cold construction of the complete 792-candidate catalogue took only 0.356 s;
panel CSV/cache/CIF/Niggli preprocessing took 2.803 s. The earlier hypothesis
that catalogue enumeration dominated the 400-step run is therefore rejected.

Profiler artifacts:

- `reports/profiler_before/stabilizer_pooling_before_trace.json`
- `reports/profiler_before/orbit_alignment_before_trace.json`
- `reports/profiler_before/*_before_top20_cpu.txt`
- `reports/profiler_before/*_before_top20_cuda.txt`
- `reports/profiler_before/summary.json`
- `reports/profiler_before/stabilizer_pooling_before.prof`

The two Chrome traces are intentionally not committed automatically; together
they are about 784 MiB. The WSL PyTorch build recorded CUDA runtime launches
but did not expose per-kernel device durations through Kineto/CUPTI. Therefore
the requested CUDA top-20 files contain runtime-dispatch order rather than
trustworthy kernel self-time. GPU utilization and memory are reported from
`nvidia-smi`, and the after-profiler will also use explicit CUDA-event macro
timings. No CUDA-time values are fabricated.

## Top hotspots and time decomposition

For `stabilizer_pooling`, `model.tensor_orbit_rotation` consumed 58.53 s of
self CPU time over 20 steps (82.75%); for `orbit_alignment` it consumed
81.54 s (85.14%). This range was entered 160 times, exactly eight graphs times
20 steps, proving that graphwise Python execution is the primary dispatch
multiplier.

The decisive cProfile result is more specific: in a one-warm-up/one-profile
run, 24 calls to `piezo_to_irreps` spent 13.16 s constructing e3nn
`ReducedTensorProducts` and their change-of-basis object. The change of basis
is constant for `ijk=ikj` but was rebuilt for every graph and step.

Other 20-step counts for `stabilizer_pooling`:

- 45,300 `cudaLaunchKernel` calls;
- 5,900 `cudaMemcpyAsync` calls;
- 15,200 `aten::_to_copy` calls;
- 9,920 `aten::item` / scalar synchronization calls;
- 160 state-derived soft-stabilizer calls, although uniform
  `stabilizer_pooling` should require none.

`stabilizer.periodic_type_self_match` itself accumulated only 0.38 s of CPU
submission time in this top-k-24 implementation; its many surrounding small
operations and synchronization costs are distributed across dispatcher
entries. Data wait is below 0.04% of step time, so adding DataLoader workers is
not justified for the current eight-record in-memory Gate A panel.

## Scientific-definition findings before optimization

1. **The implemented `stabilizer_pooling` is not state-independent.** It calls
   `soft_crystal_stabilizer_actions` for every graph, computes lattice and
   periodic self-match scores, softmaxes them, and transforms every tensor
   frame before applying uniform frame weights. Removing this path is a
   correction to the declared incoherent baseline, not merely a speed trick.
2. **`orbit_alignment` scores only the 24 lowest lattice-error candidates.**
   Although all 792 candidates undergo polar projection, the current
   `max_actions=24` top-k truncation violates the frozen 792-candidate
   requirement. The corrected implementation must retain all 792 candidates.
3. **The SO(3) polar result is not generally static.** For candidate `U`, the
   projected Cartesian action depends on the evolving lattice through
   `A^{-1} U A`. Integer `U` and tensor conversion bases can be buffers, but
   `R_U(A_t)` must be recomputed for `orbit_alignment`; caching it across flow
   states would not be scientifically equivalent.
4. **Static rotation deduplication is unsafe for `orbit_alignment`.** At an
   identity lattice, 792 candidates collapse to 468 polar rotations at
   tolerance `1e-5` (324 duplicate pairs). On a generic triclinic lattice all
   792 projected rotations are distinct. Dynamic grouping would introduce a
   state-dependent discrete partition, so the main protocol will preserve all
   candidate IDs and weights.
5. **The fixed tensor Cartesian/irrep change of basis is rebuilt repeatedly.**
   Caching this exact matrix and batching its matmul is strictly equivalent and
   addresses the largest measured hotspot.

Because items 1 and 2 correct the existing forward definition, old Gate A
checkpoints cannot be mixed with optimized checkpoints. All four methods must
be rerun from scratch under one final code revision. Nothing in this report is
a Gate A pass, a full 4,000/499/499 experiment, or a DFT/DFPT result.
