# GaugeFlow unified performance benchmark

Ten warm-up steps and twenty measured optimizer steps use the frozen Gate A panel, resident GPU batch, seed, capacity, and complete 792-candidate definition. Projected time is a linear 400-step estimate, not a replacement for completed training wall time.

| method | sec/step | projected 400-step s | process CPU | GPU | RSS MiB | torch VRAM MiB | vs direct | vs before |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct_irrep | 0.0124 | 5.0 | 82.6% | 38.3% | 1438.4 | 262.7 | 1.00x | n/a |

Sampling throughput, CUDA-event timing, utilization sample counts, and exact definitions are in `reports/performance_benchmark_after.json`. Short utilization windows are noisy and are reported as diagnostics rather than scientific outcomes.
