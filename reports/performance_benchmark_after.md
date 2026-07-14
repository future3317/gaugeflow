# GaugeFlow unified performance benchmark

Ten warm-up steps and twenty measured optimizer steps use the frozen Gate A panel, resident GPU batch, seed, capacity, and complete 792-candidate definition. Projected time is a linear 400-step estimate, not a replacement for completed training wall time.

| method | sec/step | projected 400-step s | process CPU | GPU | RSS MiB | torch VRAM MiB | vs direct | vs before |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_tensor | 0.0118 | 4.7 | 99.9% | 15.0% | 1430.7 | 19.5 | 0.77x | n/a |
| direct_irrep | 0.0153 | 6.1 | 100.0% | 14.0% | 1471.7 | 19.5 | 1.00x | n/a |
| stabilizer_pooling | 0.0160 | 6.4 | 100.3% | 13.5% | 1472.5 | 19.5 | 1.05x | 233.7x |
| orbit_alignment | 0.0220 | 8.8 | 98.8% | 22.8% | 1523.4 | 34.5 | 1.44x | 231.6x |

Sampling throughput, CUDA-event timing, utilization sample counts, and exact definitions are in `reports/performance_benchmark_after.json`. Short utilization windows are noisy and are reported as diagnostics rather than scientific outcomes.
