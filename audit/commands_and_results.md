# Commands and Results

> Provenance warning: these original commands used a non-authoritative Windows
> environment and globally suppressed warnings. Formal WSL/CUDA re-verification
> is recorded in `CODEX_VERIFICATION.md`.

Audit date: 2026-07-16
Code root: `E:\CODE\T2C-Flow\gaugeflow_perf_audit`
Environment: `D:\Anaconda\envs\EGNN` (Python 3.11, torch 2.11.0+cpu reported by interpreter; CUDA available via `NVIDIA GeForce RTX 4060 Ti`)

All commands were run from `E:\CODE\T2C-Flow\gaugeflow_perf_audit` unless stated otherwise. No source files were modified.

## 1. Repository and environment inspection

```bash
pwd
# /e/CODE/T2C-Flow/gaugeflow_perf_audit

ls -la
# shows .git, src/, tests/, scripts/, configs/, docs/, reports/, etc.

python -c "import torch; print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
# cuda True
# device NVIDIA GeForce RTX 4060 Ti
```

## 2. Test suite execution

```bash
python -W ignore -m pytest tests -q --tb=short
# 198 passed in 59.96s
```

Targeted production tests:

```bash
python -W ignore -m pytest tests/test_paper_s0_production.py tests/test_paper_s0_2_scalability_symmetry.py tests/test_paper_s0_4_cartesian_atlas_prior.py tests/test_cartesian_gauge_atlas.py -q --tb=short
# 48 passed in 46.52s
```

BF16 CUDA production path:

```bash
python -W ignore -m pytest tests/test_paper_s0_4_cartesian_atlas_prior.py::test_bf16_autocast_production_path_is_finite_and_uses_4032_candidates -q --tb=short -s
# 1 passed in 4.94s
```

20-site CUDA wrapped quotient scalability:

```bash
python -W ignore -m pytest tests/test_paper_s0_2_scalability_symmetry.py::test_scalable_wrapped_quotient_handles_twenty_sites_and_triclinic_metrics -q --tb=short -s
# 1 passed in 35.14s
```

## 3. Static checks

Initially missing; installed during audit:

```bash
python -m ruff --version
# No module named ruff

python -m mypy --version
# No module named mypy

pip install ruff mypy
# Successfully installed ruff-0.15.21 mypy-2.3.0
```

After installation:

```bash
python -m ruff check src tests scripts
# All checks passed!

python -m mypy src/gaugeflow/production
# Success: no issues found in 14 source files
```

## 4. Coordinate-head gradient connectivity

```bash
python -W ignore - <<'PY'
import torch, sys
from pathlib import Path
sys.path.insert(0, str(Path('src')))
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser

torch.manual_seed(0)
device = 'cpu'
model = HybridCrystalDenoiser(hidden_dim=64, vector_dim=8, layers=2, radial_dim=8, atlas_residual_circle_samples=4).to(device).train()

# ... build valid batch with trace-free log_shape ...
args = make_batch()
out = model(*args)
score = out.coordinate_cartesian_score
target = (score ** 2).sum()
target.backward()
cond_grad = args[6].grad

# finite difference
eps = 1e-4
base = target.item()
with torch.no_grad():
    tensor_condition2 = args[6].clone()
    tensor_condition2[0, 0] += eps
    out2 = model(*args2)
    perturbed = (out2.coordinate_cartesian_score ** 2).sum().item()
    fd = (perturbed - base) / eps

print('condition grad norm', cond_grad.norm().item())
print('finite diff', fd)
print('analytic', cond_grad[0, 0].item())
PY
```

Result:

```
condition grad norm 0.3639379143714905
finite diff -7.450580596923828e-05
analytic -4.689298657467589e-05
score diff (cond_present=1 vs 0) 0.5387298464775085
```

Conclusion: coordinate head receives condition signal; finite-difference and analytic gradients agree in sign and order of magnitude.

## 5. Atlas candidate counts and timing

```bash
python -W ignore - <<'PY'
import torch, time, sys
from pathlib import Path
sys.path.insert(0, str(Path('src')))
from gaugeflow.production.cartesian_gauge_atlas import StratifiedCartesianGaugeAtlas
from gaugeflow.tensor import piezo_from_irreps

device = torch.device('cuda')
dtype = torch.float64
atlas = StratifiedCartesianGaugeAtlas(64, residual_circle_samples=8).to(device).double().eval()

torch.manual_seed(0)
tensor = piezo_from_irreps(torch.randn((1, 18), dtype=dtype, device=device))[0]
query = torch.randn((2, 3, 3, 3), dtype=dtype, device=device)
generic = torch.diag(torch.tensor([1.0, 1.3, 1.7], dtype=dtype, device=device))
axial = torch.diag(torch.tensor([1.0, 1.0, 1.7], dtype=dtype, device=device))
isotropic = torch.eye(3, dtype=dtype, device=device)

def bench(name, cov, n=10):
    with torch.no_grad():
        frame = atlas._frame_data(cov, directional=True)
        measure = atlas._candidate_measure(frame, frame)
        for _ in range(3):
            rotated = atlas._rotate_rank_three(tensor, measure.rotations)
            score = torch.einsum("fijk,cijk,c->f", rotated, query, atlas.score_channel.to(query))
            torch.softmax(score + measure.prior.log(), dim=0)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            rotated = atlas._rotate_rank_three(tensor, measure.rotations)
            score = torch.einsum("fijk,cijk,c->f", rotated, query, atlas.score_channel.to(query))
            torch.softmax(score + measure.prior.log(), dim=0)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / n * 1000
        print(f'{name}: raw={measure.raw_count}, unique={measure.rotations.shape[0]}, {dt:.2f} ms/forward')

bench('generic', generic)
bench('axial', axial)
bench('isotropic', isotropic)
PY
```

Result:

```
generic: raw=4032, unique=4032, 0.60 ms/forward
axial: raw=258048, unique=16128, 1.10 ms/forward
isotropic: raw=4032, unique=4032, 0.60 ms/forward
```

Note: these are single-graph FP64 manual-pool timings on the audit GPU. The official S0.4 full-forward latency is `41.89 ms` (`README.md:64`).

## 6. Atlas precision: FP64 vs FP32 vs BF16

```bash
python -W ignore - <<'PY'
import torch, sys
from pathlib import Path
sys.path.insert(0, str(Path('src')))
from gaugeflow.production.cartesian_gauge_atlas import StratifiedCartesianGaugeAtlas
from gaugeflow.tensor import piezo_from_irreps

device = torch.device('cuda')
torch.manual_seed(0)

# FP64 reference
atlas64 = StratifiedCartesianGaugeAtlas(64, residual_circle_samples=8).to(device).double().eval()
tensor64 = piezo_from_irreps(torch.randn((1, 18), dtype=torch.float64, device=device))[0]
query64 = torch.randn((2, 3, 3, 3), dtype=torch.float64, device=device)
cov64 = torch.diag(torch.tensor([1.0, 1.3, 1.7], dtype=torch.float64, device=device))
with torch.no_grad():
    frame64 = atlas64._frame_data(cov64, directional=True)
    measure64 = atlas64._candidate_measure(frame64, frame64)
    rotated64 = atlas64._rotate_rank_three(tensor64, measure64.rotations)
    score64 = torch.einsum("fijk,cijk,c->f", rotated64, query64, atlas64.score_channel.to(query64))
    post64 = torch.softmax(score64 + measure64.prior.log(), dim=0)
    pooled64 = torch.einsum("f,fijk->ijk", post64, rotated64)

# FP32
atlas32 = StratifiedCartesianGaugeAtlas(64, residual_circle_samples=8).to(device).float().eval()
tensor32, query32, cov32 = tensor64.float(), query64.float(), cov64.float()
with torch.no_grad():
    frame32 = atlas32._frame_data(cov32, directional=True)
    measure32 = atlas32._candidate_measure(frame32, frame32)
    rotated32 = atlas32._rotate_rank_three(tensor32, measure32.rotations)
    score32 = torch.einsum("fijk,cijk,c->f", rotated32, query32, atlas32.score_channel.to(query32))
    post32 = torch.softmax(score32 + measure32.prior.log(), dim=0)
    pooled32 = torch.einsum("f,fijk->ijk", post32, rotated32)

# BF16 autocast
atlasbf = StratifiedCartesianGaugeAtlas(64, residual_circle_samples=8).to(device).float().eval()
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    framebf = atlasbf._frame_data(cov32, directional=True)
    measurebf = atlasbf._candidate_measure(framebf, framebf)
    rotatedbf = atlasbf._rotate_rank_three(tensor32, measurebf.rotations)
    scorebf = torch.einsum("fijk,cijk,c->f", rotatedbf, query32, atlasbf.score_channel.to(query32))
    postbf = torch.softmax(scorebf + measurebf.prior.log(), dim=0)
    pooledbf = torch.einsum("f,fijk->ijk", postbf, rotatedbf)

def rel_err(a, b):
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-12))

print('FP32 vs FP64 pooled rel err', rel_err(pooled32, pooled64))
print('BF16 vs FP64 pooled rel err', rel_err(pooledbf.float(), pooled64))
print('FP32 posterior max abs diff', float((post32 - post64).abs().max()))
print('BF16 posterior max abs diff', float((postbf.float() - post64).abs().max()))
print('unique counts', measure64.rotations.shape[0], measure32.rotations.shape[0], measurebf.rotations.shape[0])
PY
```

Result:

```
FP32 vs FP64 pooled rel err 1.3776816463102804e-07
BF16 vs FP64 pooled rel err 0.004625578704295942
FP32 posterior max abs diff 5.88942759907618e-08
BF16 posterior max abs diff 0.0014114559345032585
unique counts 4032 4032 4032
```

## 7. Production/legacy import boundary

```bash
grep -r "from gaugeflow.flow import\|from gaugeflow.discrete import\|torus_logmap\|GaugeFlowVectorField\|RiemannianCrystalFlowMatcher" src/gaugeflow/production/
# src/gaugeflow/production/s0_audit.py:129: for forbidden in ("from gaugeflow.flow import", ...)
# No legacy imports found in production runtime modules.

grep -r "TODO\|FIXME\|NotImplementedError\|XXX" src/gaugeflow/production/
# No matches found
```

## 8. Blueprint sampler search

```bash
grep -ri "blueprint" .
# src/gaugeflow/production/space_group_router.py:1:"""Full-O(3) piezoelectric compatibility for the symmetry blueprint."""
# reports/paper_architecture_compliance_v1.md:19:| Complete Wyckoff autoregressive blueprint | Not implemented in this S0 change | S2 locked |
# src/gaugeflow/production/equivariant_denoiser.py:188: determined by the sampled space-group blueprint
```

Conclusion: no blueprint sampler implementation exists.

## 9. Training/sampling entry-point verification

```bash
sed -n '199,254p' scripts/train.py
# model = GaugeFlowVectorField(...)
# terms = RiemannianCrystalFlowMatcher(...).loss(model, batch, ...)

sed -n '47,68p' scripts/sample.py
# model = GaugeFlowVectorField(...)
# state = RiemannianCrystalFlowMatcher().sample(model, batch, ...)
```

Conclusion: both entry points use the legacy architecture.

## 10. Data-leakage field inspection

```bash
sed -n '213,250p' src/gaugeflow/data.py
# Data(atom_types=..., frac_coords=..., lattice=..., piezo_irreps=...,
#      condition_present=..., niggli_transform=..., response_stratum=...,
#      zero_response=..., material_id=..., num_nodes=...)
```

`HybridCrystalDenoiser.forward` accepts none of these fields. `s0_audit.py:20-28` explicitly forbids target metadata in the signature.

## 11. Warning inventory

```bash
python -W ignore -m pytest tests --collect-only -q 2>&1 | tail -n 20
# warnings summary:
# 66 torch.jit.script deprecation warnings
# 43 TorchScript instance-annotation warnings
```

## Reproducibility notes

- All tests were run with `-W ignore` to suppress warning pollution in output.
- The active Python environment is `D:\Anaconda\envs\EGNN`. `ruff` and `mypy` were installed into this environment during the audit.
- CUDA is available; no training was performed. Only no-training diagnostics and the existing test suite were executed.
- Manual diagnostics used fixed seeds (`torch.manual_seed(0)`) and small model dimensions (`hidden_dim=64, vector_dim=8, layers=2`) to keep runtime short.
