# GaugeFlow

GaugeFlow 是面向压电晶体的 tensor-orbit-conditioned 生成模型。当前唯一
production 路径位于 `gaugeflow.production`：元素使用 absorbing categorical
diffusion，分数坐标使用周期平移商上的 wrapped diffusion，晶格使用
log-volume / trace-free log-metric 表示，三阶极性张量条件使用 Stratified
Cartesian Gauge Atlas。项目没有旧 continuous-logit flow、harmonic conditioner
或 FlowMM runtime fallback。

## 当前正式状态

| 部分 | 当前结论 |
|---|---|
| 数学与软件接口 | 已通过：混合状态空间、周期商、晶格 chart、群作用和 Cartesian atlas runtime |
| Trainer / reverse sampler | CUDA 软件闭环已通过；这不是生成质量结论 |
| 数据与群论分解 H0 | 已通过：结构 split、声子/PES 接口、finite-affine/OPD catalogue、真实 occupational occurrence |
| 真实数据 H1a | 已运行并失败：粗粒度 composition/lattice 合格，局部坐标与最近邻分布不合格 |
| 完整 parent blueprint 与 H2--H6 | 尚未开始 |
| Tensor-conditioned generation / oracle / relaxation / DFT / DFPT | 尚未开始，当前不能据此提出材料发现 claim |

项目当前停在 H1a 坐标生成器诊断。P1 cache 已完整构建并独立审计；H1b 和后续
Gate 仍被阻止。当前证据不支持继续增加 reciprocal 输出分支、训练 seed 或步数。

## 当前方法

完整层级表示为：

```text
species-free parent carrier
  -> low-index supercell
  -> exact integer occupation
  -> OPD displacement / invariant strain / bounded residual
  -> ordered child crystal
```

parent space group 是生成先验，不是终态硬约束。若 parent action 在展开节点上的
置换为 `pi_g`，完整整数元素着色为 `a`，则 occupational stabilizer 为

```text
H_occ(a) = {g : a[pi_g(i)] = a[i] for every i}
```

终态子群由 supercell、chemical ordering 和 displacement/strain modes 共同决定：

```text
H(d,a) = G_parent^B ∩ H_occ(a) ∩ intersection_l H_(l,c_l)
```

这修正了“高对称 parent 必须保持终态 species labels”的错误假设。当前真实数据
审计表明，该表示在干净、ordered、stoichiometric、`det(B)<=4` 的作用域内覆盖
`359/1023 = 0.350929` 的材料，并可由独立反序 auditor 完整重建。该结果说明表示
和数据足以进入学习阶段，不说明模型已经学会选择 parent、occupation 或 mode。

详细方法、数据和阶段结论见：

- [当前项目状态（中文）](docs/current_project_status_zh.md)
- [层级对称性破缺设计](docs/hierarchical_symmetry_breaking_v1.md)
- [Cartesian Gauge Atlas](docs/cartesian_stratified_gauge_atlas_v1.md)
- [研究迭代总结](docs/research_iteration_history.md)

## 当前数据

大型数据仅存放在 `E:/DATA/T2C-Flow`，不复制进代码仓库。

- Alex-MP-20：675,204 条结构；当前 child-first split 为
  `540,164 / 67,520 / 67,520`，跨 split 的 formula、exact prototype、
  matcher envelope 和 connected component overlap 均为零。
- PhononDB：10,034 个 compact Hessian/force-constant records；按需计算
  dynamical matrix，不保存 dense q-grid。
- MatPES-PBE：已资格化的 TensorNet 与 QET 仅作为未来离线监督，不做 reverse guidance。
- TensorOrbit-JARVIS-v2：保留给后续 tensor/oracle Gate；当前训练不读取它。

数据问题采用版本化入口清洗：保留 raw source，不修改历史结果，不给模型添加坏数据
fallback。`alex<agm004639609>` 只从未来 parent-occurrence / blueprint 数据入口剔除；
它的 child structure 对 P1 结构训练仍有效，因此不会从 Alex 结构池删除。

## H1a packed cache 与真实训练结论

正式 cache 位于 `E:/DATA/T2C-Flow/processed/gaugeflow_h1a_v1/p1_structure_cache_v1`。
675,204 条源结构全部重建成功，最大 source-equivalence error 为 `8.10e-15 A`，
float32 cache error 为 `2.79e-6 A`。reader 只接受 `qualified=true` 且文件哈希
匹配的 manifest，默认仅返回：

```text
atom_types, frac_coords, lattice, num_nodes
```

material ID、source row、split、formula、prototype、space group 和 Niggli transform
只能出现在离线 audit index，不能进入 denoiser。cache 构建只允许认证的 Niggli
`GL(3,Z)` basis change，无 unreduced fallback。

当前最佳联合 H1a checkpoint 使用完整 540,164 条 train split，20,000 steps、
1,280,000 graph presentations（约 2.37 passes）。它生成有限正体积晶格且无
sampling failure/mask；element marginal JSD 为 `0.0143`、volume Wasserstein 为
`0.0644`、formula uniqueness 为 1.0。但生成最近邻中位数为 `2.172 A`，训练参考
为 `2.698 A`，归一化最近邻 Wasserstein 为 `0.953 > 0.75`，所以 H1a 失败。

一次完整 train pass 的 coordinate-only 预训练也失败：基线 validation 为
`0.54928 > 0.35`。经过独立数学/CUDA 资格化的 signed pairwise reciprocal residual
只改善到 `0.53354`，且 `t=.005` endpoint RMS 为 `0.04494 A > 0.04 A`。该分支
确实活跃但收益不足，已从 production 删除，只在 Git commit `154e6c9` 和报告中
保留。当前模型没有外部预训练权重，也没有启动 tensor/oracle/relaxation/DFT/DFPT。

## 环境

所有报告测试和未来训练使用 WSL 2 Ubuntu-22.04：

```bash
cd /mnt/e/CODE/T2C-Flow/gaugeflow
export PYTHONPATH="$PWD/src"
PY=/home/future04/micromamba/envs/flowmm-t2c/bin/python

$PY -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

资格机器为 PyTorch `2.5.1+cu124`、CUDA 12.4、RTX 4060 Ti 16 GB。Windows
CPU-only torch 不用于报告结果。

## 验证

```bash
$PY -m pytest -q
$PY -m ruff check
$PY -m mypy src/gaugeflow/production
$PY scripts/audit_code_redundancy.py
```

## 仓库原则

- 当前代码就是唯一 runtime；不保留旧模型兼容分支。
- 当前处理数据由 manifest/hash 确认；raw source 保留在外部数据目录。
- 历史失败和旧 runner 只在 Git tag 与 `docs/research_iteration_history.md` 中复现。
- 不把 target CIF、target lattice、material ID、target space group、stabilizer 或
  species mapping 输入 denoiser。
- polar rank-three tensor orbit 使用 `SO(3)`；晶体兼容性才使用显式 parity 的 `O(3)`。
- H1a 通过前不启动完整 blueprint、tensor、oracle 或物理验证。
