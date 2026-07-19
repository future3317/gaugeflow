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
Gate 仍被阻止。局部算子筛选已经收口；下一步只允许用中噪声 oracle、残差频谱和
frozen low-k probe 判断是否确有 reciprocal 全局缺口，不增加训练 seed 或步数。

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

后续固定状态审计进一步定位了坐标学习失败，而不是把小面板结果误当成正式训练。
在平移商的 30 个物理输出方向上，完整模型 Jacobian 与最终 affine readout 都是
满秩 `30/30`，所以当前 head 并未遗漏某个坐标方向；但谱的 condition number 约为
`2.3e7--3.5e7`，entropy effective rank 只有 `2.2` 左右。精确 Helmert quotient
readout 可将一个固定状态拟合到 `5.39e-8` MSE，却需要 `2079.20` 的参数更新，而
初始 readout 范数只有 `0.80036`。这说明主要问题是严重相关、尺度失衡且超出局部
曲率半径的优化几何，不是数据损坏、解析路径不闭合或缺少输出方向。

固定特征的 exact readout 在 1/4 个状态上可精确拟合，但 16/64 个状态的最优 MSE
分别为 `0.09947/0.55232`，证明真实修复仍需 backbone 学出随状态变化的特征。
graphwise unit scaling、未正则 variable projection、screened quotient Laplacian 和
单独 `1024x` function-preserving readout scaling 均已按冻结标准否决，并从 active
production/runtime/config/test 入口删除。它们只保留在报告与 Git 历史中；当前唯一
production 模型仍是简洁的原坐标 head。

组合候选也已在训练前否决。固定 `1024x` 缩放后的 16-state exact solution 范数为
`8.894`，FP32 MSE 为 `0.099467`；但 BF16 MSE 为 `10.9886`（FP32 的 `110.47x`），
backbone gradient norm 为 `23468.3`（FP32 的 `6033.9x`），梯度方向余弦为
`-0.1572`。vector/edge 分量存在 `32.31x` 相消，说明缩小存储参数没有缩小等效函数
权重。该实验没有执行 optimizer step；scaled variable projection 不再是 active 候选。

进一步的单分支审计也没有支持直接删支。显式 Helmert 商上，vector-only 和 edge-only
在单状态均为物理满秩 `30/30`；但 vector-only 的 16-state FP32 MSE 为 `0.56437`，
edge-only 为 `0.13474 > 0.12`，且 edge-only BF16 MSE 为 `10.2160`、梯度方向余弦
为 `-0.1419`。因此两支局部都完整，却共同承担跨状态特征；删除任一支都会丢失必要
拟合或保留数值病态。该审计同样执行零 optimizer step，production 保持不变。

固定的 graph-equal block Gram--Schmidt 也已在训练前否决。它把加权 Gram 条件数
精确降到 `1.000000004`、把参数范数降到 `3.23`，并保持 FP32 MSE `0.09946`；但等效
原始权重仍为 `9108`，BF16 MSE 为 `9.77`、梯度 norm 为 `14670.5`、FP32/BF16
梯度余弦仅 `0.128`。因此后验 readout 换坐标不能消除已形成特征的量化放大。下一
机制必须在最终 readout 之前直接生成紧凑、尺度受控的 Cartesian coordinate carrier。

该上游候选随后通过零训练、零 target 的算子资格。它用 16 个一阶向量矩和二阶 STF
矩构造 `(m,Qm,Q^2m)`，与原 32-channel vector stream 合成 80 个有界 Cartesian
carriers。16 个真实状态全部达到完整 translation-quotient rank，最坏条件数为
`1.47e4`；`O(3)` covariance error 为 `6.76e-6`。BF16/FP32 carrier 余弦为
`0.99598`，probe-gradient norm 比为 `1.0012`、方向余弦为 `0.99269`；12,192-edge
面板耗时 `3.04 ms`、增量显存 `11.61 MiB`。该结果只允许单独的 production 集成
资格化，尚未执行任何 target fit 或训练。

第一次 clean production 集成的超大梯度随后被定位为 index type 错误，而不是
carrier 本身不稳定：对行坐标 `r=fL`，reverse sampler 消费的是 tangent drift，正确
变换为 `v_r=v_fL` 与 `v_f=v_rL^-1`；旧 `L^T` 路径生成的是 covector，却被静默当作
vector 更新坐标。production 现仅保留修正后的 tangent 路径。geometry-sensitive
message blocks、edge encoder 和 Cartesian carrier 使用固定 FP32，图/边归约使用
线性复杂度的 target-contiguous `segment_reduce`，无运行时排序或精度 fallback。

最终零训练 CUDA 资格在 RTX 4060 Ti 上达到 `516.03 graphs/s / 185.73 MiB`；重复
误差为零，BF16/FP32 输出余弦为 `0.999806`、loss-gradient 余弦为 `0.997593`，
GL(3,Z)/O(3)、平移、置换、round-trip 和 atlas bypass 全部通过。随后 seed 5705
在完整 540,164 条 train split 上完成恰好一遍、8,441-step tangent coordinate-only
预训练。固定 validation 从 `34.43436` 降至 `24.24037`，比值 `0.70396 > 0.5`；
`t=.005` endpoint RMS 为 `0.04207 A > 0.04 A`。`t=.1` teacher-forced RMS
`0.06143 A`、从 `t=.1/.2` 开始的 rollout RMS `0.06589/0.09861 A`、零 sampling
failure 和零 tensor candidate 均通过。该结果比旧 covector 的 `0.05672 A` 明显改善，
但仍按冻结门槛失败，因此不初始化 joint model，也不增加 seed/steps。

终态 checkpoint 的只读 readout-span 审计进一步排除了“最后一个线性 head 没收敛”。
80-channel carrier 在固定 train/validation 面板均为满秩，但 train 上离线最优的单一
head 只把 train loss 降低 `5.23%`，并使 validation loss 增加 `3.46%`。即使直接用
validation 标签离线求 oracle head，也只解释 `49.61% < 75%` 的目标能量；该上限在
`t=.005--.1` 的两个 noise replicates 中始终约为 `0.47--0.53`。oracle span 从初始化
提高 `44.94` 个百分点，说明 backbone 确实在学习，但当前跨状态 carrier family 仍
不足，正式归因为 `backbone_span_limited`。下一候选必须只改变 feature formation，例如用等变标量状态低秩地自适应混合
现有 Cartesian carriers；不再调整全局线性 head、训练步数或 seed。

当前 production 使用一个紧凑的 factorized Cartesian angular-moment backbone。
每层维护 64 维 persistent scalar edge state，用 8 个
通道形成

```text
m_jc = sum_(k->j) a_kc n_kj / sqrt(d_j),
Q_jc = sum_(k->j) b_kc (n_kj n_kj^T-I/3) / sqrt(d_j),
eta_1 = n_ij . m_jc,       eta_2 = n_ij^T Q_jc n_ij.
```

展开后它等于一阶/二阶低阶 triplet angular kernel，但实现只保留 edge/node-leading
张量，并把 3+6 个 Cartesian moment 分量合并为一次 target-contiguous reduction。
复杂度和存储为 `O(E*C)`，不构造 `sum_j d_j^2` 个 triplet。periodic
self-image 作为独立不变量输入；未经归一化的 `eta_1/eta_2` 直接进入 residual，避免
再次静默删除局部配位幅值。新增 466,944 个参数，总计 4,948,281。后续因果审计发现
串联零投影会延迟内部算子的首步梯度，当前 residual 输出统一使用固定 `1e-2` 小非零
正交初始化，并在每层用当前 node/vector/time/context 刷新 persistent edge state。
它没有旧 runtime 分支或初始化 fallback。FP64 显式 triplet
参考、O(3)（含反射）、节点/边置换、平移、GL(3,Z)、有限梯度和完整 CPU 回归均已
通过。正式 RTX 4060 Ti 资格为 `489.10 graphs/s / 182.86 MiB`，BF16/FP32
output/gradient cosine 为 `0.999916/0.999038`，零 tensor candidates。

该 backbone 的一次完整训练把 validation ratio 从 corrected-tangent 基线的
`0.70396` 改善到 `0.63864`，且 `t=.005` endpoint RMS 首次达到 `0.03916 A`；
但 ratio 仍未通过 `0.5`。固定 256-graph 因果审计排除了短程 RBF、self-image、degree
aggregation 和单一元素对作为主因，并发现 raw Cartesian MSE 与 atom count 的相关性
为 `0.579--0.643`；除以 `V^(1/3)` 后降至 `0.163--0.231`。因此 active coordinate
loss 使用数学等价的无量纲 chart

```text
v_tilde = V^(-1/3) v_r,    v_r = V^(1/3) v_tilde,
```

而 fractional torus path、`v_f=v_r L^-1` 和 reverse sampler 不变。该 chart 资格为
`350.13 graphs/s`，一遍训练将 validation ratio 进一步降到 `0.58940`；`t=.1`
teacher-forced 与 `t=.1/.2` rollout 分别为 `0.05675/0.05963/0.08444 A`，零失败，
但 `t=.005=0.040084 A` 和 ratio 仍按原阈值失败。

三阶 STF factorized moment 在相同 seed/steps/data 上只把 ratio 改善到
`0.57240`，虽使 `t=.005` 降到 `0.03938 A`，却把训练吞吐从约 `273` 降到约
`221--235 graphs/s`，仍未通过 Gate。三阶分支因此不在 active runtime；代码保留
更简洁的 `l<=2` operator。

随后 dynamic edge refresh 将 ratio 降到 `0.54417`，但仍未通过。固定硬 TopK triplet
为 `0.56794`，更差且对 noisy-neighbor 排序不连续。未平衡 induced R=8 为 `0.54583`，
深层最大 slot mass 达 `0.951`。最后的固定六次 balanced-transport R=8 为 `0.53314`，
相对 dynamic 只改善 `0.01103 < 0.02`；关闭其分支会使 validation loss 恶化 `82.61%`，
说明分支被使用，但最终最大 slot mass `0.19579`、最小表示 effective rank `1.351`、
最大 inter-slot cosine `0.99974`，仍未形成稳定的低秩邻域分解。TopK/induced/R16 与
matched-initialization 实验入口已从 active code 删除，只在本报告、研究历史和 Git 中
保留。H1a 仍失败，不初始化 joint model，也不进入 H1b 或 tensor/oracle/物理计算。

## 训练图与采样加速设计

训练 JSONL/评价 JSON 是数值真源；报告末尾可生成便于阅读的 PNG 与矢量 PDF：

```bash
$PY scripts/plot_h1a_training_diagnostics.py \
  --run /mnt/e/DATA/T2C-Flow/runs/<protocol>/seed_<seed> \
  --report reports/<protocol>
```

图中保留原始 training loss、固定 EMA validation、不同噪声时间的 score/endpoint
误差、rollout 分位数，以及 slot checkpoint/layer/time 热图，不用过度平滑替代原数据。
TensorBoard 不是 production 依赖。少步采样的只读设计见
`docs/chart_consistent_fast_sampling.md`：先做 common-random-number NFE/grid 审计，
只有确认粗时间跳跃是主误差后，才资格化周期平移商、离散元素和晶格 chart 各自正确的
finite transition-map distillation；当前不加速一个尚未通过 H1a 的生成器。

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
