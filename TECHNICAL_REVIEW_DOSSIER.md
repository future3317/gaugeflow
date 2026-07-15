# GaugeFlow 技术审查交接文档

**用途：** 本文是供独立研究者或外部模型进行数学、物理、数据管线与工程审查的自包含说明。它刻意保留负结果，并严格区分：

1. 当前实际运行的生成模型；
2. 为定位失败而运行过、但不能当作论文方法的诊断分支；
3. 已写好数学原语、但尚未接入训练的计划；
4. 已冻结的历史阈值与失败结论。

它不是论文摘要，也不是“模型已经成功生成指定张量晶体”的声明。截止 **2026-07-15**，GaugeFlow 没有通过条件生成的 Gate A；没有完成 4,000/499/499 生成 benchmark；没有运行 relaxation、DFT 或 DFPT。

## 1. 快速结论与审查边界

### 1.1 项目想解决什么问题

目标是学习条件生成分布

\[
p_\theta(X\mid [e]),
\]

其中 (X) 是周期晶体的原子类别、分数坐标和晶格，(e\in\mathbb R^{3\times3\times3}) 是压电三阶 polar tensor，且满足应变指标对称性 (e_{ijk}=e_{ikj})。方括号 ([e]) 表示 proper rotation (SO(3)) 下的 tensor orbit，而不是一个任意选择的笛卡尔 representative。

论文/方法层面的核心想法是：不应把张量的 18 个分量直接拼到模型输入；应当把它作为本构 response field，与当前生成的周期几何共同对齐，再驱动晶体 flow 的 vector field。实现名称为 `orbit_alignment`（GaugeFlow）。

### 1.2 当前最重要的证据

- Gate A 的 8 个真实训练样本上，四种方法都对 condition shuffle 有速度场反应；GaugeFlow 的随机 tensor representative velocity consistency 明显优于 raw-tensor（0.0452 vs 0.3522）。
- 但共同随机噪声采样后，GaugeFlow 的 generated target between/within ratio 仅 **1.0066**，低于冻结阈值 **1.2**。不同 condition 没有形成足够可分离的生成结构分布。
- 更强的 endpoint-ID 控制也没有通过：即使把 tensor condition 换成“这是 BN / 这是 InN”的二分类 one-hot，当前基座仍无法稳定生成正确的 119 类元素 composition/site assignment。
- A4.0 的解析路径闭环是正确的：真实解析 velocity 加当前 Euler sampler 可以恢复 endpoint。因此主要问题不是 flow 时间方向、Euler 正负号、坐标 wrap 或 lattice log/exp 的显式 bug。
- A10/A11.0 进一步表明：剩余 atom-site 错误既不只是 CIF 行顺序，也不应由确定性 geometry-only scalar decoder 在这两个端点上解决，因为未标号周期几何存在 species-mixed automorphism orbit。

因此，当前阻塞至少分为三个彼此独立的问题：

| 层级 | 当前状态 | 不能得出的结论 |
|---|---|---|
| 条件控制 | 速度会变，但采样分布没有明显分离 | 不能说 tensor conditioning 已经有效 |
| 生成基座 | endpoint-ID 下 type/site decoding 未过 | 不能把失败归咎于 GaugeFlow alignment 本身 |
| 数据评测 | v1 split 有公式和近重复泄漏 | 不能用 v1 做可信的全量 generalization benchmark |

### 1.3 当前允许与禁止的动作

已冻结：Gate A v1、A1、A2 S1、A3、A4--A11.0 的报告、split、阈值、失败结论。不得回写或改变阈值。

当前最新的 A11-Q0 仅验证了 exact assignment 的数学实现，**没有训练 Q1**。因此不得因为 Q0 通过就恢复 tensor condition、全量训练、relaxation、DFT 或 DFPT。

## 2. 仓库、运行环境和代码入口

工作树：`E:/CODE/T2C-Flow/gaugeflow_perf_audit/`。实际生成代码与 FlowMM、PiezoJet 分离；不会在运行时 import FlowMM 或 PiezoJet checkpoint。

| 路径 | 作用 |
|---|---|
| `src/gaugeflow/data.py` | CSV/CIF、tensor cache、PyG `Data/Batch` 数据读取 |
| `src/gaugeflow/tensor.py` | 3×3×3 tensor、Voigt、18D irreps、SO(3) frame、response field |
| `src/gaugeflow/unit_cell.py` | Niggli reduction 与整数 basis 变换追踪 |
| `src/gaugeflow/manifold.py` | torus 坐标、SPD lattice metric 的 log/exp 表示 |
| `src/gaugeflow/model.py` | condition encoder、周期图、scalar/vector message passing、三头 velocity |
| `src/gaugeflow/stabilizer.py` | 792 个 proper lattice-action proposal 和 state-derived soft posterior |
| `src/gaugeflow/flow.py` | continuous flow matching、Euler sample、CFG、A2/A3 auxiliary losses |
| `src/gaugeflow/discrete.py` | A6--A9 的 absorbing discrete type-flow 诊断实现 |
| `src/gaugeflow/coupling.py` | 训练期 periodic OT/translation quotient coupling |
| `src/gaugeflow/assignment.py` | A11-Q exact count-constrained assignment / quotient 原语，尚未训练接入 |
| `scripts/train.py` / `scripts/sample.py` | 当前 continuous production path 的训练/采样入口 |
| `configs/*.json` | 每一 Gate 的冻结配置、预算、种子和阈值 |
| `reports/` | 已生成的审计、CSV、manifest 和失败记录 |

**唯一可报告环境：** WSL2 `Ubuntu-22.04`，解释器 `/home/future04/micromamba/envs/flowmm-t2c/bin/python`，PyTorch `2.5.1+cu124`，CUDA 12.4，RTX 4060 Ti 16 GB。Windows Anaconda 是 CPU-only 环境，不能用于报告训练、采样或性能结论。

最近完整回归：WSL 中 `PYTHONPATH=src python -m pytest -q` 为 **57 passed**。这证明单元/回归测试当前通过，不证明科学假设通过。

## 3. 数据：来源、字段、处理和当前形态

### 3.1 本地数据来源与可追溯性

项目内的原始输入容器是 `../gaugeflow/data/piezo/{train,val,test}.csv`。每行至少有：

- `material_id`：JARVIS 风格 ID（例如 `JVASP-1180`）；
- `cif`：结构文本；
- `piezo_irreps_raw`、`piezo_irreps`、`piezo_norm`；
- `piezo_unit`；
- `voigt_convention`；
- formation energy 字段（当前生成器不使用）。

CSV 三份文件合并后，冻结 TensorOrbit-JARVIS-v1 artifact 有 **4,998** 个 ID，split 是 4,000 train / 499 validation / 499 test。模型实际使用的 condition 不是直接读取 CSV 的 raw vector，而是按 `material_id` 的 SHA-256 前缀定位到：

`../gaugeflow/data/tensororbit_jarvis_v1/reynolds_projected_targets/<hash>.pt`。

每个 cache payload 存储 Reynolds-projected 的 Cartesian target `target:[3,3,3]`、用于检查不变性的 rotations、残差和 schema。`data.py` 验证：tensor 有限、形状正确、末两指标对称；然后把它投到 18D `CartesianTensor("ijk=ikj")` 坐标中。

**重要的可复现实务限制：** 本仓库包含 cache 的消费、hash、schema 和 Reynolds-invariance 审计，但不包含“从最上游 JARVIS 原始 calculation / raw tensor 开始构建 Reynolds-projected cache”的完整 builder 与外部 source pin。因而本文能严谨声称的是“本地 TensorOrbit-JARVIS artifact 完整、hash 可审计、投影后不变性通过”，而不是已独立复现了该 cache 的最上游数据发布过程。外部审查应要求补充 JARVIS 版本、下载 URL/commit、原始 tensor 单位、Reynolds group 的来源和投影公式。

### 3.2 张量约定

1. 原始 Voigt 输入的 source metadata 顺序是 `xx,yy,zz,xy,yz,xz`，且 `engineering_shear=true`；审计脚本验证此约定。
2. 内部 canonical `piezo_voigt_to_cartesian` 使用 `(xx, yy, zz, yz, xz, xy)`，生成 `e[i,j,k]` 并令 (e_{ijk}=e_{ikj})。sample CLI 也要求这个 internal canonical 顺序。
3. 通过固定的 e3nn `CartesianTensor("ijk=ikj")` change-of-basis，Cartesian tensor 与 18D irrep coordinate 双向转换。isotypic blocks 的维度为 (6+5+7=18)，训练时按三个 block RMS 做缩放，而不是逐 component z-score。
4. rank-three rotation 是

\[
(R\cdot e)_{ijk}=R_{ia}R_{jb}R_{kc}e_{abc}.
\]

5. unit-cell basis change 不会直接旋转 Cartesian physical tensor。它只改变 lattice row basis 和 fractional coordinate representation。

这两种“旋转”必须严格分开：整数 unit-cell basis change 是表示冗余；proper Cartesian (SO(3)) 才是 tensor orbit action。对三阶 polar tensor，improper (O(3)\setminus SO(3)) 不能在未来 tensor-conditioned quotient 中静默边缘化，因为 parity 有物理信息。

### 3.3 CIF 到模型状态的处理

对每个 CIF：

1. 用 `pymatgen.Structure.from_str(..., fmt="cif")` 解析；
2. 进行 tracked Niggli reduction。若 row lattice 满足 (L_{red}=B L_{orig})，则 fractional row coordinate 变为 (f_{red}=f_{orig}B^{-1}\pmod1)。代码强制 (B) 是整数、unimodular（(|\det B|=1)），并把 `niggli_transform` 存进 record；
3. 不旋转 tensor；
4. atom type 是 `Structure.atomic_numbers`，直接作为 `torch.long`，所以实际有效 element index 是 atomic number 1--118；网络 `atom_types=119`，index 0 是未使用的冗余 channel。这是一个值得外部审查的编码选择；
5. 保存 `frac_coords:[N,3]`、`lattice:[3,3]`、`piezo_irreps:[18]`、`condition_present=True`、response norm/stratum、zero flag 和 material ID。PyG `Batch` 以 `batch` node-to-graph index 合并图。

预处理 cache `artifacts/tensororbit_jarvis_v1_preprocessed_v1.pt` 保存上述 Niggli 后的数值 record，并保存 source file、cache index、code hash、split hash、tensor convention 的 manifest。它只减少 CIF/WSL mount I/O，不改变科学定义。

### 3.4 物理零与采样分层

exact zero tensor 是数据中的合法 physical condition，不等于 CFG null condition。v1 的 zero counts 是 train/val/test = **1,853 / 221 / 223**。训练默认可按 response norm 的五个 strata（zero 作为独立类）做 square-root inverse-frequency sampling；validation/test 保留自然分布。

训练时 CFG dropout 仅将 `condition_present` mask 置为 false，encoder 再输出 learnable `null_condition`。zero tensor 的 mask 保持 true，因此不会将“零响应材料”误当成“无条件”。

### 3.5 数据完整性与 split 风险

`reports/data_quality_audit.md` 已确认：

- 4,998 个 CSV / CIF / target-cache / split join 一一对应；没有 duplicate ID、missing target 或 non-finite target；
- Voigt/Cartesian FP32 round trip 最大误差满足 (2\times10^{-7})；Reynolds-invariance 审计阈值 (5\times10^{-4})；tracked Niggli quotient round trip 阈值 (10^{-5})；
- 77 个 Niggli “二次 reduction 不同”案例是等价 boundary-cell representative，tracked quotient recovery 仍正确；
- 但 v1 **不是 formula-disjoint**：165 个 reduced-formula group（672 rows）跨 split，且有 56 个跨 split StructureMatcher near-duplicate pair。

因此 v1 只能保留为历史 Gate A 小面板；它不支持“干净的 unseen formula / generalization”论断。已有 formula-disjoint v2 candidate（同样 4,000/499/499）和 activation audit，但仍为 inactive。所有未来 validation/test 或 full benchmark 都必须通过新的 v2 protocol、新 checkpoint 和匹配 oracle qualification，而不能静默替换 v1。

## 4. 生成状态空间和 continuous flow 定义

### 4.1 模型生成什么

当前 continuous production implementation 的状态是

\[
x=(z,f,\ell),
\]

其中：

- (z\in\mathbb R^{N\times119})：元素 logit state；
- (f\in\mathbb T^{N\times3})：分数坐标三维 torus；
- (ell\in\mathbb R^6)：晶格 metric 的 SPD log coordinate。

晶格并非直接在 3×3 matrix space 生成。代码先计算 (G=LL^T)，再以 `spd_log(G)` 的 Kelvin-style six-vector 表示；decode 时以 `spd_exp` 后的 lower Cholesky 作为 cell representative。它生成的是 orientation-free metric / shape-scale representative，而不是保留任意 global lattice rotation 的 3×3 cell。

这是一个明确的架构假设，外部审查应判断它是否与“以实验室 Cartesian tensor 为条件”的物理任务完全一致。它避开 unit-cell orientation redundancy，但也可能丢失 tensor 与 absolute lattice orientation 的某些联合语义。

### 4.2 source、target、interpolant 与 velocity

target state：

\[
z_1=\operatorname{onehot}(a),\qquad f_1=f_{CIF},\qquad \ell_1=\log(LL^T).
\]

source state：

\[
z_0\sim\mathcal N(0,I),\quad f_0\sim U([0,1)^3),\quad \ell_0\sim\mathcal N(0,I_6).
\]

默认 current production path 是 Euclidean logits：

\[
u_z=z_1-z_0,\quad
u_f=\operatorname{Log}_{\mathbb T^3}(f_0,f_1)
=((f_1-f_0+0.5)\bmod1)-0.5,\quad
u_\ell=\ell_1-\ell_0.
\]

对 (t\sim U[0,1])：

\[
x_t=(z_0+t u_z,\; \operatorname{wrap}(f_0+t u_f),\;\ell_0+t u_\ell).
\]

主 loss 是三个 head 的 MSE（可以选择 target velocity RMS normalization，但 Gate A 默认没有）：

\[
\mathcal L_{FM}=\mathbb E[\|v_z-u_z\|^2+\|v_f-u_f\|^2+\|v_\ell-u_\ell\|^2].
\]

采样从同一 source distribution 出发，用固定步长 forward Euler：

\[
x_{t+\Delta t}=\big(z_t+\Delta t v_z,\;\operatorname{wrap}(f_t+\Delta t v_f),\;\ell_t+\Delta t v_\ell\big).
\]

最终 continuous production decoder 是 `argmax(z_1)` 得元素类别、wrapped fractional coords 和 Cholesky lattice。A4.0 的解析 velocity closure 已证明此处的 interpolation 和 Euler 方向彼此一致；但这不证明神经网络学会了离散元素 manifold。

### 4.3 当前采样接口的实际条件

`scripts/sample.py` 只接受 target tensor JSON、`num_samples`、**外部指定的 `num_atoms`**、steps 和 guidance scale。它不接受 paired target CIF、target lattice、target graph、target stabilizer 或 target composition。

这意味着当前系统不是完全无条件地同时生成 atom count、composition 和 structure：atom count (N) 是外部给定的。A7 等诊断开始生成 composition count，但 production `sample.py` 没有接入该路径。外部审查应把“给定 N 的 conditional crystal generator”和“无约束材料生成器”严格区分。

## 5. Tensor condition 与 GaugeFlow architecture 的具体实现

### 5.1 四种 Gate A condition baseline

所有 baseline 使用同一个 graph vector field 容量；只改 condition representation。

| 模式 | 实际输入方式 | 科学定位 |
|---|---|---|
| `raw_tensor` | 18D lab-frame irrep coordinate 经 MLP 变为 graph token；无 edge response | 故意非等变的 raw-component control |
| `direct_irrep` | 对每条 local bond direction (n) 计算两条 Cartesian contraction | 不用 spherical harmonics/CG 的真正 direct equivariant interaction baseline |
| `stabilizer_pooling` | 固定 24 个 SO(3) tensor frames，frame embedding 均匀平均 | incoherent uniform orbit/stabilizer pooling baseline |
| `orbit_alignment` | 固定 tensor orbit + state-derived proper automorphism posterior + graph-query frame attention | GaugeFlow 的实际方法 |

`direct_irrep` 的两个 covariant edge feature 为：

\[
r_i=e_{ijk}n_jn_k,\qquad
s_j=e_{ijk}n_in_k.
\]

它们以 `torch.einsum` 算出；不使用 spherical harmonic 或 Clebsch--Gordan layer。代码注释称这些 Cartesian contraction 与相应 irreps coupling 代数等价，但外部审查应检查第二个 contraction 的 index convention、parity 和所需 complete basis 是否充分。

### 5.2 response field 和固定 probes

核心本构查询是

\[
F_e(n)=e:(n\otimes n),\quad F_e(n)_i=e_{ijk}n_jn_k.
\]

六个 fixed directions 为 Cartesian axes 和三条 face diagonals：

\[
(1,0,0),(0,1,0),(0,0,1),
\frac{(1,1,0)}{\sqrt2},\frac{(1,0,1)}{\sqrt2},\frac{(0,1,1)}{\sqrt2}.
\]

由于这些 direction 的 symmetric dyads span (mathrm{Sym}(\mathbb R^3))，对 (F_e) 的这 6 个 vector query（18 scalar）在数学上可恢复所有 `ijk=ikj` tensor component；它们不是一个有意丢信息的工程 summary。当前 noisy crystal 的 local bond directions 作为补充 query。

### 5.3 周期图和 message passing

当前 production backbone 直接复用 `geometry.periodic_closest_image_edges`：对每个 graph 构造有向完全图（不含 self edge），对于每对 atom 枚举 \([-2,2]^3\) 共 125 个 periodic translation image，保留最小 Cartesian distance 的 displacement、unit direction、distance 与 image shift。distance 经 finite-cutoff Gaussian RBF 进入每个 scalar/vector message；早期只保留 direction 的 `periodic_complete_edges` 已删除，防止和 decoder 的 PBC 定义漂移。

初始 node scalar 是 `Linear(119, hidden_dim)` 的 type state，vector feature 初始化为零。time 是 `MLP(1→h→h)` 后 broadcast 到每个 node；这个显式 time 注入是 A8 修复的真实 bug：此前 original-injection 在 endpoint-ID/raw/direct condition 下，message path 实际可能 time-blind。

每层 `ResponseMessageLayer` 只有 scalar invariant 和 Cartesian covariant vector 更新：

- scalar edge invariant：
  (n\cdot r,\|r\|,\|n\|,n\cdot s,\|s\|,r\cdot s)；
- scalar message 输入：source node scalar、target node scalar、node condition token、上面 6 个 invariant；
- vector message 是 learned scalar gate 加权的 (n,r,s,v_{source}) 线性组合；
- aggregate 用 `index_add_`，再 residual update scalar/vector state。

坐标 head 先从 vector features 得 Cartesian tangent，再以 (L^{-1}) 转为 fractional velocity；lattice head 从 graph mean scalar 得 6D velocity；type head 从 node scalar 得 119D logit velocity。

**已知的 A10 架构缺口：** 当 condition 是 endpoint-ID 时，edge response 全为零；scalar message 也不含 periodic bond length、RBF、edge distance 或现有 vector-state norm/dot-product 等 geometry invariant。故 scalar site classifier 看不到足够的 periodic local geometry，这解释了“composition 已可约束，但 B/N/In 无法可靠分配到 sublattice”的一个直接实现原因。这不是尚待猜测的超参数问题，而是源码可见的 feature omission。

### 5.4 `orbit_alignment` 的完整计算

1. 建立固定 (K) 个 deterministic proper SO(3) frame（Gate A 用 (K=8)，默认 CLI (K=24)；seed 0；含 identity）。这些是 QR 得到的随机正交矩阵，不是严格 SO(3) quadrature design。
2. 对 input tensor 生成 `framed[b,k,3,3,3]=R_k\cdot e_b`。小面板会预先 cache，避免每 step 重算固定 orbit。
3. 对 `orbit_alignment`，取 792 个冻结的 finite-order proper integer unimodular matrix (U) 为 lattice-action **proposal**。所有 proposal 均限制 crystallographic order (1,2,3,4,6)，已移除 shear/hyperbolic matrix。
4. 对当前 noisy (f_t,L_t,z_t)，把 (U) 的 lattice action 转成 Cartesian row map，并以 Newton polar iteration 投影为 proper rotation (Q_U)。posterior score 由 lattice residual、translation-marginalized periodic self-match、soft type mismatch 组成，再 softmax 成 792-way (p(U\mid x_t))。这里 `U` 不是 tensor rotation 本身，也不声称 noisy intermediate state 有真实 space group。
5. 对每个 frame tensor 施加所有 (Q_U)，以 posterior 加权得到 transform frame；对每条 edge 或六固定 probe 计算 response field。
6. 每个 frame 的 feature 为 isotypic norms、平均 edge field norm、fixed responses 等 24D vector，经 candidate MLP 变为 `values[b,k,h]`。
7. 用当前 graph state seed 作为 query，对 frame embedding 做 dot-product attention：

\[
q_k=\operatorname{softmax}_k\frac{W_k\phi(e_k)\cdot W_q h_{graph}}{\sqrt h},\quad
c=\sum_k q_k\phi(e_k).
\]

8. `c` 作为 graph condition token；edge field 以相同 (q_k) 融合，进入每一层 message passing。

训练和 sampling 都只使用 current generated state + tensor condition，不使用 paired target CIF stabilizer、space group、target graph 或 target lattice。此训练—推理信息对称性是设计上正确且已在 data audit 中检查的。

### 5.5 alignment 的数学/工程局限，必须审查

- 有限随机 frames 只近似 orbit integration / representative invariance；没有给出 (K\to\infty) 的误差界或严格 equivariance proof。
- 792 个 integer action proposal 是固定 catalogue，不是完整连续 stabilizer，也未保证每个 noisy state 的后验等价于物理 group marginalization。
- current production token 是 \(\sum_k q_k\phi(e_k)\)。A1 离线诊断同时计算了 \(\phi(\sum_kq_ke_k)\)，但没有替换 production path；不能把后者当成已验证改进。
- stabilizer posterior 的 soft nearest periodic match、temperature、chemical penalty、5×5×5 image shell 和 all-pairs graph 都是明确的近似选择，应被逐项审计。
- `safe_norm(x)=sqrt(max(sum(x²),1e-12))` 在 0 给有限的零梯度，修复了 physical-zero tensor 的 NaN backward；它也改变了极小值附近 derivative，应评估其是否造成偏差。

### 5.6 A2 的 residual conditional field（已失败，不是当前生产胜利）

A2 尝试显式拆分

\[
v(x_t,t,c)=v_{base}(x_t,t)+g(t)\Delta v_{cond}(x_t,t,c),
\]

\[
g(t)=g_{min}+(1-g_{min})4t(1-t),\quad g_{min}=0.25.
\]

base path 明确接收零 tensor token/zero response；每 block 的 conditional residual block 使用 FiLM 和 condition-dependent gate，分别产生 type / coordinate / lattice residual。还实现固定循环 wrong-condition counterfactual tangent-ranking：

\[
L_{cf}=\operatorname{softplus}(m+E_{own}-E_{wrong}).
\]

该实现和三 head residual norm 已记录，但四个预注册 variant 都未使 generated separation 达阈值。它是有价值的失败实验，不可作为“最终 GaugeFlow architecture 已采用”的表述。

## 6. 训练、CFG、辅助目标与解码

### 6.1 主训练流程

`scripts/train.py`：载入 dataset → 选固定 Gate panel 或全 split → 计算 three isotypic RMS scales → 可选 randomize direct-irrep representative → graphwise condition dropout → `RiemannianCrystalFlowMatcher.loss` → AdamW → gradient norm clip 1.0 → 保存 checkpoint（包含 config、scales、data manifest）。

默认 CLI 可训练 100k steps、hidden 256、4 layer、24 frame；这不是已完成 benchmark。冻结 Gate A 采用 400 steps、batch 8、hidden 64、2 layer、8 frames、AdamW lr 0.001、seed 20260714。

### 6.2 CFG

仅在训练时 `condition_dropout>0` 才有合法 CFG。sample 时若 (s\ne0)，实现

\[
v_{CFG}=v_{null}+ (1+s)(v_{cond}-v_{null}).
\]

physical zero tensor 与 null mask 独立。Gate A v1 checkpoint 的 dropout=0，因此 A1 中 nonzero CFG 只是一项 sensitivity analysis，不可取最好结果覆盖主 protocol。

### 6.3 A3 all-negative identification（已失败）

对 batch 中所有 condition (e_j)，给定 own interpolant (x_t^i)，定义

\[
s_{ij}=-\|v_\theta(x_t^i,t,e_j)-u_i\|^2_g,
\quad
L_{id}=-\log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}}.
\]

A3 用 (e^{-t/\sigma}) 强调 early time，(	au=0.25,sigma=0.25)，权重 0.5。该 objective 已实现并保持主 FM loss；它在 BN/InN two-target test 中未过 gate。

### 6.4 A5--A9 的非 production atom-type 路径

这些实现用于回答“Euclidean logits + final argmax 是否为基座失败源”，而不是已经选为 paper final model：

| 分支 | 实现 | 目的 |
|---|---|---|
| A5 | Dirichlet/simplex source、simplex tangent、endpoint NLL；periodic OT/no-drift coordinate coupling | 排查连续 logits 和 CIF row coupling |
| A6 | absorbing discrete-flow: (p_t(x\mid y)=t\delta_y+(1-t)\delta_{mask})，cross entropy endpoint posterior | 用真正 categorical path 代替 repeated argmax |
| A7 | 119-element graph-level count head；dynamic programming sampling exact total count；reveal 时 Hungarian 受剩余 slots 约束 | 分离 composition 成功与 site assignment 失败 |
| A8 | 在 original-injection 的每个 message path 加显式 time embedding | 修复 time-blind 实现 bug |
| A9 | A6/A7/A8 保持，训练时间改为 (t=U^2\sim\mathrm{Beta}(1/2,1)) | 增加 all-mask source region 覆盖，不做连续 hyperparameter search |

在 A6--A9，mask index=119，是内部非化学 token；decoder 不会输出它。A7 count latent 是模型生成而非外部 composition 条件；但受 count constrained reveal 仍通过 SciPy Hungarian 在 CPU 做 tiny-panel diagnostic assignment，不能误当成可扩展 GPU production sampler。

### 6.5 A11-Q 的 exact assignment（已通过 Q0，不在训练）

在 InN/BN four-site 2+2 面板上，Sinkhorn 不再作为主概率。给 site-species score (C_{ik}) 和 predicted composition count (n)，枚举所有**唯一化学** assignment：

\[
\mathcal A(n)=\{Y:\#\{i:Y_i=k\}=n_k\},\quad
S(Y)=\sum_i C_{i,Y_i},\quad
p(Y)=\frac{e^{S(Y)}}{\sum_{\tilde Y\in\mathcal A(n)}e^{S(\tilde Y)}}.
\]

2+2 时支持集大小 (4!/(2!2!)=6)，不是把同 species artificial slot 也排列的 24。对 partial discrete state (y_t)，只用残余 proper automorphism group：

\[
\Gamma_t=\{\gamma\in \operatorname{Aut}(X):\gamma y_t=y_t\},
\quad p([Y])=\sum_{\tilde Y\in\operatorname{UniqueOrbit}_{\Gamma_t}(Y)}p(\tilde Y).
\]

identical-species resulting labelings 会去重，避免按 group element 重复累加 likelihood。sampling 可从 complete categorical 抽样或 assignment-level Gumbel-max；第一版没有独立 node latent。

Q0 用 neutral zero score 完成 exact enumeration、count conservation、residual group 和 FP32 node relabeling consistency 测试，状态为 `q0_passed_exact_enumeration_read_only`。它还没有 Q1 network、composition prediction、StructureMatcher result 或 tensor conditioning。保留 Sinkhorn/Hungarian code 是未来大 (N) 的近似/诊断组件，而非这一 Gate 的科学归因。

## 7. 实验时间线、设计与真实结果

### 7.0 Review 后的 substrate-v2 实质修复（已实现，尚未训练）

这不是把 claim 写小，而是把审查所指向的实现缺口拆成可验证的新基座；
A5--A11 和 Gate A 的历史结果均不改写。

1. `vocabulary.py` 定义 physical atomic number `1..118` 到 dense chemical
   token `0..117` 的双向映射，mask 固定为 `118`。因此新 decoder 不再有训练
   中永远没有 target、却能在 argmax/sample 中被选中的 chemical class 0。
2. `geometry.py` 保留 PBC closest-image displacement、unit direction、metric
   distance 和 integer image shift；`GaussianRadialBasis` 将距离进入 scalar
   message。`substrate_v2.py` 的 vector channels 从 direction 构造，scalar
   更新显式读取 vector norm/dot invariants。这直接修复 endpoint-ID 时旧
   scalar type head 没有 bond length/RBF/geometry invariant 的 feature omission。
3. 首个 `substrate_v2_decoration_only_v1` 仅在 fixed geometry、all-mask type
   state、endpoint ID、**外供 composition（只为 isolated decoder test）**下用
   exact proper-SO(3) residual-automorphism quotient assignment NLL；它不能被
   叙述为 composition generation，也不会替代未启动的 A11-Q1。
4. 历史 `direct_irrep` 的两个 `einsum` contraction 不是完备 CG baseline。
   新 `direct_irrep.py` 使用 e3nn `FullTensorProduct`，完整保留
   `(2x1o+1x2o+1x3o) x (0e+2e)` 到 `1o` 的六个路径；已用随机 rank-3 tensor
   和 SO(3) rotation 做 shape/equivariance regression test。旧 checkpoint
   仍只属于历史 gate，不能被重新命名为完整 baseline。
5. `synthetic_teacher.py` 实现非抵消 synthetic rank-3 control：
   `(a_j-a_i) exp(-r_ij/r0) n_ij⊗n_ij⊗n_ij`。普通 symmetric-weight directed
   bond sum 在 `i→j`/`j→i` 下严格抵消，不能作为 tensor teacher；新 teacher
   通过 PBC/SO(3) 非零与等变测试，但不被称为真实 piezoelectric model。
6. `provenance.py` 和 `build_tensororbit_v2_raw.py` 使上游 release hash、
   row-level Voigt order/engineering shear/unit、proper-SO(3) Reynolds
   projection、zero count、ID join、cache hash、5015-vs-4998 exclusion manifest
   成为可执行 contract。当前工作树没有原始 JARVIS file/release pin，故不能
   启动真实训练；这是一项数据输入 blocker，而不是模型负结果的解释替代。

### 7.1 v2 raw build 真实执行结果（不等于 oracle/generator qualification）

之后在本机发现并独立读取 GMTNet release 的 `jarvis_diele_piezo.pkl` 数据副本。
GaugeFlow 没有 import PiezoJet module、没有使用 PiezoJet weights；只将该 pkl 作为
可哈希原始 input，通过自己的 exporter、converter、Reynolds projector 和 audit 重建。

- source copy: 5,000 条，GMTNet source commit
  `7a606a459ee48a320ed38450e391811fb43d5e19`，pkl SHA-256
  `2a57e081f0072b2ac7fca7769095adcded1d299d2cd971db5c93fd25eb66929d`；
- v2 4,998 IDs 全部 join，`JVASP-44417`/`JVASP-8639` 以
  `absent_from_frozen_tensororbit_4998_parent_population` 显式排除；
- build 输出 4,000/499/499 CSV 和 4,998 Reynolds-projected Cartesian cache，
  cache index SHA-256 为
  `b2c06198dc2efa578e1699546c8e605c77705953fe306ef950a15728bf63b1f2`；
- exact physical zero target 为 2,297；formula overlap 为 0；逐条 final-index
  symmetry、proper-SO(3) invariance、Voigt round-trip 和 ID join 均通过；
- raw source copy 的原始 download timestamp 不可得，故它是 **local pinned source
  copy**，不是已经完成 direct-official-download provenance 的最终资格。它同样
  不等于 GMTNet/SE(3) oracle qualification，更不等于 GaugeFlow 生成成功。

### 7.1 Gate A v1：四方法最小真实面板（冻结失败）

面板为 8 个固定 train ID：`JVASP-25138, 36991, 272, 33818, 1963, 36313, 16175, 37007`，2--6 atom，包含 exact zero，response norm 从 0 到 1.59898。每个 method 400 step，batch 8，hidden 64，2 layers，8 frames，单 seed。evaluation：8 loss repeat、8 representative repeat、每 condition 4 sample、每 sample 4 atom、8 Euler steps。

| method | shuffle gap | representative velocity error | generated between/within | feature shift | reproduced final train loss | failures |
|---|---:|---:|---:|---:|---:|---:|
| raw_tensor | 0.09360 | 0.35221 | 1.00444 | 2.21489 | 2.03353 | 0 |
| direct_irrep | 0.10845 | 0.21211 | 1.00937 | 2.08390 | 2.10048 | 0 |
| stabilizer_pooling | 0.12050 | 0.11765 | 1.00339 | 2.05897 | 2.02035 | 0 |
| orbit_alignment | 0.06634 | 0.04523 | 1.00664 | 2.04028 | 2.09976 | 0 |

阈值：median shuffle gap ≥0.02、GaugeFlow representative error ≤0.15、error/raw≤0.5、between/within≥1.2、feature shift≥0.02。除 generated separation 外的 supporting controls 通过。结论不是“模型没看 condition”，而是“velocity sensitivity/invariance 没有转化为生成分布显著分离”。

完整 Gate A 另外还要求：独立冻结 external tensor-oracle ensemble、训练集 orbit tensor error distribution、预注册 DFPT micro-audit；三者都未完成。

### 7.2 Gate A.1：conditional-to-trajectory causal audit（冻结负结论）

不训练，固定 792 proposal、8 targets、8-step sample。主要发现：

- 四种方法均未达到 1.2 separation，故首要归因是共享 conditional injection/backbone 或更下游生成基座，而不是仅 alignment pooling；
- teacher-forced own-target win rate / mean margin：raw 0.6944/0.1633，direct 0.6250/0.2215，pooling 0.6806/0.1927，alignment 0.6528/0.1586。margin 正但 retrieval 不足；
- common-noise trajectory 的 type/coord velocity difference 常在 early time 衰减，lattice signal 多能持续；最终 type logit state 差异依然小；
- alignment frame posterior 在 common initial noise 的 (t=0) 相同，之后 JSD 和 token RMS 变大；没有发现“posterior 不同但下游 token 完全一样”的单一解释；
- embedding audit 显示 pooling 会压缩 target distance，但 raw/direct/pooling/alignment 都失败，不能把失败只归因于 pooling collapse；
- 未训练 CFG 的 sensitivity sweep 最好也未达 1.2，且不能替代 protocol main result。

### 7.3 Gate A2 S1：shared conditional-control screen（失败）

仅 direct_irrep、同 8 IDs、预注册 400/800 checkpoint。四个 variant：original injection；residual field；residual+counterfactual；residual+counterfactual+graphwise dropout 0.1。主要 800-step 数字：

| variant | train loss | generated ratio | own win | mean margin | failures | 全部通过？ |
|---|---:|---:|---:|---:|---:|---|
| original | 1.77164 | 1.00683 | 0.63889 | 0.22056 | 0 | 否 |
| residual | 1.72490 | 1.00381 | 0.55556 | 0.24150 | 0 | 否 |
| residual + cf | 1.90162 | 1.00685 | 0.56944 | 0.41504 | 0 | 否 |
| residual + cf + dropout | 1.63154 | 0.99828 | 0.63889 | 0.24681 | 0 | 否 |

通过门槛要求 ratio≥1.2、win≥0.75、positive margin、terminal branch retention、own not worse、zero failure、三头 residual log。尽管 counterfactual margin 和 common-noise state branch 有正信号，仍不允许继续 S2 或 hyperparameter search。

### 7.4 Gate A3：two-target early branching（失败）

选择规则固定后得到 4-atom InN (`JVASP-1180`) 与 BN (`JVASP-22673`)：relative tensor-orbit distance 0.98325，lattice-shape distance 0.26234，均 nonzero high-response。400 step、direct_irrep，比较 FM-only 和 all-negative early identification。

| variant @ 400 | early retrieval | all-time retrieval | ratio | generated nearest-centroid | decoded endpoint retrieval | failures |
|---|---:|---:|---:|---:|---:|---:|
| FM-only | 0.60 | 0.5625 | 1.01222 | 0.75 | 0.375 | 0 |
| early all-negative | 0.70 | 0.50 | 1.01288 | 0.75 | 0.375 | 0 |

要求是 early≥0.9、all≥0.8、ratio≥1.2、decoded retrieval≥0.75。continuous branches 有、argmax composition 也确实发生变化，但 decoded crystal 仍无法可靠回到 BN/InN endpoints。该失败触发 A4 generator substrate audit，不允许继续 4/8 target。

### 7.5 A4：无 tensor 的 generator substrate audit（失败但定位了路径问题不在解析闭环）

先用当前 production interpolant + 解析真实 velocity + 当前 sampler 对固定 BN/InN endpoint 做 closure：type-only、coordinate-only、lattice-only、joint 都恢复 endpoint；最大 continuous error (3.11\times10^{-7})（阈值 (10^{-5})），decoded accuracy 1.0。

再把 tensor 去掉、以二类 endpoint-ID 做 condition：

| subspace | type composition | geometry retrieval | joint retrieval | ratio | failures |
|---|---:|---:|---:|---:|---:|
| type-only | 0.000 | 1.000 | 1.000 | 0.955 | 0 |
| geometry-only | 1.000 | 0.562 | 0.938 | 1.142 | 0 |
| joint | 0.000 | 0.562 | 0.625 | 0.896 | 0 |

原连续 119-logit 的 endpoint top-1 为 InN/BN = 0/0；active element mask 0.25/0.25；diagnostic `{B,N,In}` 0.50/0.50；simplex 0.50/0.75；categorical diagnostic 0.50/0.25。小词表不得作为 final model。结论：先修 type manifold/decoder 或 site architecture，不能回 tensor Gate。

### 7.6 A5--A10：逐层基座诊断（均未合格）

| Gate | 主要改动 | 有效改善 | 关键失败/结果 |
|---|---|---|---|
| A5 | Riemannian simplex + endpoint NLL；periodic OT/no-drift | simplex/unit-sum、OT/no-drift invariants 通过 | type composition 0.312；geometry retrieval 0.562；joint ratio 1.163；未过 |
| A6 | absorbing discrete flow + exact jump sampler | 解析 closure atom acc 1.0，mask 0 | composition 0.75，site atom 0.46875 |
| A7 | generated 119D composition count + count-constrained reveal | graph / decoded composition 均 1.0 | site atom 0.40625 |
| A8 | 修复 original-injection time-blind | site atom 提升到 0.59375 | 仍远低于 0.95 |
| A9 | fixed Beta(1/2,1) source-weighted time | 保持 exact count/mask closure | composition 0.9375（阈值 0.95），site atom 0.578125 |
| A10 | species-aware periodic StructureMatcher read-only audit | 排除“仅 CIF row 顺序”解释 | A7/A8/A9 match=0.438/0.562/0.188，确认真实 chemical sublattice mismatch |

### 7.7 A11.0：unlabeled periodic site-orbit identifiability（通过诊断，阻断 A11-G）

步骤：Niggli reduce → 所有 atom 替成 H → `SpacegroupAnalyzer` 只从 geometry 求 periodic operation/site permutation → 最后才读真实 elements。结果：

| material | proper SO(3) operations / orbits | full O(3) operations / orbits | species-mixed orbit | deterministic fixed-CIF ceiling |
|---|---|---|---|---:|
| InN | 4 / 1 | 4 / 1 | four-site `In:2,N:2` | 0.5 |
| BN | 2 / 2 | 2 / 2 | two `B:1,N:1` pairs | 0.5 |

所以 A11-G 的“geometry-only deterministic O(3)-scalar type decoder + fixed-CIF site accuracy ≥0.95”在这个 panel 上不可识别，禁止启动。注意：proper/full 在这两个材料数据上恰好给相同 partition，但 future tensor method 仍只可使用 proper group。

### 7.8 A11-Q0：exact enumeration / residual-group action（仅数学通过）

对 InN 与 BN 的 2+2 composition，exact support 均为 6；neutral uniform score 下 fixed-CIF (p=1/6)，proper quotient (p=1/3)，entropy (\log6=1.791759)。32 次 assignment-level Gumbel-max 每材料均 zero count failure，node-relabeling maximum log-prob error 0（阈值 (2\times10^{-6})）。

这是“公式与实现能够正确去重、取 residual group”的通过，不是 Q1 learned assignment accuracy。Q1/Q2 都是 `not_started`。

## 8. 性能、GPU 与等价性修复

模型一直在 CUDA；早期 4060 Ti 看起来低利用率，是 Python graph loop、重复 e3nn basis construction、数万 tiny kernel 和 host synchronization 使 GPU 饥饿，不是 CPU fallback。已做的优化包括：

- cache 固定 Cartesian/irrep change-of-basis；
- vectorize graph/frame/edge 维度；
- 792 candidate 全量 batch，并仅以 chunk 控制 memory；
- cache fixed tensor orbit/probes；
- small Gate A panel 使用 resident CUDA batch；
- CIF/Niggli/tensor preprocessing cache；
- SVD polar factor 改为 seven scaled Newton step，避免 repeated singular value 的 SVD backward NaN。

after benchmark（warmup 10 + measure 20 resident CUDA optimizer steps）：

| method | sec/step | 400-step 估计 | relative to direct | peak torch VRAM | sampling graphs/s |
|---|---:|---:|---:|---:|---:|
| raw | 0.0118 | 4.7 s | 0.77× | 19.5 MiB | 193.1 |
| direct | 0.0153 | 6.1 s | 1.00× | 19.5 MiB | 162.0 |
| pooling | 0.0160 | 6.4 s | 1.05× | 19.5 MiB | 139.6 |
| alignment | 0.0220 | 8.8 s | 1.44× | 34.5 MiB | 62.4 |

这是重要的工程修复：旧 alignment 5.092 s/step，修复后 0.0220 s/step，且新版本反而评估完整 792（旧实现意外 top-24）。但是这些优化不改变 Gate A 的失败结论。`torch.compile` 可使单次 direct step 变快，但 dynamic shapes 触发 8.56 s recompilation、warmup 28.4 s、VRAM 263 MiB，故未作为依赖。

## 9. 当前证据强度：可以说什么，不能说什么

### 可以说

- standalone code 能读 CIF/tensor cache，构造条件 vector field，CUDA 训练/采样；
- current `orbit_alignment` 比 raw tensor 更满足小面板的 representative velocity consistency；
- data join、tensor symmetry/Reynolds invariance、tracked Niggli round trip 的本地审计通过；
- continuous probability path 的解析闭环通过；
- A6--A9 已证明 composition count 与 site assignment 是不同难题；
- A10/A11.0 给出一次具体的 sublattice/automorphism 不可识别性证据；
- exact count-constrained quotient assignment 的 tiny-panel 数学原语 Q0 通过。

### 不能说

- “GaugeFlow 已实现 tensor-conditioned crystal generation 并成功控制压电 tensor”；
- “生成物已满足目标 tensor orbit”——缺独立 qualified oracle 与 orbit-error distribution；
- “在 JARVIS test 上泛化”——v1 split leakage 且 full run 未做；
- “已经完成 DFT/DFPT 物理验证/材料发现”；
- “A5--A9 或 A11-Q 是最终论文方法”；
- “fixed-CIF 低 site accuracy 必然是失败”——在 symmetry-equivalent assignment 下它只是诊断；
- “Q0 passed 等于 Q1 trained/passed”。

## 10. 建议外部审查者优先回答的问题

以下问题按“先决定科学定义，再决定是否换库/改代码”的顺序排列。

### 10.1 数据和物理 target

1. TensorOrbit-JARVIS 的最上游数据、DFPT setting、unit、tensor convention、Reynolds group、projection formula 和 cache builder 是否被完整 pinned？如果没有，先补 provenance，而不是继续训练。
2. 对 piezo tensor，按 crystal basis 与 Cartesian lab frame 的 convention 是否严格一致？source Voigt order 与 internal canonical order 的转换是否应有 independent symbolic / numerical reference？
3. conditional generative task 是“生成与 tensor compatible 的 crystal orbit”，还是“重构 paired training crystal”？若多个 structures share similar tensor，现有 conditional FM pairing 的 statistical semantics 是否合适？
4. 以 (G=LL^T) quotient 掉 lattice orientation 是否合理？它与 fixed Cartesian tensor 的 relative orientation 关系是否被全部由 response field 处理？
5. v2 formula-disjoint split 是否还应做 stronger prototype/structure family grouping，而不只是 formula grouping？

### 10.2 probability path 与离散化

1. Euclidean 119-logit straight path + terminal argmax 是否有合理的 categorical transport interpretation？A4 已显示它在 endpoint-ID 都失败。
2. A6 absorbing DFM 路径的 endpoint posterior objective 与 model input representation 是否完整匹配？time=0 all-mask state 是否仍能对称地表示多个 site assignment mode？
3. A7 count-constrained sequential reveal 是否仅是 sampler constraint，还是对应一个一致的 joint training likelihood？
4. A11-Q 的 exact assignment categorical + quotient NLL，如何同 generated composition distribution、masked partial state 和 latent symmetry breaking 形成可微、可扩展的一致 generative model？
5. 对 assignment 的 residual group 使用 \(\Gamma_t\) 是否应同时作用于 type path 的 forward corruption law，而不仅是 NLL？这是关键数学审查点。

### 10.3 symmetry/equivariance

1. closest-image selection 在 exact nearest-image degeneracy 处仍是 piecewise 而非 globally smooth；需要在未来大规模 protocol 中报告这类 boundary 的比例，并继续测试 cell-basis/symmetry covariance。direction-only omission 已修复：当前 backbone 同时使用 distance/RBF，且与 decoder 共用 PBC primitive。
2. direct-irrep 两个 contraction 是否构成充分且正确的 condition-to-geometry equivariant feature basis？
3. finite random SO(3) frames + attention 是否仅 approximate invariant；是否可用 exact irreps/equivariant network 或 deterministic quadrature 更严谨地处理？
4. 792 lattice proposals 的 construction、polar projection、soft type-aware match 是否物理上对应 latent gauge，还是可能把 generic noisy geometry 当作伪 symmetry？
5. `full_O3` 仅做 scalar-decoder diagnostic、proper-SO3 做 production tensor quotient 的规则是否正确处理 polar rank-3 parity？

### 10.4 GNN/decoder architecture

1. A10 已定位 endpoint-ID scalar site path 缺 bond length/RBF/periodic geometry scalar invariant。最小正确修复应添加什么？仅加 RBF 是否够，还是需要 local frame / higher-order equivariant features / distinct node latents？
2. complete graph + 125 image nearest shell 是否需要替换为 standard periodic radius graph（并保留严格 PBC invariance）？
3. type site assignment 是否需要 exchangeable node latent、assignment-level latent 或 permutation-equivariant set decoder？这应在不泄露 CIF order 的前提下设计。
4. 同一 backbone 同时预测 high-dimensional type, coordinate, lattice velocity。head scale/gradient audit 结果应否支持 multi-task balancing 或 separate trunks？不得盲目 search weight。
5. 当前 graph mean node embedding 对 composition 与 lattice head 是否过于弱，尤其对于 multi-modal crystal distribution？

### 10.5 评估与因果归因

1. generated between/within ratio 是足够好的 early Gate metric 吗？它使用何种 state distance，是否会被 different logit scale / lattice scale 支配？
2. teacher-forced tangent ranking 与 free-running sample separation 间的 gap，应该归因于 integration error、mode collapse、path coupling 还是 dataset conditional entropy？
3. 对 two endpoints，fixed-CIF accuracy 与 automorphism quotient / species-aware StructureMatcher 应如何 jointly report？
4. external tensor oracle 应采用哪些 orbit metrics、calibration/abstention 规则，才可避免 “oracle error 被误读成 generator failure”？

## 11. “可用成熟库解决”与“必须重新审查数学/架构”的初步分类

这不是最终建议，而是让外部审查者快速聚焦。

| 问题 | 成熟库能帮助的部分 | 不能靠换库自动解决的部分 |
|---|---|---|
| CIF/PBC/Niggli/symmetry | `pymatgen`、`spglib` 可负责 parsing、standardization、space group operation | 应不应该 quotient、tensor 如何随坐标约定变换是物理定义 |
| periodic neighbor graph | PyG / `torch_cluster` / e3nn / materials GNN 工具能做 radius graph、PBC edge | cutoff、image degeneracy、cell-basis equivariance、距离是否应是 feature 仍需定义 |
| SO(3) representation | e3nn 可提供 exact irreps、tensor product、spherical harmonics | “alignment posterior 是否必要/正确”与 tensor-orbit likelihood 仍是新方法问题 |
| assignment | SciPy Hungarian、OR-Tools、Sinkhorn package 可做 discrete/relaxed solver | composition latent、exact categorical law、residual automorphism quotient、joint likelihood 是模型设计 |
| discrete flow | 可参考/复用 DFM 实现范式 | crystal type/coordinate/lattice 的联合 probability path 与 symmetry quotient 仍需推导 |
| SPD / torus geometry | geoopt / PyTorch linear algebra 可辅助 | lattice metric quotient 是否符合 condition task、joint orientation semantics 不是库问题 |
| external oracle | GMTNet/e3nn predictor 可作为独立 model class | 数据 pin、matched split、qualification threshold、ensemble use 需 protocol |

## 12. 推荐审查顺序（避免错误地“先调参”）

1. **数据 provenance + tensor convention**：先补能独立重建/验证 Reynolds cache 的信息。
2. **任务定义与 state quotient**：明确 (p(X\mid[e])) 的 orbit、global orientation、atom count、composition 是否为条件/输出。
3. **解析 path 与离散 joint likelihood**：判断 continuous logits、DFM、count and assignment 是否形成一致 probabilistic model。
4. **site decoder 信息量和 symmetry breaking**：先解决 A10/A11 指出的不可识别/feature omission，再讨论 tensor condition。
5. **等变性与 alignment posterior**：以独立 numerical equivariance/unit-cell tests 审查，不以低训练 loss 代替。
6. **冻结一个最小新 protocol**：先 endpoint-ID 或 Q1/Q2 small panel；只有基座通过才重新测试 tensor condition；最后才使用 v2 full benchmark 和 physical validation。

## 13. 关键证据文件索引

| 主题 | 文件 |
|---|---|
| Gate A 数据/性能/科学等价性 | `reports/performance_data_scientific_audit.md` |
| 数据完整性和 split leakage | `reports/data_quality_audit.md` |
| Gate A.1 four-method causal audit | `reports/gate_a1_causal_audit/gate_a1_causal_audit.md` |
| A2 S1 residual-control screen | `reports/gate_a2_conditional_control_v1/gate_a2_s1_report.md` |
| A3 two-target screen | `reports/gate_a3_early_branching_v1/gate_a3_two_target_report.md` |
| A4 substrate summary / path closure | `reports/gate_a4_generator_substrate_v1/gate_a4_generator_substrate_v1_summary.md` / `path_closure_report.md` |
| A5--A9 reports | `reports/gate_a{5,6,7,8,9}_*/` |
| A10 StructureMatcher audit | `reports/gate_a10_site_representation_audit_v1/site_representation_audit.md` |
| A11.0 geometry automorphism audit | `reports/gate_a11_0_periodic_site_orbits_v1/a11_0_periodic_site_orbits_report.md` |
| A11-Q0 exact assignment audit | `reports/gate_a11_q_exact_assignment_v1/gate_a11_q0_exact_assignment_report.md` |
| Frozen initial Gate A protocol | `configs/gate_a_v1.json` |
| Latest Q exact-assignment contract | `configs/gate_a11_q_exact_assignment_v1.json` |
| v2 external oracle preparation | `configs/tensororbit_jarvis_v2_oracle_qualification_v1.json` |

## 14. 可直接发送给外部 GPT 的审查提示词

> 你是一位同时熟悉晶体生成、equivariant GNN、flow matching、离散生成、群作用与材料张量的严格审稿人。请阅读随附 GaugeFlow 技术审查交接文档，并按以下要求输出：
>
> 1. 将每个问题分为：数据 provenance、物理/数学定义错误、实现 bug、architecture information bottleneck、评估设计不足、仅需工程库替换。
> 2. 审查 rank-three polar piezo tensor 的 Cartesian/Voigt/SO(3)/O(3) 处理、Niggli basis change、lattice SPD quotient、periodic graph 和 stabilizer posterior 是否一致。
> 3. 审查 continuous 119-logit flow、absorbing discrete flow、composition count latent、exact assignment quotient 是否能形成一致的联合概率模型；明确指出不能成立的步骤。
> 4. 审查 A10/A11 给出的 site assignment 不可识别性结论是否充分，以及应如何定义正确的 quotient-aware target/metric。
> 5. 给出按优先级排序的最小改进方案。每项必须说明：保持哪些已冻结 Gate，修改的理论假设，所需数据/代码，最小验证实验，失败时如何归因。禁止只给“加大模型、更多 epoch、调 loss weight”之类无因果依据的建议。
> 6. 对每项建议标记：可直接使用成熟库、需要自行实现、需要先做数学证明/反例、或需要上游数据补全。
> 7. 不要把 Q0 数学测试通过误写成 Q1 训练成功，也不要把任何小面板负/正诊断写成 full benchmark、DFT 或 DFPT 结论。
