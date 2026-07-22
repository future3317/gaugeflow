# GaugeFlow 预训练与物理适配路线（A0--F）

更新日期：2026-07-21。

## 1. 当前结论

GaugeFlow 的数学表示不是当前主要短板。周期坐标、晶格 chart、Cartesian tensor
conditioner、parent--distortion--child 群论接口、trainer/EMA/checkpoint 和 reverse
sampler 都已通过各自的软件或数据资格。真正未闭合的是可自由采样的生成概率律及其
on-policy 组合：

\[
p(B,N,C,A,L,F)
=p(B,N)\,p(C\mid N)\,p(A\mid C,B)\,
p(L\mid A,C,N,B)\,p(F\mid A,L,N,B).
\]

这里新增的 \(B\) 是显式 carrier/parent 变量。以前的简写
\(p(N,C,A,L,F)\) 在 assignment 与 lattice 因子中又条件于 `parent`，但没有定义
parent 从哪里来；在 free generation 中这不是闭合概率律。`B=free` 表示不施加精确
parent 约束的 flexible stratum，`B=parent blueprint` 表示有资格化群作用的分层
carrier。两者是模型状态，不是运行时 fallback。

当前已通过：

- Alex-MP-20 全量数据与 `540,164/67,520/67,520` child-first split；
- exact stoichiometry-first \(p(C\mid N)\)；
- 条件坐标 \(p(F\mid A_{clean},L_{clean},N)\) 的冻结范围；
- 454 条 geometry-complete parent carriers；
- remaining-count orderless assignment Q0 的数学/软件资格；
- supported-IID learned exact-count assignment、显式 \(p(N)\) 与 lattice L1；
- clean/generated assignment/lattice 四臂 coordinate exposure；
- 34M/58M/98M 等 exposure 容量筛选，并选择 34.28M 最小充分 backbone。

当前未通过：

- 历史 free joint H1a 的局部 packing：nearest-neighbour normalized W1 为
  `0.953 > 0.75`；
- 旧 unary oracle-C Q1 的 held-out assignment likelihood/retrieval；
- 自由 parent/carrier law \(p(B,N)\)；
- 选定 34.28M backbone 上的完整 tensor-free joint A1/M1。

因此不能把“Alex-MP-20 有 675k 结构”写成“675k 条 parent-conditioned assignment
训练数据”。目前只有 454 条结构拥有完整、审计合格的显式 parent carrier。全 Alex
可以训练 flexible product field；显式 parent quotient objective 只能在有合格 carrier
的数据上训练，除非先建立 target-independent 的全数据 carrier law。

## 2. 数据角色

| 数据 | 当前规模/状态 | 主要角色 | 不能混淆的边界 |
|---|---:|---|---|
| Alex-MP-20 | 675,204 总计；540,164 train | GaugeFlow-base 结构先验与固定 benchmark | validation/test 不进入后续大数据训练 |
| geometry-complete parent carriers | 454 candidates | parent-aware exact assignment 的资格与小规模学习 | 不能代表全 Alex 覆盖 |
| MatPES-PBE / r2SCAN | 六个 immutable artifacts 合计 433,189 / 386,544 行；(N\le20) 为 387,129 / 362,737 | PES 表征、energy/force/stress 与 noisy-state distillation | 以 `matpes_id` 跨 artifact/functional 联合 split，ID 不进入模型 batch |
| LeMat force-labelled data | 待 manifest 资格化 | off-equilibrium force/stress 表征 | 缺失标签使用 mask，不做数值填充 |
| LeMat-BulkUnique | 5,438,436 总库存；5,005,017 `unique_pbe` | 大规模结构 continued pretraining | 先去重并排除 Alex val/test overlap |
| TensorOrbit-JARVIS-v2 | 4,998 | 后期 tensor adapter 与独立 oracle 数据接口 | 当前 base 不读取 |
| JARVIS/MP piezo、PhononDB、Born/dielectric/internal strain | 约 3k--10k 各任务 | 独立多任务物理模型 | tensor convention、functional、重复结构需统一 |

MatPES 的公开论文报告 504,811 个候选结构并分别给出 PBE/r2SCAN 成功率；实际训练
规模必须以本地 source manifest 为准，不能把候选数、成功数和两种 functional 混写。
LeMat 的 5.4M 是所有 functional 的库存量，不是一个完全同质的 PBE split。

## 3. 阶段 A0：GaugeFlow-base 接口闭包

A0 不做大规模联合训练，先确保每个自由生成变量都有合法来源。

### A0.1 learned exact assignment

使用 Q0 已通过的 orderless remaining-count law。对 target-independent 均匀 reveal
order \(Z=(z_1,\ldots,z_N)\)，部分着色为 \(A_{S_{r-1}}\)，剩余 count 为
\(n_k^{(r)}\)：

\[
p_\theta(A_{z_r}=k\mid A_{S_{r-1}},C,B,Z)
=\frac{n_k^{(r)}\exp \ell_{z_rk}}
{\sum_j n_j^{(r)}\exp \ell_{z_rj}}.
\]

score 读取 species-free geometry、完整 all-pair periodic RBF、partial assignment、
remaining counts 和 parent action；不读取 target coloring、CIF 行号或 prototype ID。
训练采用均匀 order 的合法 joint-path NLL；小系统 assignment marginal 与 unique-orbit
quotient 用 subset DP 精确审计。第一项 learned Gate 只在 IID assignment split 训练，
formula/prototype-disjoint panel 单独报告 OOD，不合并判定。

### A0.2 carrier 与 node-count law

必须明确生成时的 \(B,N\) 来源。依次比较：

1. train-only empirical \(p_0(N)\)；
2. learned \(p_\theta(N)\)；
3. parent blueprint 决定的 multiplicity delta law。

每个样本记录 node-count source。不能把 test 真值 \(N\) 或 target parent 静默输入。
显式 parent 分支必须先资格化 \(p(B,N)\)；未建立 parent law 前，full Alex base 只能
声称 flexible-carrier generation，不能声称 parent-conditioned free generation。

### A0.3 lattice L1 与 near-prior coordinates

L1 报告 positive volume、log-volume/shape、angles、condition number、density、crystal
family consistency，以及 generated lattice 输入坐标模型后的分布漂移。坐标 Gate 要
补从近先验开始的完整条件反向轨迹，并分别报告 clean/generated \(A,L\) 的四个角：

\[
(A_c,L_c),\ (A_g,L_c),\ (A_c,L_g),\ (A_g,L_g).
\]

A0 的 bounded supported-IID 子模块及 production product-space runtime 均已通过。
`joint` trainer/sampler 真实调用 `p(C|N)`、all-MASK 初态和 orderless exact-count
assignment，并拒绝旧 independent-site checkpoints。该闭合授权并已完成一次 A1；它仍
不等于自由 parent law 已闭合，也不授权 tensor、RL 或物理计算。

## 4. 阶段 A1：GaugeFlow-base 联合预训练

该阶段已按冻结协议完成。34.28M 模型在 seed 5705 下从头训练 8,441 steps，恰好一次
遍历 540,164 条 Alex train。A1-v1.1 final free-generation 指标为 NN-W1 `0.555003`、
volume-W1 `0.073341`、元素 JSD `0.047493`、对声明 train-only `p(N)` 的 JSD
`0.003924`、exact composition `1.0`、finite-positive lattice `1.0` 和零 mask/failure。
这只资格化 flexible-carrier tensor-free base，不覆盖 unseen parent action、metastability、
relaxation 或 tensor targeting。

数据以 Alex-MP-20 的 540,164 条 train 为结构主池，validation/test 固定不动。backbone
使用容量 Gate 选择的 34.28M 参数配置；58M/98M 仅保留为已审计容量点，不作为 runtime
fallback。训练
node count、unordered composition、exact count-constrained assignment、lattice、
coordinates 与 joint sampler，不接 tensor condition，不接 RL。

同一 product-field backbone 训练三种互补 measure：

- 全 Alex flexible-carrier joint states，学习一般结构先验；
- 有资格 parent carrier 的 exact quotient assignment states，学习显式对称先验；
- clean-side conditional states，防止 joint task 破坏已通过的坐标条件能力。

三种 measure 的 ID 是 audit metadata，不输入模型。parent carrier 不能过强：终态可由
occupation、OPD、strain 与 residual 降低对称性；模型必须保留 flexible stratum，避免
把所有材料压回常见理想原型。

最低 Gate：

- zero sampling failure、finite positive lattice、terminal masks 为零；
- exact atom count、composition validity、assignment count preservation；
- periodic minimum-distance 下尾、collision rate、nearest-neighbour distribution；
- lattice volume/shape/density validity；
- frozen MLIP single-point energy/force/stress 作为 metastability proxy；
- novelty、uniqueness、coverage 与 species-aware StructureMatcher；
- oracle/free gaps 按 \(N,C,A,L,F\) 每一层拆开；
- MatterGen、DiffCSP、FlowMM 等使用同 split、同训练图呈现数、同 NFE/采样数比较。

原生预训练模型与同预算重训模型必须分栏报告，不能用 MatterGen 的更大预训练预算与
GaugeFlow 的 Alex-only 预算直接比较。

## 5. 阶段 B：物理表征预训练

目标是改善当前短键/过度压缩问题，而不是在采样时调用 learned energy guidance。

训练前数据闭包现已覆盖 PBE/r2SCAN 各自 train/valid/test 六个 immutable artifacts。
原始 819,733 functional-rows 中有 69,867 条因 (N>20) 被排除；合格索引包含
749,866 functional-rows、387,697 个 unique material IDs，并按 `matpes_id` 重新得到
674,709/37,054/38,103 的 train/calibration/test。索引中坏行数为零；随机逐行 split
被禁止，ID 绝不进入 model batch。cohesive energy、force、stress 在 eligible rows 上
全覆盖；formation energy 仍只部分覆盖，因此统一 energy auxiliary 使用显式
cohesive-energy-per-atom target，不允许字段回退。functional normalization 只使用
train split，并采用标量 energy shift/scale、force scalar scale、isotropic stress shift
与 Kelvin scalar scale，从而保持旋转协变。

实现上，生成 forward 与物理预训练共享唯一的 periodic message-passing 编码路径；
clean physical 接口在编码后直接返回 scalar/vector features，不再白算 composition、
assignment、lattice 与 coordinate terminal heads。PBE/r2SCAN functional embedding 只在
Cartesian readout 前注入，保留共享几何表示且允许两个泛函拥有不同归一化预测。
MatPES loss 与 Alex replay loss 由同一个 optimizer 在同一步累加，避免两个 optimizer
竞争修改同一 backbone。

正式执行采用两个 rank 共同复现一个全局 MatPES permutation，再以 stride 方式取得无
padding shard；最后 21 条样本按 11/10 分配，绝不通过重复样本补齐。Alex replay 使用
独立的 deterministic wrapped stream。两条 stream 的 cursor、CPU/CUDA RNG、显式 diffusion
generator、rank-0 optimizer/EMA 与完整模型共同进入 hash-verified checkpoint。energy、
force、stress 和 PBE-only feature 各自按跨 rank 的实际有标签 graph 数归一化，而不是用
总 batch size 近似 mask 的分母。

validation 对 PBE 与 r2SCAN 分开报告 normalized energy/force/Kelvin-stress、force cosine；
PBE 另报逐原子 TensorNet feature cosine。所有 head 同时检查 aggregate 与 per-functional
step-0-relative 改善。生成保持性继续使用 A1-v1.1 的固定 512 个 reference、512 个 free
samples、100-step reverse SDE 和原随机种子，不能换一个更容易的 validation panel。

### B0 teacher qualification

- 冻结两个架构不同的 teacher，比较 clean/off-equilibrium feature transfer；
- 对 noisy generator state \(x_t\)，优先对齐 clean teacher target
  `stopgrad(T(x0))`，而不是在高噪声坐标上盲算 `T(xt)`；
- 分层报告时间、元素、配位环境与 functional transfer；
- teacher 选择依据是 generator-side validity/retention，不只是 energy MAE。

### B1 训练目标

\[
\mathcal L_B=\mathcal L_{gen}
+\lambda_h\,\|P h_\theta(x_t,t)-\operatorname{sg}T(x_0)\|^2
+\lambda_E\mathcal L_E+\lambda_F\mathcal L_F+\lambda_\sigma\mathcal L_\sigma.
\]

energy 使用 per-atom/source calibration；force 采用等变 vector head；stress 使用对称
Cartesian/Kelvin head；缺失标签只通过 mask 移除。PBE 与 r2SCAN 有独立 source/
functional embedding 与 normalization，结果分别报告。

collision、异常密度与短键项使用按元素对和密度校准的平滑 guardrail，不使用统一硬
距离阈值。B 训练持续 replay Alex base batches，避免物理目标覆盖生成概率律。

## 6. 阶段 C：扩大结构先验

从 A/B checkpoint 对 LeMat-BulkUnique continued pretraining，而不是从头覆盖：

1. 先按结构 hash、formula/prototype 和 StructureMatcher envelope 去除 Alex
   validation/test overlap；
2. source-balanced、functional-balanced、元素/原子数分层采样，防止高频体系主导；
3. 保留 Alex train replay 与 MatPES/force-labelled replay，防止 benchmark prior 和
   物理表征遗忘；
4. functional 作为明确 metadata condition 或 auxiliary label，不静默混成一个能量标尺；
5. Alex test 始终作为固定外部 benchmark，另设 LeMat IID/OOD 评估。

等待 Stage-B 期间已经完成独立的 source-balanced 数据流小测，但尚未启动 Stage-C
训练。对 16,196 条 bounded train rows（PBE/PBEsol/SCAN 原始计数分别为
13,826/665/1,705），1,000 个全局 batch 的采样比例为
0.32967/0.33353/0.33680；两个 rank 各取得 32,000 条且 checkpoint 后精确恢复。
该实现位于独立模块，不改动 Stage-B 冻结哈希。它只证明数据混合与恢复接口成立，不能
作为 LeMat 表征学习或生成效果证据。

Alex benchmark overlap 也已形成独立的 audit-only exclusion artifact：资格化的 val/test
各 67,520 条，规范化 ID 的并集为 135,040，两个 split 之间没有 ID 重复。ID 只在构建
LeMat index 时用于排除，绝不进入训练 batch。完整 LeMat index 必须绑定该列表的
SHA-256，不能仅凭数据集名称声称 benchmark 已隔离。

完整构建得到 5,068,754 条 `N<=20` rows，train/calibration/test 为
4,563,032/252,475/253,247，零坏行。进一步以 Alex ID 命中的 129,302 个 LeMat-native
`entalpic_fingerprint` 扩展排除时，新增跨 ID 命中为 0，index tensor 与 ID-only 版本
哈希相同。因此当前 provider-native fingerprint envelope 已闭合；它不被夸大成周期结构
的数学完备不变量，独立 StructureMatcher 仍可作为压力面板。

大数据阶段优先采用 packed graphs、按 node/edge 数动态 batching、向量化 segment
reduction、BF16 scalar path + FP32 geometry path、预取与 shard-local shuffle。任何加速
都必须保持 exact count、群作用和 periodic reference test。

## 7. 阶段 D：独立压电多任务模型

D 与 C 在项目日历上可以并行准备，但它在依赖关系上是 E 的前置，不参与 base
generation。目标模型为

\[
x\mapsto(e,\epsilon,C,Z^*,\omega_\Gamma,\Lambda).
\]

数据构建先统一 relaxed-ion/clamped-ion、Voigt/Kelvin 顺序、单位、极性 tensor parity、
functional、物理零与 missing label。formula/prototype-disjoint split 在去重后生成；同一
结构跨 JARVIS、MP、PhononDB 的记录按 source 保留，但不能跨 split。

模型使用共享等变结构 encoder 和 task-specific Cartesian tensor heads，masked
multi-task loss 利用极性模式、键合、电荷响应和软模之间的共享表征。至少保留一个
架构不同、训练数据隔离的外部 oracle；训练 adapter、RL reward 和最终评估不能只用同
一个 predictor。

## 8. 阶段 E：tensor-orbit adapter

E 同时等待合格 GaugeFlow-base 与合格 D 模型。先冻结元素 embedding、大部分 message
blocks 与底层 geometry encoder，只训练 tensor encoder、atlas posterior、condition
FiLM、末端 1--2 blocks 和 tensor heads；只有当生成结构 prior 保持且条件效应不足时才
逐层解冻。

资格顺序：exact synthetic equivariant tensor Gate → matched-capacity conditioner
ablation → real tensor-orbit adaptation → external-oracle/relaxation retention。必须报告
prompt swap、target separation、atlas posterior 相对 prior 的信息增益，以及最终样本
分布是否随 tensor orbit 改变；只改变隐藏表示不算条件生成成功。

## 9. 阶段 F：受约束 reward post-training

RL 仅在 direct tensor conditioning 已经有效之后运行。比简单加权和更安全的实现是：

- validity、sampling failure、collision 与 uncertainty 作为硬约束；
- tensor-orbit、stability 与 relaxation retention 为主要 Lagrangian objectives；
- novelty/diversity 为 batch-level objectives；
- 使用相对 GaugeFlow-base 的 KL/trust-region 防止结构 prior 崩塌；
- reward ensemble 与最终 evaluator 数据/架构隔离。

若写为标量 reward，可记为

\[
R=w_tR_{orbit}+w_sR_{stability}+w_rR_{retention}
+w_nR_{novelty}+w_dR_{diversity}-w_uR_{uncertainty},
\]

但权重、裁剪、constraint violation 和 reward hacking audit 必须在运行前冻结。完整
tensor-orbit distance、弛豫前后目标保持和 oracle disagreement 都是必需项。

## 10. 立即执行顺序

1. 从头运行一个 learned orderless oracle-C IID assignment Gate；
2. 若通过，单独报告原 OOD stress panel，不用 OOD 失败覆盖 IID calibration；
3. 资格化 \(p(B,N)\) 与 `p(N)` source；
4. 运行 lattice L1；
5. 完成 near-prior 与 generated-side-state coordinate Gate；
6. 冻结 GaugeFlow-base A1 配置并在 Alex train 上预训练；
7. A1 通过后才接 B，B 通过后再扩大到 C；
8. D 可并行做数据 schema/oracle 资格，但 E/F 不能提前。

当前不启动 tensor condition、RL、relaxation、DFT 或 DFPT。
