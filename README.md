# GaugeFlow

GaugeFlow 是面向压电晶体的 tensor-orbit-conditioned 生成模型。当前唯一
production 路径位于 `gaugeflow.production`：分数坐标使用周期平移商上的 wrapped
diffusion，晶格使用
log-volume / trace-free log-metric 表示，三阶极性张量条件使用 Stratified
Cartesian Gauge Atlas。元素 reverse 尚未资格通过；现有 uniform categorical、
graph-composition 和 exchangeable-histogram 实现只保留为已完成 E1 机制证据，
不是合格 production substrate。项目没有旧 continuous-logit flow、harmonic
conditioner 或 FlowMM runtime fallback。

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
Gate 仍被阻止。局部算子筛选已经收口。随后完成的冻结 checkpoint 归因审计中，
中噪声 endpoint 检索、reciprocal 残差频谱与 frozen low-k probe 三项均未支持
低频全局缺口，因此不实现 reciprocal carrier。随后完成的 all-pair clean-topology
审计与固定架构两遍学习曲线均只用于 H1a 归因，不改变历史失败结论，也不授权
tensor 条件或后续 Gate。

## 当前方法

当前最终生成基座不再表述为“依次拼接元素、坐标和晶格模块”，而是同一个异质
乘积状态空间上的 typed reverse field：

```text
X = (A, F, L),       t = (t_A, t_F, t_L)
R_theta = (r_A, s_F, r_L)
```

其中 `A` 是离散元素/occupation 状态，`F` 是周期平移商坐标，`L` 是
log-volume / trace-free log-metric。joint generation、已知元素和晶格的 coordinate
generation、CSP 与未来的分段/交替采样只是 `[0,1]^3` 模态噪声空间中的不同路径，
不是永久分叉的模型。tensor orbit `[e]` 是贯穿所有路径的 quotient-valued condition，
不是第四个被扩散的状态。

训练端的五种 J1 regime 现在由
`gaugeflow.production.modality_task_measure.FiveRegimeTaskMeasure` 统一采样；它是
task-path measure 的等质量五节点 stochastic cubature，保持既有
`13/13/13/13/12` 覆盖和随机数顺序，regime ID 只作审计元数据，绝不输入 denoiser。
E1、L1、M1 和 J2 分别只资格化同一 reverse field 的分量、共享训练和 on-policy
路径，不进入最终模型图。精确 conditional family 应满足 nested-corruption tower
identity；当前只将其作为后续零训练审计目标，尚未加入 consistency loss 或信息时钟。

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

同一两遍 coordinate checkpoint 上的零训练采样器审计也已完成。25/50 NFE
probability-flow 虽分别只需 reverse-SDE-100 的 24.5%/50.2% 延迟，但最近邻
Wasserstein 非劣性失败（归一化值 1.02007/0.79848，对照 0.56626）。因此当前
production 继续保留 reverse SDE，不把“关闭随机性”当作无损加速。观察到
reverse-SDE-50 的 0.56892 接近 100 NFE，但这只是事后候选，尚未通过独立
held-out 资格测试。

该事后候选随后在零重叠的 512-structure panel 上被否定：SDE-50 的结构级
W1 差异 UCB95 为 `0.05767 A > 0.03 A`，并恶化 1%/5% 最近邻下尾，因此即使
延迟减半也不能替换 SDE-100。Sampler 搜索至此停止。

exposure-conditioned clean-topology audit 的正式分类为 `mixed`。两遍时总体
oracle gain 为 `0.09293`、相对 0.25-pass 保留率为 `0.6640`；其效应明显随时间
变化，在 `t=0.4` 降至 `0.04099`，但在 `t=0.6` 仍为 `0.14203`。这不允许直接
加入完整 ACF，也不支持单纯继续增加 exposure；下一步只应做零训练、按时间
定位的 Tweedie self-conditioning/conditional-variance 诊断。

局部算子收口后，`h1a_midnoise_reciprocal_attribution_v1` 在同一 seed 5705、step 8441
checkpoint 上完成了不训练的三重归因。`t=.35--.65` 的同 composition endpoint
top-1 检索均值为 `0.40315 < 0.75`，并从 `0.53543` 降至 `0.25984`；低频
`0--1.5 A^-1` 与高频 `2.5--4.0 A^-1` 的归一化残差比均值为
`1.05348 < 1.15`，支持时间点为 `0/5`；冻结 12-channel low-k ridge probe 的
held-out 改善仅 `0.002257`，高频匹配对照为 `-0.000682`，二者差
`0.002939 < 0.03`。低频图覆盖率为 `0.9766--0.9883`，排除了空频带解释。
因此三项预注册检查全部失败，正式决策为 `do_not_implement_reciprocal_carrier`。
报告、独立哈希复核和可复现三联图位于
`reports/h1a_midnoise_reciprocal_attribution_v1/`。当前证据把下一问题指向
中高噪声条件方差、数据暴露、probability path 或 staged/self-conditioned
coordinate generation，而不是另一个 local/global feature branch。

这一结论同时合并独立 Bridge worktree 的零优化器 NO-GO：其中
`t=.4--.6` low-frequency held-out explained fraction 为 `-0.001368`，相对
random Fourier 仅 `+0.000695`，相对 graph token 为 `-0.001368`，最低壳层
超出 atom-permutation null 仅 `0.007755`。其原始 result SHA-256 与两套实验的
职责边界记录在同一报告目录的 `bridge_no_go_synthesis.md`；不再重复 low-k
实验。局部聚合和低频全局 Fourier 两条假设均收口后，下一项只允许先做
clean-topology oracle/probe 的零训练诊断；该诊断与随后固定架构学习曲线现已完成。

完整 all-pair clean-topology v2 覆盖 `100%` clean coordination mass。中噪声
soft Jaccard 为 `0.50413`、hard topology switch 为 `0.26469`；clean oracle
使 residual energy 改善 `0.10716`，noisy-topology control 为 `-0.00354`。
冻结 probe 可预测 clean field（AUC `0.87923`、explained fraction `0.61362`），
但把 probe probability 直接代入 oracle linear carrier 反而改善 `-0.04391`。
因此正式结论是 `probe_predictive_but_topology_correction_not_residual_causal`：
不把 frozen probe 作为 production correction，也不据此否定联合学习 topology field
的可能性。报告位于 `reports/h1a_all_pair_clean_topology_attribution_v2/`。

固定 dynamic production 架构随后以 seed 5705 从头训练精确两遍，不改模型、优化器、
EMA、数据顺序或容量。0/0.25/0.5/1/2-pass validation ratio 依次为
`1.00000/0.73837/0.63348/0.54371/0.49103`；新 run 的一遍结果复现历史
`0.54417`。一遍到两遍相对改善 `0.096876`，处于预注册的 plateau `<=0.05`
与 undertraining `>=0.10` 之间，正式分类为 `ambiguous`。它排除“明显平台”，
但没有资格静默增加 pass/seed 或宣布 H1a 通过。下一项最多是一个零训练的
exposure-conditioned topology residual persistence audit，检查 clean-oracle 增益是否
随 0.25/0.5/1/2 pass 消退；production 保持不变。完整报告与图位于
`reports/h1a_fixed_dynamic_coordinate_learning_curve_v1/`。

后续的 exposure-conditioned audit 已完成，正式分类为 `mixed`。两遍 checkpoint
的中噪声 clean-oracle gain 为 `0.09293`，相对 0.25-pass 保留率为 `0.6640`；
它在 `t=.4` 降至 `0.04099`，但在 `t=.6` 仍为 `0.14203`。因此更多 exposure
不是充分修复，同时该结果也不授权直接加入完整 ACF。

最后一个零训练诊断从同一两遍 EMA score 构造 translation-quotient Tweedie
endpoint estimate。`t=.6` 时，它相对 noisy topology 降低 topology MSE
`31.27%`，但 AUC 为 `0.77003 < 0.8`；把它代入同一个 clean-oracle-fitted
Cartesian carrier 后，held-out coordinate residual 反而恶化 `4.95%`，结构级
bootstrap 95% 区间为 `[-0.06020,-0.03890]`。更强的 frozen linear probe
同样不能形成正确的 residual correction。正式决策是
`self_conditioned_topology_not_predictive_revisit_conditional_variance`：不加入
ACF、一步 Tweedie self-conditioning 或旧线性 topology carrier，下一项只能是
单独冻结的 probability-path conditional-variance / nonlinear conversion 归因。
完整报告位于 `reports/h1a_self_conditioned_topology_attribution_v1/`。

随后两项只读归因进一步排除了“carrier 拟合方式不对”这一解释。即使给 noisy、
probe 和 Tweedie topology 各自拟合最优 ridge carrier，held-out gain 仍分别只有
`-0.00041/-0.00045/-0.00012`，而 clean oracle 保持 `+0.14203`。匹配复杂度的
非线性 pair-to-vector MLP 同样在训练集过拟合：加入 topology 后相对 base 的
held-out 增量仅 `+0.00537`，结构 bootstrap 95% 区间跨零
`[-0.00503,0.00564,0.01570]`。因此不把 ACF、Tweedie 或另一条 topology feature
branch 接入 production。

代码审计同时定位了一个与上述 topology 假设独立、但会实质损害 coordinate-only
学习的训练—推理契约错误：条件采样固定真实元素与晶格，旧 coordinate-only 训练却仍
对这两项加噪。修复后只有坐标经过 torus path，元素和晶格作为 observed side
information 在训练、validation 和 rollout 中保持一致。在相同 seed 5705 和 2,111
step（约 0.25 pass）预算下，validation ratio 从历史 `0.73837` 降至
`0.49382`，`t=.6` explained fraction 从 `0.13024` 升至 `0.39070`；从
`t=.1/.2` 开始的 reverse-SDE-100 rollout RMS 为 `0.07684/0.12153 A`，零失败。
这证明主要可修复项是 conditional task contract，而不是缺少 topology carrier。
该小屏幕不改写历史 H1a，也不授权 tensor/oracle/物理计算。完整结果位于
`reports/h1a_coordinate_clean_side_information_v1/`。

随后同一修复合同在不改变模型、优化器、sampler 或容量的情况下，从头完成了全部
`540,164` 条训练结构的一遍训练（seed 5705，8,441 steps）。预注册检查全部通过：
validation ratio 为 `0.33219`，相对 0.25-pass screen 再改善 `0.16163`；
`t=.005/.1` teacher-forced endpoint RMS 为 `0.03756/0.04919 A`，`t=.6`
explained fraction 为 `0.63509`；从 `t=.1/.2` 开始的 reverse-SDE-100 rollout
RMS 为 `0.05123/0.07039 A`，零失败。训练吞吐 `267.57 graphs/s`，PyTorch 峰值
显存 `4917.15 MiB`。这正式资格化了“已知元素与晶格时的条件坐标生成基座”，但不
改写自由联合 H1a 的失败。结果、归因报告与图位于
`reports/h1a_coordinate_clean_side_information_one_pass_v1/`。

随后 J0 在同一 checkpoint 上做零训练 side-information sensitivity：`t_F=.5` 时，
受控元素腐化使 coordinate-score MSE 增加 `5.335x`，晶格腐化增加 `5.163x`，两者
同时腐化增加 `9.939x`。因此模型确实使用 chemistry/lattice，而不是因为 clean-side
任务简单就忽略条件。

J1 在同一 Cartesian backbone 中加入显式 `(t_F,t_A,t_L)` Fourier 时钟；64 图批次
固定覆盖 `13/13/13/13/12` 个 clean-clean、noisy-element、noisy-lattice、diagonal
和 independent-interior 状态。seed 5705、2,111-step 冻结实验通过：各角点 validation
ratio 为 `0.47273/0.51407/0.56107/0.57304/0.64015`。clean-clean 通过 `0.51851`
保留阈值，diagonal 通过 `0.66453` 改善阈值；全部时钟梯度非零，吞吐
`247.65 graphs/s`，峰值显存 `4714 MiB`。这支持继续统一 multimodal hybrid
diffusion，但尚未把独立时钟的收益与五任务 mixture 和 3.9% 参数增量分离，也还不是
自由 joint generation 资格。参数完全匹配的 C0/C1/C2 对照现已完成并失败：C2 相比
C0 的 diagonal/interior 配对区间跨 0；它只在 clean 与 noisy-element 上有显著收益。
因此 J1 应解释为“五任务 mixture + clocks”的 composite 成功，不能声称独立时钟导致了
noisy/noisy 改善。零 optimizer-step 梯度审计同时显示 median clip scale 为 `0.2661`，
所有 regime-pair median cosine 为正，没有持续冲突，所以保留 global clipping，不加入
blockwise clipping、AGC 或 target-RMS normalization。随后仍须分别资格化 E1 元素
reverse、L1 晶格 reverse 和联合 M1，才可用同一 reverse-clock 时刻的 on-policy side
states 建立 J2；当前 coordinate-only checkpoint 的两个上游 head 未训练，不能拿来伪造
J2。J1 原报告及 matched/gradient 报告分别位于
`reports/h1a_j1_independent_modality_times_v1/`、
`reports/h1a_j1_matched_clock_attribution_v1/` 和
`reports/h1a_j1_gradient_geometry_audit_v1/`。

## E1 元素 reverse 资格结论

元素分量在固定真实坐标和晶格、完整 118 类词表、seed 5705 和 2,111-step
预算下完成了三轮有边界机制筛选，均未通过，因此 L1、M1 和 J2 仍不允许启动。

- absorbing-mask 基线的 teacher-forced NLL 会下降，但 free site accuracy 只有
  `0.03843`，早期错误会被 absorbing path 永久锁死；
- uniform D3PM 允许每一步修正 token，并以 `O(N*118)` 的
  diagonal-plus-rank-one posterior 替代逐节点 `118x118` 矩阵，但 exact composition
  仍为 `0/256`；
- 独立 graph-composition head 没有改变高噪声失败。离线给同一 terminal logits
  正确 counts 后，site accuracy 达到约 `0.70`，证明主要瓶颈是全局 species multiset，
  不是 site ranking 或 Hungarian；
- exchangeable histogram residual 在 `t=.25` 将 composition overlap 从
  `0.68352` 提高到 `0.87534`，clean-token oracle exact 达到 `0.89062`，说明低噪声
  计数链路已正确；但 `t=.9` overlap 仅 `0.08530`，free reverse overlap/site
  accuracy 仅 `0.06831/0.03396`，所以它仍失败。

当前结论是：现有独立 site-token 高噪声状态没有形成全局一致 formula 的随机变量；
继续加 composition head、局部特征、训练步数或 sampler 调参均未获授权。只读数据审计
显示 train split 的 `99.7408%` 结构不超过四种元素、最大七种，因此下一步只准备
稀疏显式 `composition C + count-constrained assignment Y` 的 exact synthetic kernel
资格测试。结果与图位于 `reports/h1a_e1_*` 和
`reports/h1a_e1_summary_v1/e1_element_qualification.pdf`。

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

## E1 explicit composition status (2026-07-20)

The element-only site-token mechanisms remain failed. The current explicit
composition candidate is an exact stoichiometry-first law:

```text
N -> integer partition lambda -> distinct element tokens -> exact composition C
```

For `N<=20` and at most seven species, it enumerates 1,840 partitions and fits
only a smoothed train-split `p0(lambda|N)`. Equal-count element slots are
deduplicated by increasing-token tie order. A shared count-position encoder
and current-count query replace the retired interleaved species/count
autoregressor and per-partition lookup table; target composition is never an
input.

The no-training Q2 kernel passes exact normalization, FP32/BF16, gradients and
RTX 4060 Ti performance (`3.55 ms` teacher forcing and `15.17 ms` sampling per
256 graphs). Its single-pass IID calibration screen passes partition TV
`0.03551`, support TV `0.00826`, element JSD `0.00119`, exact atom count and
zero failures, but fails final/initial NLL `0.77569 > 0.75`. E1 therefore
remains failed and count-constrained assignment is not started.

A subsequent zero-training attribution isolates the trained species term. Its
ratio is `0.750885`; the larger archived total ratio includes a fixed
`1.617298`-nat/graph partition-prior term, and the random network is itself
`0.178444` nat/decision worse than the exactly legal uniform distribution.
The final law beats a train-fit count-slot reference by `0.383854`
nat/decision. With the reference integer partition fixed, pair JSD is
`0.010461`, pair-probability RMSE is `0.000451`, and frequent-pair recall is
`1.0`. This identifies the initial/final total-NLL ratio as an indirect Gate
metric, but does not rewrite its failure. A future independently frozen E1
must use absolute conditional likelihood and co-occurrence; assignment is
still blocked.

The formula/prototype-disjoint H0 validation split is intentionally also
stoichiometry-disjoint (conditional partition TV `1.0`). It remains the OOD
novelty/coverage reference, but is not used as an IID marginal-calibration
target and is never added to training.

## 仓库原则

- 当前代码就是唯一 runtime；不保留旧模型兼容分支。
- 当前处理数据由 manifest/hash 确认；raw source 保留在外部数据目录。
- 历史失败和旧 runner 只在 Git tag 与 `docs/research_iteration_history.md` 中复现。
- 不把 target CIF、target lattice、material ID、target space group、stabilizer 或
  species mapping 输入 denoiser。
- polar rank-three tensor orbit 使用 `SO(3)`；晶体兼容性才使用显式 parity 的 `O(3)`。
- H1a 通过前不启动完整 blueprint、tensor、oracle 或物理验证。
