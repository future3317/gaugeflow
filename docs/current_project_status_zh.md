# GaugeFlow 当前实现、方法演进与实验状态（2026-07-18）

## 一句话结论

GaugeFlow 已经从早期连续 logit/ODE 原型重构为混合离散—连续晶体扩散框架，并完成 Cartesian tensor-orbit conditioner、反向采样软件闭环、H0 数据/群论资格化和 Alex-MP-20 全量 H1a cache。真实数据 H1a 已运行，但局部坐标生成与最近邻分布未通过，因此当前不能声称已生成满足目标压电张量的晶体。

本项目现停在 H1a 坐标生成器诊断。H1b、H2--H6、真实 tensor、oracle、relaxation、DFT 和 DFPT 均未启动。

## 已经完成了什么

### 1. 生成基座从旧 flow 改为匹配状态空间的 hybrid diffusion

当前生成状态为：

\[
x=(a,f,L),
\]

其中元素 (a) 使用 absorbing-mask categorical process，分数坐标 (f\in\mathbb T^{3N}/\mathbb T^3) 使用周期平移商上的 wrapped kernel，晶格使用 log-volume 与 trace-free log-metric shape。旧的高斯元素 logits、硬 nearest-image flow 和 Euler ODE 不再是 production fallback。

S1a-I0 v1.3 证明 trainer、EMA、checkpoint 恢复和联合 reverse sampler 在 CUDA 上可以闭环运行，且终态 mask 与 sampling failure 均为零。它只是软件资格测试，不是真实数据生成质量结果。

### 2. 用 Stratified Cartesian Gauge Atlas 表示完整三阶极性张量轨道

张量条件是 proper-(SO(3)) orbit，而晶体 Neumann compatibility 使用显式 parity 的 full-(O(3)) point group。当前 conditioner 定义状态依赖有限测度

\[
\nu_{x,e}=\sum_{R\in\mathcal A(x,e)}w_R(x,e)\,\delta_R,
\]

并以带权 posterior 聚合旋转后的 rank-three tensor/response field。generic stratum 保留经过证明的 4,032 个候选；axial 与 descriptor-isotropic 情况使用 multiplicity-corrected residual rules 和 smooth blending。24-frame-only S0.3-v1 已冻结失败，不能回退；保持同一 4,032-candidate prior 的 S0.4.1 在 RTX 4060 Ti 上通过 20 ms 门槛（14.62 ms，15.19 MB）。

### 3. 将“终态必须等于精确 parent symmetry”改为 parent--distortion--child 分解

生成分布写为

\[
p_\Theta(x_c\mid[e])=
\sum_{b_p,d}\int p(b_p,x_p\mid c_{\rm inv})
p(d\mid x_p,[e])p(y\mid x_p,d,[e])
\delta\!\left(x_c-\Phi(x_p,d,y)\right)d\mu(x_p)dy.
\]

这里 exact space group 只是 parent prior。低指数 HNF 超胞、finite affine quotient、physical-real irreps、OPD fixed spaces、Kelvin strain、质量加权 mode displacement 和小 residual 共同构成有界的有序对称性破缺空间。v1 明确不把缺陷、混占、无序、大超胞或有限温度相变伪装成 residual。

### 4. 修正了最关键的 occupational-order 表示错误

早期 parent occurrence 要求高对称 parent action 同时保持终态元素标签，这会错误排除“几何高对称、化学着色降对称”的真实路径。现在 parent 是无元素的几何 carrier，元素是独立的完整 118 类整数着色 (a_i)。若 parent operation 对节点的作用为 (\pi_g)，定义

\[
H_{\rm occ}(a)=\{g:a_{\pi_g(i)}=a_i,\ \forall i\},
\]

并使用

\[
H(d,a)=G_p^B\cap H_{\rm occ}(a)\cap
\bigcap_{\ell:z_\ell=1}H_{\ell,c_\ell}.
\]

这使 occupational ordering 成为真正的离散对称性破缺变量，同时保持 exact coloring reconstruction、整数元素、群闭包和 subgroup certificate。它不是 target species mapping，也不是 partial occupancy。

### 5. 完成 H0 数据与目录资格化

- H0-A：675,204 条 Alex-MP-20 结构被分为 540,164/67,520/67,520；formula、exact prototype、StructureMatcher envelope 和 connected component 均无跨 split 重叠。
- H0-B：10,034 个 PhononDB compact Hessian 通过全量代数审计，另有冻结的 1,024-material 数值审计。
- H0-C：TensorNet 与架构不同的 QET MatPES-PBE teachers 通过 512/32 离线资格审计；不得用于 reverse guidance。
- H0-D-v2：覆盖 230 个空间群、6,188 个 \(\det B\le4\) HNF orbits、53,441 个 physical-real irreps 和 75,416 个 OPD classes。
- H0-E-v4：O1 在 835 个与 O0 完全不重叠的 held-out 材料上找到 224 个新材料和 454 条唯一路径；与历史 125 个和 O0 的 10 个材料合并后为 \(359/1023=0.350929\)，超过冻结阈值 0.15。完整反序独立审计一致，H0-v4 正式通过。

历史负结果没有覆盖：H0-E-v1 的 \(125/1024=0.12207\) 仍是失败；E1a maximal-t 与 K0 maximal-k 的 0/64 也仍是冻结失败。它们说明几何 parent projection 不足，最终促成 occupational-order 修正。

## 数据清洗原则

原始数据不删除，历史 artifacts 不重写。确认属于数据损坏或具体任务域不兼容时，只在版本化数据入口进行 fail-closed 清洗，并记录 ID、理由、证据和 manifest hash；不为坏数据增加模型 fallback。

`alex<agm004639609>` 仅从未来 parent-occurrence / blueprint-activation 数据中剔除，因为其观察到的 parent-path Hencky strain 为 0.48977，超出冻结的 0.15 domain。其有限 child structure 对 P1 结构生成仍有效，因此不会从 Alex 原始源或 H1a 结构池中删除。

## H1a cache、训练与当前暂停点

`h1a_p1_structure_cache_v1` 已完整运行并独立通过。675,204 条结构全部重建成功，split 为 540,164/67,520/67,520；最大 source-equivalence error 为 `8.10e-15 A`，float32 cache error 为 `2.79e-6 A`。ID、split、prototype、space group 和 Niggli transform 全部留在 audit index，没有进入 denoiser。

联合 tensor-free H1a 使用全部 train split，20,000 steps 共呈现 1,280,000 个 graphs（约 2.37 passes）。晶格有限且正体积，sampling failure 和 terminal mask 均为零；element marginal、volume 和 formula uniqueness 通过。但生成最近邻中位数为 `2.172 A`，训练参考为 `2.698 A`，归一化最近邻 Wasserstein 为 `0.953 > 0.75`，因此 H1a 失败。

随后 seed 5705 做了恰好一遍 540,164-graph 的 coordinate-only 预训练。validation 从 1.037 降至 `0.54928`，但未达到 `0.35`；`t=.005` endpoint RMS 为 `0.04640 A > 0.04 A`。raw/EMA 和 train/validation 对照排除了 EMA lag 与普通泛化差距作为主因。重复元素代表元的 raw target 虽不同，但替代代表元后验质量至多 `5.42e-14`，因此没有引入 Sinkhorn/Hungarian 或 permutation-path 修改。

最后资格化并测试了更一般的 signed pairwise reciprocal residual。算子本身在 FP64 的平移、周期代表元、置换、O(3) 和 unimodular 基变换误差均约为 `1e-16`，CUDA 训练步为 `490.77 graphs/s`、`1.73 GiB`。但一次完整预训练只把 validation 改善到 `0.53354`，仍失败。分支归因证明它确实活跃，所以结论是“收益不足”，不是“没有接通”；该实现已从 production 删除。

### 坐标 tangent、精确 readout 与优化几何

随后的小面板均是对上述全量 H1a 失败的机制审计，不是用小数据替代训练集。纠正后的
平移商 Jacobian 在 30 个物理方向上满秩 `30/30`。阻尼 Gauss--Newton 线性模型预测
完整步可消除 `99.9337%` 的单状态 loss，但该步长是全部 active parameter 范数的
`3.1575` 倍；在真实局部曲率半径内，最佳预注册步只消除 `0.1388%`，更大的步随即
恶化。这排除了“直接做一个大 pseudoinverse 更新”，同时表明问题不是严格不可表达。

使用显式 Helmert basis 精确消除三个公共平移零模后，最终 225-parameter affine
coordinate readout 覆盖 `30/30` 方向，target projection residual 为 `1.12e-15`，
应用于 production forward 的 MSE 为 `5.39e-8`。但 quotient condition number 为
`3.496e7`、effective rank 为 `2.23`，最小范数更新为 `2079.20`，而初始化范数仅
`0.80036`。因此物理方向完整，但 basis 高度相关且弱方向参数尺度异常。

固定 backbone features 时，1/4/16/64 个状态的最优 affine MSE 分别为
`1.55e-27`、`1.43e-14`、`0.09947`、`0.55232`。小面板可以被 readout 记忆，
16/64 状态则要求 backbone 学出不同特征。graphwise unit scaling 虽把 update norm
降到 `6.14`，仍未达到 spectrum、吞吐与 translation guardrail；未正则 variable
projection 令 head norm 从 `9109` 膨胀到 `4.83e7`；screened quotient Laplacian
没有改善谱；单独的 powers-of-two `1024x` readout scaling 虽把精确解范数降到
`2.03`，但 Adam/global clipping 下 1,024-step MSE 为 `0.40491`，劣于历史
`0.34414`。这些候选已全部从 active runtime、配置与测试入口删除，只留报告和 Git
历史。当前 production 恢复为简洁原坐标 head。

综合判断是：当前阻塞不是 cache 损坏、解析 probability path 不闭合、坐标 head
缺少物理方向，或只需增加 seed/steps；而是严重各向异性的优化几何与随状态变化的
feature learning。

预注册的 16-state FP32/BF16 stability audit 随后在训练前否决了 scaled variable
projection。固定 `1024x` chart 保持函数到 `5.96e-7`，design 为满秩 `225/225`，
scaled solution norm 为 `8.894`，FP32 exact MSE 为 `0.099467`、backbone gradient
norm 为 `3.889`，说明代数和 FP32 路径正常。但 BF16 MSE 为 `10.9886`，是 FP32 的
`110.47` 倍；BF16 gradient norm 为 `23468.3`，是 FP32 的 `6033.9` 倍，梯度余弦
为 `-0.1572`。vector/edge 两个预测分量的范数为 `272.59/271.00`，合计仅
`16.83`，即 `32.31` 倍相消。缩放只把存储解降到 `8.894`，等效未缩放权重仍为
`9107.83`，没有改变数值病态。

该审计执行了零个 optimizer step，参数精确恢复，没有改 production。现在没有 active
scaled-variable-projection 候选；下一机制必须先证明一种紧凑、等变且不依赖 target 的
basis decorrelation 能消除相消，再允许训练。不得搜索 scale、ridge、precision、
solve frequency、steps 或 seeds。

随后对“直接删除一支”进行了同一 16-state、零训练的最小性审计。显式 Helmert basis
严格消除三个平移模后，vector-only、edge-only 和 combined 在第一个 11-site 状态上
均为 quotient rank `30/30`，target projection residual 均小于 `1.8e-13`。这说明两支
各自在单状态都包含全部物理方向，问题在跨状态表达和数值尺度。

vector-only 的 BF16/FP32 MSE 比为 `0.9988`，数值相对稳定，但 16-state FP32 MSE
为 `0.56437`、low-time endpoint RMS 为 `0.05046 A`、solution norm 为 `1022.67`，
明显缺少跨状态拟合。edge-only 的 FP32 MSE 为 `0.13474 > 0.12`、endpoint RMS 为
`0.02401 A`、solution norm 为 `1325.83`；BF16 MSE 又升至 `10.2160`，梯度 norm
从 FP32 的 `4.295` 变为 `16794.1`，方向余弦为 `-0.1419`。因此不能通过删掉某一支
解决：vector 和 edge 在局部方向上冗余，却在跨状态 feature family 上互补。

该审计也没有 optimizer step 或 production 修改。当前 combined head 保留；下一候选
必须是 target-free、等变、紧凑的 orthogonal-residual basis，在保持联合跨状态 span 的
同时消除相消，而不是恢复单分支、改成 FP32-only 或增加训练预算。

随后执行的固定 block-orthogonal residual chart 在代数上完整通过：graph-equal Gram
条件数为 `1.000000004`，最大 Gram 误差为 `4.96e-10`，与原 combined span 投影的
相对差为 `1.35e-10`；正交参数范数为 `3.2299`，block cancellation 为 `1.3801`，
FP32 MSE/endpoint RMS 为 `0.099464/0.020287 A`。CUDA chart 算子只需 `0.0255 ms`
和 `0.360 MiB`。

但它仍在训练前失败：等效 raw solution norm 仍为 `9108.38`，BF16 MSE 为 `9.7679`
（FP32 的 `98.20x`），endpoint RMS 为 `0.30036 A`，backbone gradient norm 为
`14670.5`（FP32 的 `3050.5x`），梯度余弦仅 `0.1278`。这证明固定可逆 readout
换坐标只能改善存储参数几何，不能消除 BF16 对已形成高相关 feature 的扰动放大。
实验执行零 optimizer step，production 未改变。下一机制必须在最终 readout 之前
形成紧凑、尺度受控的 Cartesian coordinate carrier，而不是再加后验 whitening、
scale、ridge、precision 或 solve-frequency 变体。

上移到 feature formation 后，紧凑 Cartesian moment/Krylov carrier 在无 target、
零训练范围内通过。它以 16 个 scalar moment channels 构造一阶向量矩 `m`、二阶
对称无迹矩 `Q` 与三维 Cayley--Hamilton 截断 `(m,Qm,Q^2m)`，再与原 32-channel
vector stream 合成 80 个 RMS-balanced 极向量 carrier。16 个固定真实状态全部达到
完整 translation-quotient rank，最坏条件数为 `14657.96`；含 improper reflection
的 `O(3)` covariance error 为 `6.76e-6`，translation-horizontal error 为
`1.54e-7`。

真实 BF16 backbone 下 carrier 相对 RMS 为 `0.08966`、与 FP32 余弦为 `0.99598`；
target-free probe-gradient norm 为 `9.448/9.459`，比值 `1.00121`、余弦 `0.99269`。
12,192-edge 面板的向量化算子为 `3.043 ms / 11.609 MiB`。该实验读取零 coordinate
targets、执行零 optimizer steps，只授权另行冻结的 production 集成资格测试；H1a
仍失败，尚未允许 fixed-state target fit 或真实训练。

第一次 clean production 集成暴露的超大梯度最终被定位为 tensor index type 错误。
旧实现把 Cartesian covector 经 `L^T` 拉回 fractional chart，却让 reverse sampler
把它当作 tangent vector 直接加到坐标。当前唯一正确路径为

\[
v_r=v_fL,\qquad v_f=v_rL^{-1}.
\]

随后逐项修复了 periodic-lift 数值重构、CUDA atomic reduction 的不确定性和 BF16
geometry sensitivity。最终 geometry-sensitive message blocks、coordinate edge
encoder 与 Cartesian carrier 固定为 FP32 typed path，terminal scalar heads 仍可
BF16；graph/edge reduction 使用 target-contiguous `segment_reduce`，保持线性
复杂度且无运行时排序或精度 fallback。零训练资格达到 `516.03 graphs/s`、
`185.73 MiB`，重复误差为零，BF16/FP32 output 和 loss-gradient cosine 分别为
`0.999806/0.997593`，平移、置换、GL(3,Z)、O(3) 与 round-trip 全部通过。

在此基础上，seed 5705 使用全部 540,164 条 train split 完成 8,441 steps、恰好一个
完整 pass 的 Cartesian-tangent coordinate-only 预训练。validation 曲线为
`34.43436 -> 30.46289 -> 26.04380 -> 24.24037`，最终比值
`0.70396 > 0.5`；`t=.005` teacher-forced endpoint RMS 为
`0.04207 A > 0.04 A`，两项失败。`t=.1` teacher-forced RMS 为 `0.06143 A`，
从 `t=.1/.2` 开始的 100-step rollout 为 `0.06589/0.09861 A`，且 sampling
failure 与 tensor candidates 均为零。修正 tangent 将旧 covector 的低噪声 RMS
从 `0.05672 A` 改善到 `0.04207 A`，但仍不能按冻结协议放行。该 checkpoint 不会
初始化 joint model；不增加 seed、steps 或修改阈值。

## 现在能声称与不能声称的内容

可以声称：数学接口、奇偶性、Cartesian atlas、production trainer/sampler 软件闭环、finite-affine/OPD catalogue、occupational parent occurrence、H0-v4 数据资格与 H1a cache 已通过各自 Gate；真实 H1a 已产生可解释的负结果。

不能声称：真实 Alex 生成质量、H1a/H1b 通过、完整 parent blueprint 已训练、tensor condition 能引起 target-separated samples、oracle 已合格、结构已 relaxation、DFT/DFPT 已验证，或发现了新压电材料。

## 恢复任务时需要什么

目前不需要用户补数据或修改阈值。最新结果说明 tangent 类型修复是正确且有物理收益的，
但一个 train pass 内仍未充分学习完整条件期望。下一项工作必须是单独预注册的 H1a
因果诊断，区分时间/噪声分层下的欠拟合、状态特征不足与 objective variance；不能用
同一 checkpoint 增加 seed、延长训练或直接初始化 joint model。任何 tensor、oracle
或物理计算仍必须等待 H1a/H1b 通过。
