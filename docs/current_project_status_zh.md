# GaugeFlow 当前实现、方法演进与实验状态（2026-07-18）

## 一句话结论

GaugeFlow 已经从早期连续 logit/ODE 原型重构为混合离散—连续晶体扩散框架，并完成 Cartesian tensor-orbit conditioner、反向采样软件闭环和 H0 数据/群论资格化；但尚未完成 Alex-MP-20 全量缓存、真实数据 H1a 训练或 tensor-conditioned 生成验证，因此当前不能声称已生成满足目标压电张量的晶体。

本项目现已暂停在 H1a 数据入口之前。后续唯一允许的步骤是完成已冻结的 P1 packed cache 构建与独立审计；H1b、H2--H6、真实 tensor、oracle、relaxation、DFT 和 DFPT 均未启动。

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

## 当前暂停点

已冻结 `h1a_p1_structure_cache_v1` 协议，但尚未运行。协议要求保留全部 675,204 条 H0-A 结构、执行可证明的 Niggli \(GL(3,\mathbb Z)\) basis change，并把 ID、split、prototype、space group、Niggli transform 等全部留在 audit index，不能进入 denoiser。

代码中已有 fail-closed 的 `PackedAlexP1Dataset` reader；它只接受 `qualified=true` 且 hash 匹配的 cache，并默认只暴露 atom tokens、fractional coordinates、lattice 和 graph size。当前没有正式 cache artifact、没有 builder/auditor 结果，也没有 H1a checkpoint。

## 现在能声称与不能声称的内容

可以声称：数学接口、奇偶性、Cartesian atlas、production trainer/sampler 软件闭环、finite-affine/OPD catalogue、occupational parent occurrence 和 H0-v4 数据资格已通过各自冻结 Gate。

不能声称：真实 Alex 生成质量、H1a/H1b 通过、完整 parent blueprint 已训练、tensor condition 能引起 target-separated samples、oracle 已合格、结构已 relaxation、DFT/DFPT 已验证，或发现了新压电材料。

## 恢复任务时需要什么

目前不需要用户补数据或修改阈值。恢复时只需确认允许继续 H1a data-plane，并保证 `E:/DATA/T2C-Flow` 可读；cache 构建是 CPU/磁盘任务。之后若进入正式训练，需要 WSL CUDA 环境和空闲的 RTX 4060 Ti 16 GB。任何 tensor、oracle 或物理计算仍必须等待前序 Gate 通过。
