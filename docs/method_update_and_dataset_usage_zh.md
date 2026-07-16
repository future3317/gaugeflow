请仔细阅读下面的方法修改意见，结合你刚刚做的实验的发现和修正，这里面的你也可以思考一下去做idea的微调，然后有关数据：

T2C-Flow 的数据根目录是：

E:\DATA\T2C-Flow

不要把大型数据复制到 E:\CODE\T2C-Flow 代码仓库中。

主要数据路径：

1. 统一压电数据（JARVIS + Materials Project）
E:\DATA\T2C-Flow\processed\piezo_unified.parquet

2. 分来源压电数据
JARVIS：
E:\DATA\T2C-Flow\processed\jarvis_piezo.parquet

Materials Project：
E:\DATA\T2C-Flow\processed\materials_project_piezo.parquet

JARVIS/MP 对应关系：
E:\DATA\T2C-Flow\processed\jarvis_mp_piezo_overlap.parquet

3. GaugeFlow 已审计的 JARVIS 张量数据
E:\DATA\T2C-Flow\processed\tensororbit_jarvis_v2_full_o3_v2

4. Alex-MP-20（生成模型结构先验）
E:\DATA\T2C-Flow\raw\huggingface\Alex-MP-20

5. MatPES-PBE（势能面/能量—力—应力教师）
E:\DATA\T2C-Flow\raw\huggingface\MatPES-PBE

6. PhononDB（非零 q、声子带、动力学稳定性）
原始官方 ZIP：
E:\DATA\T2C-Flow\raw\phonondb\archives
然后我再让另一个codex做从 phonopy_params 中的位移—力数据重建 force constants 和动力学矩阵，这个你就先不用管，你就先改项目代码和论文methods等就行
推荐训练索引：
E:\DATA\T2C-Flow\processed\phonondb_mode_v1\index.parquet

提取后的 phonopy 核心文件：
E:\DATA\T2C-Flow\processed\phonondb_mode_v1\phonopy_params


7. JARVIS-DFPT（Γ 点极性模式、BEC、介电和压电分解）
该数据不在 T2C-Flow 中重复保存，直接读取：

E:\DATA\PiezoJet\processed\jarvis_dfpt_v9_full_public

完整 internal strain Lambda 只能使用：
E:\DATA\PiezoJet\processed\jarvis_strain_completion_v10_zero_dimensional_fix

8. 数据说明和清单
E:\DATA\T2C-Flow\README.md
E:\DATA\T2C-Flow\MANIFEST.json
E:\DATA\T2C-Flow\processed\phonondb_mode_v1\SCHEMA.md

所有联合训练都必须保留 source_id/source_database。
JARVIS、MP、PhononDB 的绝对数值不能默认属于同一计算分布。

9.LeMat-BulkUnique（材料结构基础数据）
E:\DATA\LeMat-BulkUnique

# 一、四类数据各自承担什么角色

建议严格分离四个数据域：

[
\mathcal D
==========

\mathcal D_{\rm struct}
\cup
\mathcal D_{\rm mode}
\cup
\mathcal D_{\rm PES}
\cup
\mathcal D_{\rm tensor}.
]

## 1. Alex-MP-20：结构分布与父子结构分解

[
\mathcal D_{\rm struct}
=======================

{x_i=(a_i,f_i,L_i)}.
]

用途：

* 训练现有 tensor-free parent generator；
* 学习元素、空间群、Wyckoff、晶格和坐标的结构先验；
* 从低对称结构中自动寻找候选高对称父相；
* 拟合 parent–mode–child 分解；
* 构造 formula/prototype-disjoint split。

不要使用 Alex-MP-20 的能量与 MatPES-PBE 能量直接混合。它在这里主要提供**结构分布**。MatterGen 的公开基础模型正是以 Alex-MP-20 作为无条件结构训练集。

## 2. JARVIS-DFPT：(\Gamma) 点极性模式监督

JARVIS-DFPT 对 5,015 个非金属材料提供：

* (\Gamma) 点声子；
* Born effective charges；
* piezoelectric tensors；
* dielectric tensors；
* infrared information。([arXiv][1])

它最适合监督：

* (\Gamma)-point polar soft modes；
* 模式是否具有电活性；
* mode effective charge；
* 极性模式与压电响应之间的关系。

它**不能单独支持 zone-boundary mode 或大超胞相变**，因为其主要声子标签是 (\Gamma) 点。

## 3. PhononDB：非零波矢和完整动力学监督

Phonon Database at Kyoto University 包含约 10,034 个材料，并被用于完整 phonon-band 和拓扑声子分析。([arXiv][2])

根据你下载文件的实际 schema，优先使用：

* force constants；
* dynamical matrices；
* (q)-point frequencies；
* eigenvectors；
* primitive/supercell mapping。

它承担：

* 非零 (k) 模式；
* zone-boundary instability；
* 低指数超胞；
* 模式子空间和简并结构；
* 动力学稳定性监督。

JARVIS 和 PhononDB 的绝对频率不能不加区分地合并。计算泛函、赝势和收敛参数可能不同；建议保留 `source_id`，分别校准频率尺度。

## 4. MatPES-PBE：势能面教师，不是生成数据主体

需要区分：

* **MatPES-PBE 数据集**；
* **在 MatPES-PBE 上训练好的冻结 MLIP**。

只有后者才是教师：

[
E_T(x),\qquad
F_T(x)=-\nabla_r E_T(x),\qquad
\sigma_T(x).
]

MatPES 最新公开版本为 2025.2，PBE 和 r2SCAN 版本分别发布；官方也提供 TensorNet、CHGNet 和 M3GNet 等 MatPES 训练模型。

推荐做法：

* 使用冻结的 `TensorNet-PES-MatPES-PBE-2025.2`；
* 最好再配一个不同架构的 MatPES-PBE 模型作为 disagreement teacher；
* 不在 reverse sampling 中用教师梯度直接引导；
* 教师只用于离线标注、表示迁移、过滤和辅助损失。

---

# 二、修改后模型的总体因子分解

不要把 parent、child、mode 和 tensor 一次性塞入一个巨大扩散状态。推荐拆成两个生成阶段：

[
\boxed{
\text{父结构生成器}
\quad+\quad
\text{条件对称性破缺生成器}
}
]

完整概率模型为

[
\begin{aligned}
p_\Theta(x_{\rm c}\mid[e])
={}&
\sum_{b_{\rm p},x_{\rm p},d}
p_{\theta_{\rm p}}
\left(
b_{\rm p},x_{\rm p}
\mid c_{\rm inv}([e])
\right)
\
&\times
p_{\theta_{\rm d}}
\left(
d\mid x_{\rm p},[e]
\right)
p_{\theta_{\rm q}}
\left(
y\mid x_{\rm p},d,[e]
\right)
\
&\times
\delta!
\left[
x_{\rm c}
=========

\Phi(x_{\rm p},d,y)
\right].
\end{aligned}
]

其中：

* (b_{\rm p})：父相离散蓝图；
* (x_{\rm p})：生成的父结构；
* (d)：对称性破缺蓝图；
* (y=(s,\varepsilon,\delta r))：连续模式幅度、应变和残差；
* (\Phi)：确定性的 parent-to-child reconstruction；
* (x_{\rm c})：最终子相结构。

这样现有模型成为严格的特殊情形：

[
d=\varnothing
\quad\Longrightarrow\quad
x_{\rm c}=x_{\rm p}.
]

这是非常重要的设计性质：新模型不是推翻当前模型，而是把它嵌套为 exact-symmetry branch。

---

# 三、第一层：父相蓝图

定义

[
b_{\rm p}
=========

\left(
G_{\rm p},
M,
{W_m,\nu_m,a_m}_{m=1}^{M}
\right).
]

其中：

* (G_{\rm p})：父空间群；
* (W_m)：父相 Wyckoff orbit；
* (\nu_m)：multiplicity；
* (a_m)：species。

使用现有 revised hybrid generator 生成：

[
x_{\rm p}
=========

(a_{\rm p},f_{\rm p},L_{\rm p}).
]

这一层完全复用现有：

* absorbing categorical element process；
* wrapped-coordinate quotient；
* log-volume/trace-free-log-metric lattice process；
* symmetry blueprint；
* exact asymmetric-unit expansion；
* tensor-free S1a trainer 和 reverse sampler。

父生成器必须先独立通过 S1a。层级蓝图不能替代尚未完成的 parent generation qualification。

---

# 四、第二层：对称性破缺蓝图

定义

[
d
=

\left(
B,
\left{
k_\ell,\Gamma_\ell,c_\ell,z_\ell
\right}_{\ell=1}^{K}
\right).
]

## 符号含义

* (B\in\mathbb Z^{3\times3})：Hermite-normal-form 超胞矩阵；
* (k_\ell)：父相 Brillouin zone 中的 commensurate wave vector；
* (\Gamma_\ell)：(k_\ell) little group 的不可约表示；
* (c_\ell)：order-parameter-direction/isotropy branch；
* (z_\ell\in{0,1})：模式是否激活；
* (K)：同时激活的模式数。

第一版建议固定：

[
\det B\leq4,
\qquad
K\leq2.
]

这已经能够覆盖：

* (\Gamma)-point polar displacement；
* 反极性模式；
* 低指数 octahedral rotations；
* 单一 zone-boundary instability；
* 两模式耦合；
* 1、2、3、4 倍 commensurate supercells。

暂不处理更大超胞、缺陷和无序。

---

# 五、不要直接扩散任意 irrep 向量：先采样 OPD 分支

一个 irrep 可能有多种 order-parameter direction，对应不同 isotropy subgroup。建议预先为每个

[
(G_{\rm p},B,k,\Gamma)
]

构造有限 OPD catalog：

[
\mathcal C_{G_{\rm p},B,k,\Gamma}
=================================

\left{
(c,U_c,H_c)
\right}.
]

其中：

* (H_c\subseteq G_{\rm p})：对应 isotropy subgroup；
* (U_c)：该 OPD 固定子空间的正交基；
* 连续序参量写成

[
q_\ell=U_{c_\ell}s_\ell.
]

这里

[
s_\ell\in\mathbb R^{d_{\ell,c}}
]

才是扩散变量。

优点是：**子群在扩散前已经由离散蓝图确定**，不会因为 noisy (q_t) 而在训练中不断跳变。

模式连续过程为

[
s_{\ell,t}
==========

\alpha_t s_{\ell,0}
+
\sigma_t\xi_\ell,
\qquad
\xi_\ell\sim\mathcal N(0,I).
]

模式激活使用离散 absorbing/masked process：

[
z_{\ell,t}\in{\mathrm{ACTIVE},\mathrm{MASK}}.
]

---

# 六、最终子群

先定义与超胞 (B) 兼容的父群子集：

[
G_{\rm p}^{B}
=============

\left{
g\in G_{\rm p}:
g\Lambda_B=\Lambda_B
\right}.
]

对每个激活模式，OPD 分支已经给出 (H_{\ell,c_\ell})。最终 generic child subgroup 为

[
\boxed{
H(d)
====

G_{\rm p}^{B}
\cap
\bigcap_{\ell:z_\ell=1}
H_{\ell,c_\ell}.
}
]

如果所有模式均未激活：

[
z_\ell=0\ \forall\ell
\quad\Longrightarrow\quad
H(d)=G_{\rm p}.
]

多个模式耦合时，子群自然成为各稳定子群的交。这样能够表达 hybrid improper 路径，而不是只允许单个 polar mode。

---

# 七、父相到子相的重建

设父结构扩展到超胞后为

[
(r_{\rm p}^{B},L_{\rm p}^{B}),
\qquad
L_{\rm p}^{B}=B L_{\rm p}.
]

## 1. 对称适配位移

令 (\Psi_\ell) 是质量加权、正交归一的 mode basis：

[
\Psi_\ell^\top\Psi_\ell=I.
]

物理 Cartesian displacement 为

[
\Delta r_{\rm mode}
===================

M^{-1/2}
\sum_{\ell=1}^{K}
\Psi_\ell U_{c_\ell}s_\ell .
]

## 2. 子群允许的应变

构造 (H(d))-invariant symmetric strain basis：

[
\varepsilon_H
=============

\sum_{\alpha=1}^{d_\varepsilon}
\eta_\alpha E_\alpha^{H}.
]

晶格为

[
L_{\rm c}
=========

B L_{\rm p}
\exp(\varepsilon_H).
]

## 3. 小残差

加入一个严格受限的残差分支：

[
\Delta r_{\rm res}
==================

P_{H(d)}\delta r,
]

其中

[
P_H
===

\frac{1}{|H|}
\sum_{h\in H}
\rho_{\rm disp}(h)
]

是位移表示上的 Reynolds projector。

最终坐标：

[
r_{\rm c}
=========

r_{\rm p}^{B}
+
\Delta r_{\rm mode}
+
\Delta r_{\rm res},
]

[
\boxed{
f_{\rm c}
=========

\operatorname{wrap}
\left(
r_{\rm c}L_{\rm c}^{-1}
\right).
}
]

残差分支不能承担主要结构变化。建议设置：

[
\operatorname{RMS}(\Delta r_{\rm res})
<
\tau_{\rm res},
]

初期可取 (0.05\sim0.10) Å。长期超过阈值说明 parent path 或 mode catalog 不充分。

---

# 八、父空间群不能再直接接受压电 hard router

这是修改中最关键的一点。

一个中心对称父相可能通过反演奇模式生成非中心对称压电子相。因此不能使用

[
r_{G_{\rm p}}([e])
]

直接拒绝父相。

应当对父相可达的 child paths 边缘化：

[
\boxed{
p(G_{\rm p}\mid[e])
\propto
p_0(G_{\rm p}\mid c_{\rm inv})
\sum_{d\in\mathcal C(G_{\rm p})}
p_0(d\mid G_{\rm p})
\exp
\left[
-\beta r_{H(d)}([e])^2
\right].
}
]

采样父相后：

[
p(d\mid x_{\rm p},[e])
\propto
p_0(d\mid x_{\rm p})
\exp
\left[
-\beta r_{H(d)}([e])^2
\right].
]

也就是说：

* 父相只需要存在一条兼容目标 tensor 的子群路径；
* 最终物理兼容性由 (H(d)) 决定；
* exact branch 对应 (H=G_{\rm p})。

---

# 九、GaugeFlow atlas 应该注入到哪里

完整 Cartesian atlas 不必参与所有离散决策。

## 只用 orbit-invariant token 的部分

在尚无具体几何 frame 时：

[
p(b_{\rm p}\mid[e]),
\qquad
p(d\mid b_{\rm p},[e])
]

使用：

* tensor orbit invariants；
* physical-zero flag；
* (r_{H(d)}([e]))；
* 组成和设计约束。

## 使用完整 atlas 的部分

在父结构 (x_{\rm p}) 已存在后，Cartesian atlas 用于：

[
s_\theta(s_t\mid x_{\rm p},d,[e]),
]

[
s_\theta(\varepsilon_t\mid x_{\rm p},d,[e]),
]

[
s_\theta(\delta r_t\mid x_{\rm p},d,[e]).
]

因此完整 condition 路径变为：

[
[e]
\rightarrow
\text{reachable child subgroup}
\rightarrow
\text{mode selection}
\rightarrow
\text{atlas-aligned mode amplitude}.
]

这比让 atlas 直接决定父空间群更自然，也避免在没有具体结构 frame 时过早做 alignment。

---

# 十、JARVIS-DFPT 和 PhononDB 如何监督模式

## 1. 基础动力学方程

对动力学矩阵：

[
D(k)\Psi_{\lambda k}
====================

\omega_{\lambda k}^{2}
\Psi_{\lambda k}.
]

训练目标不要直接回归带符号、尺度差异很大的原始频率。拆成：

[
y_{\rm soft}
============

\mathbf 1[\omega^2<0],
]

[
y_{\rm mag}
===========

\log
\left(
1+
\frac{|\omega^2|}{\omega_0^2}
\right).
]

损失：

[
\mathcal L_\omega
=================

\mathcal L_{\rm BCE}(\hat y_{\rm soft},y_{\rm soft})
+
\lambda_\omega
\mathcal L_{\rm Huber}(\hat y_{\rm mag},y_{\rm mag}).
]

JARVIS 和 PhononDB 使用独立 source calibration：

[
\hat y
======

h_{\rm shared}(x,k,\Gamma)
+
b_{\rm source}.
]

不要把两个来源的 (\omega^2) 当作完全同一标度。

## 2. 简并模式用子空间监督

模式存在符号、排序和简并 gauge。不要比较单个 eigenvector：

[
|\hat\Psi-\Psi|^2.
]

应比较投影算子：

[
P_{\lambda}
===========

\Psi_{\lambda}\Psi_{\lambda}^{\top},
]

[
\mathcal L_{\rm subspace}
=========================

\left|
\hat P_\lambda-P_\lambda
\right|_F^2.
]

## 3. JARVIS 极性模式监督

利用 Born effective charge：

[
\boxed{
Z^{\rm mode}_{\lambda,i}
========================

\sum_{\kappa,\alpha}
Z^{*}*{\kappa,i\alpha}
\frac{
\Psi*{\lambda,\kappa\alpha}
}{
\sqrt{M_\kappa}
}.
}
]

该量用于判断：

* 模式是否 polar；
* 哪些 polarization directions 被激活；
* 模式与压电 tensor response field 的相关性。

可加入：

[
\mathcal L_{\rm polar}
======================

\left|
\widehat Z_{\lambda}^{\rm mode}
-------------------------------

Z_{\lambda}^{\rm mode}
\right|^2.
]

注意：模式有较大 effective charge 不等于最终一定具有大压电响应；还需要 strain coupling、elasticity 和相稳定性。

---

# 十一、MatPES-PBE 教师如何进入模型

## 1. 构造针对性 mode scans

从 Alex-MP-20 父结构 (x_{\rm p}) 出发，沿选定模式采样：

[
x(s,\eta)
=========

\Phi
\left(
x_{\rm p},d,
(s,\eta,0)
\right).
]

建议初期采样范围：

* displacement RMS：(0.01)–(0.20) Å；
* strain：不超过 (3%)；
* 每个 parent–mode 路径采样 8–32 个点；
* 同时包含 (+s) 和 (-s)。

教师计算：

[
E_T(x),\qquad
F_T(x),\qquad
\sigma_T(x).
]

## 2. 投影到序参量空间

广义模式力为

[
\boxed{
g_{\ell,T}
==========

U_{c_\ell}^{\top}
\Psi_\ell^{\top}
M^{-1/2}F_T.
}
]

因为 (F_T=-\partial E_T/\partial r)，所以 (g_{\ell,T}) 是沿序参量方向的负能量梯度。

## 3. 辅助 invariant energy head

可新增一个只用于训练的低维势能头：

[
\mathcal F_\phi
===============

\mathcal F_\phi
\left(
x_{\rm p},
d,
s,
\eta
\right),
]

并要求它在 parent group 下不变。

定义：

[
g_{\ell,\phi}
=============

-\frac{\partial\mathcal F_\phi}{\partial s_\ell},
]

[
\sigma_\phi
===========

\frac{1}{V}
\frac{\partial\mathcal F_\phi}{\partial\varepsilon_H}.
]

辅助损失：

[
\begin{aligned}
\mathcal L_{\rm PES}
={}&
w_E
\rho
\left(
\frac{\Delta\mathcal F_\phi-\Delta E_T}{N}
\right)
\
&+
w_g
\sum_\ell
\rho
\left(
g_{\ell,\phi}-g_{\ell,T}
\right)
\
&+
w_\sigma
\rho
\left(
\sigma_\phi-\sigma_T
\right).
\end{aligned}
]

这个 energy head 的表示可以与 mode denoiser 共享，但它**不应直接替代扩散 score**，也不应在采样时作为无约束梯度 guidance。

## 4. 教师的三个用途

MatPES-PBE teacher 只用于：

1. 模式表示预训练；
2. 合成 mode scan 标注；
3. 过滤明显高能、碰撞和 OOD 扰动。

最终验证仍需独立 MLIP/DFT。MatterGen 的公开文档也明确提醒，MLFF relaxation 和能量评价不能替代最终 DFT 结论。

---

# 十二、如何从 Alex-MP-20 自动得到 parent–child 数据

Alex-MP-20 只有结构，不天然提供 parent label。需要离线构造候选路径。

对每个低对称结构 (x)：

## 1. 搜索候选父相

限制：

[
G_x\subseteq G_{\rm p},
\qquad
[G_{\rm p}:G_x]\leq4,
]

并要求：

* composition 不变；
* Wyckoff splitting 合法；
* atom mapping 一一对应；
* cell index (\det B\leq4)；
* strain 不超过阈值；
* reconstructed child 与原结构匹配。

## 2. 拟合模式

对每个候选 parent (j)，求解

[
\min_{s,\eta,\delta r}
d_{\rm struct}
\left(
x,
\Phi(x_{\rm p}^{(j)},d^{(j)},y)
\right)^2
+
\lambda_{\rm res}|\delta r|^2.
]

## 3. 路径代价

[
\begin{aligned}
C_j
={}&
\lambda_{\rm rec}d_{\rm rec,j}^{2}
+
\lambda_\varepsilon
|\varepsilon_j|*F^{2}
+
\lambda_B\log\det B_j
\
&+
\lambda_K K_j
+
\lambda_q
\sum*\ell|s_{\ell,j}|
+
\lambda_r|\delta r_j|^2.
\end{aligned}
]

保留前 (K_{\rm p}) 条路径，例如：

[
K_{\rm p}\leq4.
]

不要硬选择唯一 parent。训练时边缘化：

[
\boxed{
\mathcal L_{\rm path}
=====================

-\log
\sum_{j=1}^{K_{\rm p}}
\pi_\phi(j\mid x)
\exp(-\mathcal L_j).
}
]

这样可以避免“人为 canonical parent”问题。

---

# 十三、实际训练数据池

最终不要只有一个统一 dataset loader，而是四个可独立采样的数据池。

## A. 结构池

[
\mathcal D_{\rm struct}
=======================

{x}_{\rm Alex-MP-20}.
]

用于 parent generator。

## B. 父子路径池

[
\mathcal D_{\rm pair}
=====================

{
x_{\rm c},
x_{\rm p},
B,
k,\Gamma,c,
s,\varepsilon,\delta r
}.
]

来自 Alex-MP-20 自动分解。

## C. 模式池

[
\mathcal D_{\rm mode}
=====================

{
x,k,\Gamma,
\omega^2,
P_{\rm eig},
Z_{\rm mode},
\text{source}
}.
]

来自 JARVIS-DFPT 和 PhononDB。

## D. 势能扫描池

[
\mathcal D_{\rm scan}
=====================

{
x_{\rm p},d,s,\varepsilon,
E_T,F_T,\sigma_T,u_T
}.
]

来自 Alex parent + mode displacement + MatPES-PBE teacher。

所有损失都使用 label masks，不要求每条数据同时拥有全部标签。

---

# 十四、总损失

[
\boxed{
\begin{aligned}
\mathcal L
={}&
\mathcal L_{\rm parent}
+
\lambda_d\mathcal L_{\rm distortion\ blueprint}
+
\lambda_s\mathcal L_{\rm mode\ score}
\
&+
\lambda_\varepsilon\mathcal L_{\rm strain}
+
\lambda_r\mathcal L_{\rm residual}
+
\lambda_{\rm rec}\mathcal L_{\rm reconstruction}
\
&+
\lambda_\omega\mathcal L_{\rm phonon}
+
\lambda_Z\mathcal L_{\rm polar}
+
\lambda_T\mathcal L_{\rm PES}.
\end{aligned}
}
]

其中：

* (\mathcal L_{\rm parent})：现有 element/coordinate/lattice/blueprint losses；
* (\mathcal L_{\rm distortion\ blueprint})：(B,k,\Gamma,c,z) 的 categorical loss；
* (\mathcal L_{\rm mode\ score})：(s_t) 的 diffusion loss；
* (\mathcal L_{\rm strain})：(\eta_t) 的 diffusion loss；
* (\mathcal L_{\rm residual})：投影 residual score；
* (\mathcal L_{\rm reconstruction})：child reconstruction；
* (\mathcal L_{\rm phonon})：频率符号、大小和子空间；
* (\mathcal L_{\rm polar})：mode effective charge；
* (\mathcal L_{\rm PES})：MatPES teacher。

---

# 十五、代码层面的模块变化

不需要重写现有 GaugeFlow；应按模块扩展。

## 保持不变

* hybrid element process；
* wrapped-coordinate quotient；
* parent lattice chart；
* 230 space-group tables；
* exact asymmetric-unit expansion；
* full-(\mathrm O(3)) Reynolds router；
* S0.4.1 Cartesian atlas；
* denoiser time/condition FiLM；
* physical-zero/null separation。

## 重命名或重新解释

现有：

```text
SymmetryBlueprint
```

变成：

```text
ParentBlueprint
```

它描述父相，不再默认等于最终结构空间群。

## 新增四个核心对象

```text
ModeCatalog
DistortionBlueprint
ModeDiffusionState
ChildReconstructor
```

概念数据结构：

```text
ParentBlueprint:
    parent_space_group
    wyckoff_orbits
    multiplicities
    species

DistortionBlueprint:
    supercell_matrix
    modes: [(k, irrep, opd_class, active)]
    child_subgroup

ModeDiffusionState:
    mode_amplitudes
    child_strain
    residual_displacements

HierarchicalSample:
    parent_structure
    distortion_blueprint
    child_structure
```

## 新生成路径

```text
sample parent blueprint
→ sample parent element/coordinate/lattice state
→ exact parent expansion
→ enumerate valid distortion candidates
→ sample B, k, irrep, OPD and activation
→ diffuse/reverse-sample mode amplitudes and strain
→ reconstruct child structure
→ identify realized child symmetry
→ collision and validity checks
→ MLIP relaxation
```

---

# 十六、训练顺序

## H0：数据资格化

先输出：

* Alex-MP-20 实际版本和 hash；
* MatPES-PBE 2025.1/2025.2；
* JARVIS 和 PhononDB schema；
* 四个数据集的结构交集；
* 元素覆盖；
* parent decomposition 成功率；
* mode-data 匹配率。

## H1：现有 S1a

只训练 exact parent generator。

这是不可跳过的 gate。

## H2：模式表征预训练

使用 JARVIS-DFPT + PhononDB：

* frequency sign；
* frequency magnitude；
* mode subspace；
* polar effective charge；
* source calibration。

## H3：parent–child reconstruction

只训练：

[
x_{\rm p},d,y
\rightarrow
x_{\rm c}.
]

先验证模式参数化能否重建 Alex-MP-20 的低对称结构。

## H4：MatPES-PBE 势能辅助

在合成 mode scans 上训练 energy/force/stress auxiliary heads。

## H5：tensor-free hierarchical generation

冻结或低学习率更新 parent generator，训练：

* distortion blueprint；
* mode diffusion；
* strain diffusion；
* residual head。

## H6：tensor-conditioned fine-tuning

最后才把 Cartesian atlas 接入 mode-amplitude denoiser。

建议最初冻结：

* parent element embedding；
* parent 前 2–3 个 message blocks；
* mode catalog；
* MatPES teacher。

主要训练：

* child-compatible router；
* distortion blueprint heads；
* condition-FiLM；
* mode/strain denoiser；
  -最后 1–2 个共享 blocks。

---

# 十七、数据划分必须提前完成

在任何 parent search、mode scan 和教师标注之前，先按：

[
(\text{reduced formula},\text{prototype cluster})
]

划分 Alex-MP-20。

随后要求：

* 一个 child 的所有 parent candidates 留在同一 split；
* 由该结构生成的所有 mode scans 留在同一 split；
* JARVIS/PhononDB 匹配结构跟随 Alex split；
* 同一结构的不同 standard cells 不得跨 split；
* MatPES teacher 可以参与训练，但不能作为最终独立评价器；
* 最终测试使用独立 MLIP、DFT 和 DFPT。

---

# 十八、第一版应明确限制

建议第一版只宣称：

> 有序、化学计量、周期性无机晶体中的低指数 commensurate 对称性破缺生成。

第一版不加入：

* partial occupancy；
* substitutional disorder；
* vacancies；
* charged defects；
* incommensurate phases；
* (\det B>4) 的超胞；
* finite-temperature ensemble。

这些应在层级蓝图被证明有效后，再作为独立模块增加。

---

## 最终模型的一行定义

[
\boxed{
[e]
\rightarrow
\underbrace{x_{\rm parent}}*{\text{Alex-MP-20 prior}}
\rightarrow
\underbrace{(B,k,\Gamma,\mathrm{OPD})}*{\text{JARVIS/PhononDB}}
\rightarrow
\underbrace{(s,\varepsilon,\delta r)}*{\text{hybrid diffusion}}
\rightarrow
\underbrace{x*{\rm child}}*{\text{subgroup constrained}}
\rightarrow
\underbrace{E,F,\sigma}*{\text{MatPES-PBE teacher}}
\rightarrow
\text{independent DFT/DFPT}.
}
]

这个版本保留了当前模型最强的部分——matched state spaces、exact group actions、完整 tensor orbit 和 Cartesian atlas——同时让真实压电材料中最重要的软模、模式耦合和低对称子相成为模型内部变量，而不是事后扰动或过滤。

[1]: https://arxiv.org/abs/1910.01183?utm_source=chatgpt.com "High-throughput Density Functional Perturbation Theory and Machine Learning Predictions of Infrared, Piezoelectric and Dielectric Responses"
[2]: https://arxiv.org/abs/2412.01280?utm_source=chatgpt.com "Realization of Hopf-link structure in phonon spectra: Symmetry guidance and High-throughput investigation"
