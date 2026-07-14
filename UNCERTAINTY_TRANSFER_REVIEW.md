# 从 ICML Tensor-UQ 到 GaugeFlow：迁移判断与实施说明

## 结论

可以迁移，但不能把原 ICML 的完整协方差头直接复制到 GaugeFlow。

应迁移的是三个原则：

1. 在满足几何变换规律的切空间中表达不确定性，而不是在任意坐标中直接回归方差。
2. 用对数尺度/Log-Euclidean 参数化确保方差严格为正，并在训练中与均值联合优化。
3. 将每一步向量场的不确定性沿生成轨迹累积，作为单个生成样本的可靠性指标。

不应迁移的是为 rank-2 对称张量满协方差构造的球谐/四阶 CartesianTensor 头。它针对 6 维 Kelvin--Mandel 输出；GaugeFlow 的压电条件是 18 维 rank-3 张量，满协方差有 171 个独立参数，且正确的不可约分解需要高阶耦合。直接照搬会同时增加计算量、代码复杂度和错误概率，且不会自然解决生成时的一对多结构不确定性。

## 已阅读的 ICML 工作

论文：E:\PAPER\Uncertainty-Aware Equivariant Neural Networks\thesis_icml.tex

实现重点：

- E:\CODE\ICML\equivariant_network.py
- E:\CODE\ICML\stable_loss_implementation.py
- E:\CODE\ICML\voigt_utils.py
- E:\CODE\ICML\train.py

该工作为对称 rank-2 张量预测全 6x6 SPD 协方差。其主要机制是预测对称对数协方差 A，再通过 Sigma = exp(A) 获得严格 SPD 协方差，并以 Mahalanobis 距离和 log-determinant 共同训练。

这一思路本身很有价值，但原实现中包含了适用于旧训练工程的数值 fallback（例如异常时退回 MSE）。GaugeFlow 不复用这些 fallback；新实现对非有限值直接报错。

## Gauge-UQ 的对象到底是什么

GaugeFlow 的首个 UQ 对象不是“压电张量标签本身的满协方差”，而是条件生成过程中每一步速度预测的 aleatoric uncertainty：

- 原子类型连续 logit 速度的不确定性；
- 原子几何速度的不确定性；
- 晶格对数 SPD 坐标速度的不确定性。

这回答的是：在当前时间、当前部分生成的晶体和给定张量轨道下，网络对下一小步应该如何移动有多不确定。

这与下列概念不同，必须在论文与实验中分开报告：

- 生成多样性：同一压电条件下本来就可能有多个结构，这是模型分布的性质，不等于模型不知道。
- 对齐熵：轨道候选的 softmax 熵表示规范/对称性对齐有多模糊；它是单独的 gauge ambiguity 信号。
- epistemic uncertainty：训练数据不足造成的模型参数不确定性。当前单模型的异方差头不估计它；后续可用独立 ensemble 或 Fisher/Laplace 近似研究。

## 几何上正确且低成本的参数化

### 坐标块

分数坐标不是旋转下的笛卡尔向量。Gauge-UQ 先将速度残差转换到当前晶格的笛卡尔切空间：

    r_cart = (v_pred_frac - v_target_frac) @ L

然后每个原子预测一个标量 log sigma，并使用协方差：

    Cov(r_cart) = sigma^2 I_3

其中 sigma = exp(log sigma)。旋转只改变 r_cart 的方向，不改变其范数，所以这个 NLL 对 SO(3) 保持不变；不需要球谐基。

### 类型与晶格块

类型 logit 和当前的 6 维晶格 SPD-log 坐标使用块内各向同性的独立 Gaussian NLL。它们是生成流的内部坐标，不把它们伪称为压电张量的物理协方差。

每个 log standard deviation 由平滑有界函数给出：

    log_sigma = midpoint + radius * tanh(raw)

因此 sigma 永远严格为正，同时避免早期训练中尺度爆炸或塌缩。

## 当前实现

新增文件：

- src/gaugeflow/uncertainty.py：切向异方差 UQ、Gaussian NLL、采样不确定性数据结构。
- tests/test_uncertainty.py：旋转不变性、训练目标、采样传播测试。

修改文件：

- src/gaugeflow/model.py：加入三个不确定性头；坐标方差在笛卡尔切空间定义。
- src/gaugeflow/flow.py：在可选 UQ 模式下联合优化 MSE flow matching 和异方差 NLL，并按 Euler 步长平方累积方差代理。
- scripts/train.py：加入 UQ warmup 与权重开关。

训练接口：

    --uncertainty-weight 0.1
    --uncertainty-warmup-steps 5000

前 5000 步只做确定性 flow matching；随后才启用 UQ NLL。这借鉴了 ICML 工作的先稳定均值、后学习不确定性的训练策略，但不使用其高阶球谐协方差头。

采样时可调用：

    matcher.sample(model, batch, return_uncertainty=True)

它返回生成状态以及三类沿轨迹累积的方差代理和平均对齐熵。

## 与文献的关系

Flow Matching 将生成训练转化为条件路径上的向量场回归，因此速度不确定性是自然的 UQ 位置：[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)。

一项近期工作 UA-Flow 也直接预测异方差速度不确定性，并沿流动力学传播，将结果用于样本可靠性和引导；Gauge-UQ 借鉴这一方向，但为晶体的流形坐标、张量轨道与旋转语义重新设计：[Flow Matching with Uncertainty Quantification and Guidance](https://arxiv.org/abs/2602.10326)。

应避免把 aleatoric 与 epistemic uncertainty 混为一谈。近期 Fisher-Laplace 工作专门指出这种混淆会使生成模型的 epistemic UQ 不可靠；因此 Gauge-UQ 首先明确报告自身是 aleatoric tangent uncertainty：[Quantifying Epistemic Uncertainty in Diffusion Models](https://arxiv.org/abs/2602.09170)。

## 推荐的实验路线

1. 确定性 GaugeFlow 与 Gauge-UQ 在相同训练预算下比较生成有效性和目标响应误差。
2. 在每个压电条件下采样多个结构，对弛豫/重算后的响应误差按预测 UQ 分位数分桶，画 risk--coverage 曲线。
3. 分别分析：切向 UQ、对齐熵、生成样本间离散度。三者不应合并成一个未经校准的“总不确定性”。
4. 在验证集做温度校准；校准参数只能在验证集选择，再冻结到测试集。
5. 后续若需要 epistemic UQ，优先比较小型独立 ensemble 与成本更高的 Fisher/Laplace 方法；不要把单模型异方差尺度解释为数据覆盖不足。

## 对论文叙事的建议

UQ 可以成为 GaugeFlow 的可信生成层，而不是另一个拼接模块。论文中的新主线应保持为：

    张量轨道 + 响应场 + 稳定子商空间 + 流形生成

Gauge-UQ 是该主线自然导出的可靠性机制：它估计生成轨迹中何处不确定、何时规范对齐含混，并支持风险受控的候选筛选。不要把 ICML rank-2 满协方差工作重新包装成 GaugeFlow 的主要创新；二者共享几何 UQ 原理，但问题、表示和计算方案不同。
