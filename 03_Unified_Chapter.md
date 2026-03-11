# 基于迭代学习控制先验的模型预测路径积分控制

---

## 1 引言与动机

在移动机器人的高速轨迹跟踪任务中，名义运动学模型通常假设车轮与地面之间不存在滑移。然而，实际运行时地面摩擦力的空间不均匀性、车体惯性效应以及执行器延迟等因素，会使机器人在特定空间位置处反复出现相似的跟踪偏差。这类偏差的显著特征是**空间重复性**——同一弯道、同一段坡面在每次经过时均会产生类似的误差模式。

传统反馈控制器（如 PID、标准 MPC）只能在误差发生之后进行修正。迭代学习控制 (Iterative Learning Control, ILC) 则利用重复任务的结构性，从历史迭代中学习前馈补偿信号，使系统在下一次经过同一位置时能够**提前**抵消可预测的扰动。然而，ILC 与在线优化控制器的融合方式是一个开放的设计问题。经典 ILC-MPC 方法（如 Rosolia 和 Borrelli 提出的 Learning MPC 系列）在确定性优化框架中修正参考轨迹或约束集，但这些方法不直接适用于基于采样的随机优化器。

模型预测路径积分 (MPPI) 控制器通过蒙特卡洛采样与信息论加权实现无梯度的随机最优控制，天然适用于非凸代价函数和非线性动力学。其采样分布的均值参数为 ILC 学习成果的注入提供了一个自然的接口——将 ILC 偏置注入为采样分布的**均值偏移 (Mean Shift)**，即作为一种控制先验 (Control Prior)。该策略与在输出端直接叠加 ILC 偏置的经典做法有本质区别：先验注入使 ILC 偏置进入 MPPI 的代价评估循环**内部**，所有采样轨迹均在含偏置的控制量下进行仿真和评估，MPPI 的代价函数对偏置效果拥有完整的"裁决权"。这一设计在保留避障安全性的前提下，使采样分布集中于经验指示的高质量区域，显著提升采样效率与跟踪精度。

本章完整阐述这一融合控制架构的理论与实现。全章组织如下：§2 给出系统描述与参考路径的数学表示；§3 建立引入侧向滑移的三维运动模型；§4 构建空间索引 ILC 的学习框架（记忆库、更新律与门控机制）；**§5 作为本章的核心贡献，论证将 ILC 偏置注入为 MPPI 采样先验（而非输出端直接叠加）构成更优的融合架构**，并从信息论角度阐明其提升采样效率的机制；§6–§8 给出 MPPI 控制器的代价函数、在线滑移估计与信息论加权更新；§9–§10 描述从三维体速度到八维轮指令的升维映射与滑移补偿层；§11 分析 ILC-MPPI 融合架构的收敛性与协同稳定性；§12 总结全章。

---

## 2 问题建模

### 2.1 系统描述

本系统控制一台四轮全向转向 (Swerve Drive) 移动机器人。系统状态向量定义为

$$
\mathbf{x} = \begin{bmatrix} x & y & \psi \end{bmatrix}^\top \in \mathbb{R}^3
$$

其中 $(x, y)$ 为全局坐标系下的位置，$\psi$ 为航向角。控制输入为体坐标系速度指令

$$
\mathbf{u} = \begin{bmatrix} v_x^{cmd} & v_y^{cmd} & \omega^{cmd} \end{bmatrix}^\top \in \mathbb{R}^3
$$

分别表示前向速度、侧向速度和角速度指令。里程计还提供体坐标系下的实际速度观测 $(v_x, v_y, \omega)$，供 ILC 误差计算和滑移估计使用，但它们不作为 MPPI 前向仿真的状态变量。

名义运动学模型（理想无滑移假设）为

$$
\dot{x} = v_x \cos\psi - v_y \sin\psi, \quad
\dot{y} = v_x \sin\psi + v_y \cos\psi, \quad
\dot{\psi} = \omega
$$

实际运行中，地面摩擦力的空间变化和执行器响应延迟使实际速度偏离指令值。将这一未建模偏差表示为

$$
\mathbf{v}_{actual}(s) = \mathbf{u}(s) + \boldsymbol{\delta}(s)
$$

其中 $\boldsymbol{\delta}(s) \in \mathbb{R}^3$ 为与空间位置 $s$ 相关的残差扰动，正是 ILC 需要学习和补偿的目标。

### 2.2 参考路径的表示

参考路径以有序路径点序列给出：

$$
\mathcal{P} = \left\{(\mathbf{p}_i, \psi_i)\right\}_{i=0}^{N-1}, \quad \mathbf{p}_i = (x_i, y_i)^\top
$$

其中 $N$ 为路径点总数，$\psi_i$ 为第 $i$ 个路径点处的切线方向角。对路径做弧长参数化，定义累积弧长

$$
s_0 = 0, \qquad s_i = s_{i-1} + \|\mathbf{p}_i - \mathbf{p}_{i-1}\|_2, \quad i = 1, \ldots, N-1
$$

路径总长度记为 $L = s_{N-1}$。弧长数组 $\{s_i\}$ 在后续的先验索引构建与圈次检测中起核心作用。

---

## 3 引入滑移的三维运动模型

### 3.1 名义运动学

四轮全向转向 (Swerve Drive) 机器人的体坐标系控制输入为

$$
\mathbf{u} = \begin{bmatrix} v_x \\ v_y \\ \omega \end{bmatrix}
$$

分别表示前向速度、侧向速度和角速度。在理想无滑移假设下，全局坐标系中的运动学方程为

$$
\dot{x} = v_x \cos\psi - v_y \sin\psi, \quad
\dot{y} = v_x \sin\psi + v_y \cos\psi, \quad
\dot{\psi} = \omega
$$

其中 $\psi$ 为航向角。该模型假设机器人的实际速度严格等于指令速度，不存在侧向偏移。

### 3.2 滑移耦合效应

在高速转弯时，由于离心惯性力与轮胎-地面接触力的非线性耦合，机器人的实际侧向速度会偏离指令值。这种耦合效应的物理本质是：当机器人同时执行前向运动 ($v_x \neq 0$) 和转向运动 ($\omega \neq 0$) 时，轮胎的侧偏刚度有限，在离心力作用下产生不可忽视的侧向滑移。

本系统采用一阶耦合滑移模型来描述这一效应。定义标量滑移因子 $K_s \geq 0$，引入滑移修正后的有效侧向速度为

$$
v_{y,eff} = v_y + \underbrace{(-K_s \cdot v_x \cdot \omega)}_{\text{滑移修正项}} = v_y - K_s \, v_x \, \omega
$$

滑移修正项的符号和形式具有明确的物理含义：当机器人向前行驶 ($v_x > 0$) 并左转 ($\omega > 0$) 时，离心力使车体向右侧滑，表现为负的附加侧向速度 $-K_s v_x \omega < 0$。

### 3.3 滑移修正运动学与前向仿真

将有效侧向速度代入运动学方程，得到引入滑移的三维运动模型

$$
\dot{x} = v_x \cos\psi - v_{y,eff} \sin\psi, \quad
\dot{y} = v_x \sin\psi + v_{y,eff} \cos\psi, \quad
\dot{\psi} = \omega
$$

采用前向欧拉法离散化，时间步长为 $\Delta t$：

$$
\begin{aligned}
x_{t+1} &= x_t + (v_x \cos\psi_t - v_{y,eff} \sin\psi_t)\,\Delta t \\
y_{t+1} &= y_t + (v_x \sin\psi_t + v_{y,eff} \cos\psi_t)\,\Delta t \\
\psi_{t+1} &= \psi_t + \omega\,\Delta t
\end{aligned}
$$

此模型在 MPPI 的前向仿真（轨迹展开）中充当状态转移函数。给定初始状态 $\mathbf{x}_0$ 和第 $k$ 条采样控制序列 $\{\mathbf{u}_t^{(k)}\}_{t=0}^{T-1}$，逐步前向仿真生成对应的状态轨迹：

$$
\mathbf{x}_{t+1}^{(k)} = f\!\left(\mathbf{x}_t^{(k)},\; \mathbf{u}_t^{(k)},\; K_s\right), \quad t = 0, 1, \ldots, T-1
$$

滑移因子 $K_s$ 由在线估计器自适应学习（见 §7），在每个控制周期的所有采样轨迹中共享同一估计值。前向仿真是 MPPI 计算的主要瓶颈。由于 $K$ 条轨迹之间相互独立，系统采用 OpenMP 多线程并行加速，将 $K$ 条轨迹分配到多个 CPU 核心同时计算。

---

## 4 空间索引迭代学习控制

### 4.1 空间索引 vs. 时间索引

经典 ILC 以时间步 $t$ 为索引。然而，移动机器人每次通过同一空间位置的时刻 $t$ 因速度不同而变化，导致时间索引下同一物理位置的经验无法对齐。相比之下，地面摩擦特性、坡度等物理量与**空间位置**强相关，因此本系统采用基于路径点索引 $i$（等价于弧长 $s_i$）的空间 ILC。

### 4.2 ILC 记忆库

ILC 记忆库 $\mathcal{M}$ 是一个与参考路径等长的查找表 (Lookup Table)，每个条目存储一个三维速度偏置向量：

$$
\mathcal{M} = \left\{\mathbf{b}_i = \begin{bmatrix} \delta v_{x,i} & \delta v_{y,i} & \delta\omega_i \end{bmatrix}^\top\right\}_{i=0}^{N-1}
$$

初始化时所有偏置为零向量。当系统收到新的参考路径且路径几何发生实质性变化时，可根据策略选择清空记忆库或保留已有学习成果。

### 4.3 路径点索引的确定与闭合路径处理

在每个控制周期中，系统须确定机器人当前所处的路径点索引 $i^*$。这一问题看似简单，但在自交叉路径（如"8"字形赛道）和闭合路径的起/终点附近，朴素的全局最近邻搜索会导致索引在不同路径段之间频繁跳跃。为此，本系统采用**加窗搜索-全局回退**两级策略。

**加窗搜索**。基于上一周期的最近点索引 $i_{prev}$，在窗口 $[i_{prev} - W_b,\; i_{prev} + W_f]$ 内搜索加权最近点：

$$
i^*_{win} = \arg\min_{i \in \mathcal{W}} \left[\; \|\mathbf{p}_i - \mathbf{p}_{robot}\|_2 \;+\; \lambda_h \cdot |\Delta\psi_i|\;\right]
$$

其中 $\mathcal{W}$ 为窗口内的索引集合（闭合路径时含环绕部分），$\lambda_h$ 为航向权重（默认 $0.4\;\text{m/rad}$），$\Delta\psi_i = \mathrm{wrap}_{[-\pi,\pi]}(\psi_{robot} - \psi_i)$ 为航向偏差。窗口大小取 $W_b = 25$，$W_f = 60$。引入航向项可有效区分自交叉点处两个几何距离相近但行驶方向相反的候选段。

**全局回退**。设全局最近点为 $i^*_{global}$，对应距离 $d_{global}$。窗口搜索结果 $i^*_{win}$ 须通过两项检验：

$$
|\Delta\psi_{i^*_{win}}| \leq \theta_{gate}, \qquad d_{win} \leq \eta \cdot d_{global}
$$

默认 $\theta_{gate} = 1.2\;\text{rad}$，$\eta = 1.6$。若任一条件不满足，则回退至 $i^* = i^*_{global}$。这保证了机器人被外力推离路径后仍能恢复到正确的路径段。

**闭合路径扩展**。当路径首尾点距离小于阈值 $d_{close}$（默认 $0.6\;\text{m}$）时，判定为闭合路径：

$$
\|\mathbf{p}_0 - \mathbf{p}_{N-1}\| < d_{close} \;\Longrightarrow\; \text{闭合}
$$

闭合路径需要以下额外机制：(1) **索引环绕**——在加窗搜索中，若窗口超出 $[0, N-1]$ 的边界，则将超出部分从路径的另一端继续搜索；先验索引构建中弧长预测值取模 $\hat{s}_t \leftarrow \hat{s}_t \bmod L$。(2) **移动目标点**——MPPI 的目标点不固定在路径末尾（否则机器人在起终重合区域会触发"到达目标"而停车），而是沿路径前方动态推进：

$$
s_{goal} = s_{i^*} + \max\!\left(v \cdot T \cdot \Delta t,\; d_{min}\right), \qquad s_{goal} \leftarrow s_{goal} \bmod L
$$

(3) **圈次检测**——通过监测弧长坐标的跳变实现：当 $s_{current} + \delta_m < s_{previous}$（默认 $\delta_m = 0.5\;\text{m}$）时，判定完成一圈。每圈结束后系统记录横向 RMSE（区分直道与弯道）、航向 RMSE、ILC 偏置 RMS、饱和次数以及更新增量 RMS 等指标，并输出至日志与 CSV 文件，用于离线分析 ILC 的圈间收敛行为。

### 4.4 跟踪误差的定义

确定最近路径点索引 $i^*$ 后，系统计算以下跟踪误差量。

**横向误差** $e_{lat}$ 定义为机器人在路径法线方向上的投影距离。设 $\Delta\mathbf{p} = \mathbf{p}_{robot} - \mathbf{p}_{i^*}$，路径在 $i^*$ 处的切线方向角为 $\psi_p$，则

$$
e_{lat} = -\Delta x \sin\psi_p + \Delta y \cos\psi_p
$$

其中 $\Delta x = x_{robot} - x_{i^*}$，$\Delta y = y_{robot} - y_{i^*}$。正值表示机器人位于路径左侧。

**航向误差** $e_{head}$ 定义为机器人航向与路径切线方向之差：

$$
e_{head} = \mathrm{wrap}_{[-\pi,\pi]}\!\left(\psi_{robot} - \psi_p\right)
$$

**路径曲率** $\kappa$ 通过以 $i^*$ 为中心、前后各偏移 $\delta$ 个索引的三点 Menger 曲率公式估计。取 $\mathbf{p}_a = \mathbf{p}_{i^* - \delta}$，$\mathbf{p}_b = \mathbf{p}_{i^*}$，$\mathbf{p}_c = \mathbf{p}_{i^* + \delta}$，则

$$
\kappa = \frac{2\,\mathcal{A}(\mathbf{p}_a, \mathbf{p}_b, \mathbf{p}_c)}
{\|\mathbf{p}_a - \mathbf{p}_b\| \cdot \|\mathbf{p}_b - \mathbf{p}_c\| \cdot \|\mathbf{p}_c - \mathbf{p}_a\|}
$$

其中 $\mathcal{A}$ 为三点构成的三角形有向面积（由叉积计算），曲率符号取叉积方向（正值为左转）。默认取 $\delta = 3$。曲率值在后续 ILC 更新律中起门控作用。

**路径末端衰减**。对于非闭合路径，在最后若干个点附近引入线性衰减因子 $f_{end} = (N - i^*) / N_{fade}$，对所有误差量做缩放以避免末端产生不必要的大幅修正。

### 4.5 ILC 更新律

本系统采用带遗忘因子的 P 型 ILC 更新律。在每个控制周期，根据当前最近路径点索引 $i^*$ 及对应的跟踪误差，**就地**更新记忆库条目 $\mathbf{b}_{i^*}$：

$$
\mathbf{b}_{i^*}^{(k+1)} = \gamma \cdot \mathbf{b}_{i^*}^{(k)} + \mathbf{K}_l \cdot \mathbf{e}^{(k)}(i^*)
$$

展开至各分量：

$$
\delta v_x(i^*) \;\leftarrow\; \gamma \cdot \delta v_x(i^*)
$$

$$
\delta v_y(i^*) \;\leftarrow\; \gamma \cdot \delta v_y(i^*) \;+\; \alpha_{lat} \cdot e_{lat}(i^*)
$$

$$
\delta\omega(i^*) \;\leftarrow\; \gamma \cdot \delta\omega(i^*) \;+\; \alpha_{head} \cdot e_{head}(i^*)
$$

其中 $\gamma \in (0,1)$ 为遗忘因子 (Forgetting Factor)，$\alpha_{lat}$ 和 $\alpha_{head}$ 分别为横向和航向学习增益。

需要指出两个设计要点。第一，前向速度偏置 $\delta v_x$ **仅衰减而不从误差中学习**，因为前向速度主要由 MPPI 的参考速度控制，ILC 聚焦于侧向滑移和转向修正。第二，与传统 ILC 在每次迭代结束后批量更新不同，本系统在每个控制周期内实时就地更新，同一圈内后半段的学习成果可立即影响当前圈的控制。

遗忘因子 $\gamma$ 控制历史偏置的衰减速率。考虑路径点 $i$ 处持续存在恒定扰动导致的稳态误差 $e_0$，令稳态偏置 $b_\infty = b^{(k+1)} = b^{(k)}$，可解得

$$
b_\infty = \frac{\alpha \cdot e_0}{1 - \gamma}
$$

偏置从任意初始值衰减到 $1/e$ 所需的更新次数为 $n_{1/e} = -1/\ln\gamma$。以默认值 $\gamma = 0.995$ 为例，$n_{1/e} \approx 200$ 次；在 $50\;\text{Hz}$ 控制频率下对应约 $4\;\text{s}$ 的记忆半衰期。$\gamma$ 的选取体现了**稳态精度**与**环境适应性**之间的折中：当 $\gamma \to 1$ 时遗忘极慢、稳态误差趋于零，但对环境变化的响应也极慢；当 $\gamma$ 较小时遗忘快、对新扰动适应迅速，但会引入与 $(1-\gamma)/(\alpha G)$ 成比例的残余稳态误差。

### 4.6 门控与限幅机制

原始更新律在所有路径点上均匀应用。为提高鲁棒性，系统引入以下门控与限幅策略。

**误差死区**。当 $|e_{lat}|$ 或 $|e_{head}|$ 小于死区阈值 $\epsilon_{db}$（默认 $0.005$）时，对应误差置零，不进行学习。这避免了在跟踪精度已满足要求时继续积累微小偏置。

**曲率衰减**。高曲率区域的跟踪误差中包含大量瞬态动态效应（如离心力引起的侧滑），不宜被完全学习。当 $|\kappa| > \kappa_{th}$ 时（默认 $\kappa_{th} = 0.10\;\text{m}^{-1}$），误差输入按系数 $\beta_{curv}$ 衰减（默认 $\beta_{curv} = 0.3$）：

$$
|\kappa(i^*)| > \kappa_{th} \;\Longrightarrow\; e_{lat} \leftarrow \beta_{curv} \cdot e_{lat}, \;\; e_{head} \leftarrow \beta_{curv} \cdot e_{head}
$$

**单步更新限幅**。每次更新的误差输入被限制在 $[-\Delta_{max}, \Delta_{max}]$ 内（横向 $0.02\;\text{m}$、航向 $0.02\;\text{rad}$），防止异常大误差导致偏置瞬间跳变。

**低速防护**。当机器人速度 $v < v_{th}$ 时，误差按 $v/v_{th}$ 线性缩放，避免起停阶段的瞬态偏差被错误学习。

**偏置饱和**。所有偏置值被硬限制在物理可行范围内：

$$
|\delta v_x|,\; |\delta v_y| \leq v_{bias,max}, \qquad |\delta\omega| \leq \omega_{bias,max}
$$

默认 $v_{bias,max} = 0.6\;\text{m/s}$，$\omega_{bias,max} = 0.8\;\text{rad/s}$。饱和机制既保证安全性，也为系统提供了监控手段：若饱和频繁触发，说明存在超出 ILC 补偿能力的系统性扰动或学习增益过大。

---

## 5 ILC 先验注入 MPPI 的采样策略

上一节建立了空间索引 ILC 的学习框架。本节是全章的核心：阐述 ILC 学到的空间偏置以何种方式注入 MPPI 优化器，为什么选择这种方式，以及它在采样效率上的理论意义。

### 5.1 融合策略的设计选择

将 ILC 偏置与 MPPI 控制器结合，存在两种直觉上均成立的设计路径。

**路径 A：输出端直接叠加。** ILC 偏置在 MPPI 求解完成后、发送至执行层之前叠加：

$$
\mathbf{u}_{output} = \mathbf{u}_{MPPI}^{*} + \mathbf{b}_{ILC}(s)
$$

此方案是经典 ILC 文献中的标准做法，实现简单。但在 MPPI 框架下，它引入了一个结构性缺陷：MPPI 在优化过程中对 ILC 偏置的存在**一无所知**。MPPI 求解的 $\mathbf{u}^*_{MPPI}$ 是在"不含任何偏置"的前提下的最优解，但实际执行的 $\mathbf{u}_{output}$ 叠加了偏置——后者可能将控制量推出速度可行域、引发碰撞或加剧滑移，而 MPPI 的代价函数对此没有任何评估机会。本质上，该方案将 ILC 置于 MPPI 安全保障体系的**外部**。

**路径 B：采样分布先验注入。** ILC 偏置在 MPPI 采样**之前**注入，作为采样分布的均值偏移：

$$
\mathbf{u}_t^{(k)} = \underbrace{(\mathbf{u}_t^{opt} + \mathbf{b}_{i_t})}_{\text{偏移后的采样中心}} + \boldsymbol{\epsilon}_t^{(k)}
$$

此设计使 ILC 偏置进入 MPPI 的代价评估循环**内部**。所有 $K$ 条采样轨迹均在含偏置的控制量下进行前向仿真和代价计算，MPPI 对偏置效果拥有完整的感知能力。若 ILC 偏置方向正确，含偏置的样本代价较低，在加权平均中获得较大权重；若偏置方向错误（例如环境已变化），含偏置的样本代价上升，MPPI 通过加权机制将其自然抑制。

本系统采用**路径 B**。这一选择将 ILC 从"后置前馈"升级为 MPPI 优化循环的**内部参与者**，使 ILC 的学习成果始终受到代价函数的持续审核。表 1 总结了两种路径的关键差异。

| 特性 | 路径 A：直接叠加 | 路径 B：先验注入 |
|:---:|:---|:---|
| 偏置是否参与代价评估 | 否 | 是 |
| 安全性保证 | 需额外的输出限幅 | 由 MPPI 代价函数保障 |
| 对 MPPI 最优性的影响 | 破坏（输出非 MPPI 最优解） | 保留（MPPI 在含偏置中心附近重新优化） |
| 对采样效率的影响 | 无 | 提升（见 §5.5） |

### 5.2 MPPI 采样框架中的先验位置

标准 MPPI 在每个控制周期维护一个最优控制序列 $\mathbf{U}^{opt}$，并以其作为下一周期的采样中心。这一序列可视为 MPPI 对"当前应执行何种控制"的**最佳猜测**——它源自上一周期的加权平均更新，携带了关于目标方向、避障约束和速度规划的综合信息。

然而，$\mathbf{U}^{opt}$ 仅反映了上一个控制周期末的在线优化结果。它不包含对**当前空间位置处特定扰动**的显式记忆——即使机器人在上一圈的相同弯道处观测到了相似的侧向偏移，这一经验不会被 $\mathbf{U}^{opt}$ 自动携带到下一圈。MPPI 的"记忆"仅限于上一控制周期到当前周期之间的序列平移，而非跨圈次的空间经验。

外部控制先验 $\mathbf{U}_{prior}$ 填补的正是这一空白。先验注入后，采样中心变为

$$
\bar{\mathbf{u}}_t = \mathbf{u}_t^{opt} + \mathbf{b}_{i_t}, \quad t = 0, 1, \ldots, T-1
$$

其中 $\mathbf{b}_{i_t}$ 为路径索引 $i_t$ 处的 ILC 偏置。至此，MPPI 的采样中心同时包含了两类信息：$\mathbf{U}^{opt}$ 提供的在线优化记忆，与 $\mathbf{U}_{prior}$ 提供的跨圈次空间记忆。

### 5.3 预测时域内的索引映射

MPPI 在预测时域 $T$ 步内逐步前向仿真，每一步 $t$ 都需要对应的 ILC 偏置。因此需要将时间步 $t$ 映射到路径上的空间索引 $i_t$。

本系统默认使用**弧长-速度预测法**：基于当前速度 $v = \sqrt{v_x^2 + v_y^2}$（若 $v < v_{min}$ 则取参考速度 $v_{ref}$），预测未来第 $t$ 步对应的弧长

$$
\hat{s}_t = s_{i^*} + v \cdot \Delta t \cdot t, \quad t = 0, 1, \ldots, T-1
$$

然后通过在弧长数组 $\{s_i\}$ 上的二分查找，将 $\hat{s}_t$ 映射为路径点索引 $i_t$。对于闭合路径，在查找前对弧长取模 $\hat{s}_t \leftarrow \hat{s}_t \bmod L$。

给定索引序列 $\mathcal{I} = \{i_0, i_1, \ldots, i_{T-1}\}$，从记忆库中提取先验控制序列

$$
\mathbf{U}_{prior} = \left[\mathbf{b}_{i_0},\; \mathbf{b}_{i_1},\; \ldots,\; \mathbf{b}_{i_{T-1}}\right]
$$

### 5.4 采样分组策略

MPPI 的 $K$ 条采样轨迹分为两组：exploitation 样本（占比 $1 - r$，默认 $90\%$）与 exploration 样本（占比 $r$，默认 $10\%$）。ILC 先验对两组的注入方式不同，体现了经验利用与空间探索之间的权衡。记 $\mathbf{u}_{base,t}^{(k)}$ 为采样基准中心，则

$$
\mathbf{u}_{base,t}^{(k)} = \begin{cases}
\mathbf{u}_t^{opt} + \mathbf{b}_{i_t}, & k < K(1-r) \quad (\text{exploitation}) \\[4pt]
\mathbf{b}_{i_t}, & k \geq K(1-r) \quad (\text{exploration})
\end{cases}
$$

$$
\mathbf{u}_t^{(k)} = \mathrm{clamp}\!\left(\mathbf{u}_{base,t}^{(k)} + \boldsymbol{\epsilon}_t^{(k)}\right)
$$

**Exploitation 样本**以"上周期最优解 + ILC 偏置"为中心做小范围高斯扰动，是 MPPI 性能的主要贡献者。ILC 先验的作用是将采样中心从"上周期对当前位置的局部最优响应"偏移到"上周期最优解 + 空间经验修正"，使高质量样本的密度增大。

**Exploration 样本**以 ILC 偏置本身为中心。在标准 MPPI 中，exploration 样本以控制空间原点为中心（即纯噪声），大量算力消耗于物理上不合理的区域。引入 ILC 先验后，exploration 样本被引导到"经验合理区域"附近展开，显著降低了无效采样的比例。

此外，系统支持可选的先验正则化代价。当正则化权重 $w_{prior} > 0$ 时，在代价函数中增加二次惩罚项

$$
J_{prior} = w_{prior} \sum_{t=0}^{T-1} \|\mathbf{u}_t^{(k)} - \mathbf{u}_{base,t}^{(k)}\|^2
$$

鼓励采样控制量靠近先验偏移后的均值。默认 $w_{prior} = 0$，即 ILC 仅通过 mean shift 影响采样分布，不修改代价函数结构。

### 5.5 先验注入的信息论解释与采样效率

MPPI 的本质是对最优控制分布的蒙特卡洛近似。最优控制分布由路径积分理论给出：

$$
p^*(\mathbf{U}) \propto \exp\!\left(-\frac{J(\mathbf{U})}{\lambda}\right)
$$

MPPI 使用提议分布 $q(\mathbf{U}) = \mathcal{N}(\bar{\mathbf{U}},\; \boldsymbol{\Sigma} \otimes \mathbf{I}_T)$ 对 $p^*$ 进行重要性采样。近似质量取决于 $q$ 与 $p^*$ 的**重叠程度**——重要性采样的方差与 $q$ 和 $p^*$ 之间的 KL 散度正相关。当提议分布的均值 $\bar{\mathbf{U}}$ 远离 $p^*$ 的众数时，大量样本落入 $p^*$ 的低密度区域，有效采样数急剧下降。

在标准 MPPI 中，$\bar{\mathbf{U}} = \mathbf{U}^{opt}$（上周期最优解）。由于 $\mathbf{U}^{opt}$ 仅包含上一控制周期的在线优化结果，它不携带对特定空间位置处重复性扰动的记忆。当机器人一圈又一圈经过赛道的同一弯道时，MPPI 在每圈的该弯道处都需要从头"重新发现"最优修正量。

注入 ILC 先验后，$\bar{\mathbf{U}} = \mathbf{U}^{opt} + \mathbf{U}_{prior}$。若 ILC 在先前的圈次中已学到弯道处的正确修正，$\bar{\mathbf{U}}$ 被偏移到更接近 $p^*$ 众数的位置，$q$ 的均值更接近 $p^*$ 的众数，KL 散度减小，重要性采样方差随之降低。用有效采样尺寸 (Effective Sample Size, ESS) 量化这一改善：

$$
\mathrm{ESS} = \frac{1}{\sum_{k=1}^{K} (w^{(k)})^2}
$$

其中 $w^{(k)}$ 为归一化权重。ESS 越大，表示权重分布越均匀，越多的样本对最优估计有有效贡献。ILC 先验通过缩小 $q$ 的均值与 $p^*$ 众数之间的距离，使权重分布更均匀，ESS 升高。其实际意义是：**在固定采样数 $K$ 下达到更高的控制品质，或以更少的采样数达到同等品质**。后者在嵌入式部署场景中尤为关键——ILC 的空间记忆可作为计算资源受限时的性能补偿手段。

这也解释了为什么 ILC 先验的注入位置必须在采样阶段而非输出阶段：只有在采样前修改 $q$ 的参数，才能影响所有 $K$ 条样本的生成过程，从而在信息论意义上提升 MPPI 的求解质量。若在输出端叠加（不改变 $q$），$K$ 条样本的采样和评估不受影响，ILC 贡献完全脱离了 MPPI 的优化框架。

### 5.6 残差优化与自适应信用分配

注入 ILC 先验后，MPPI 的优化目标发生了本质变化。在标准 MPPI 中，优化变量是绝对控制序列 $\mathbf{U}$，目标为

$$
\mathbf{U}^* = \arg\min_{\mathbf{U}} \; \mathbb{E}_{\boldsymbol{\epsilon}}\left[J(\mathbf{U} + \boldsymbol{\epsilon})\right]
$$

注入先验后，等价于引入变量替换 $\mathbf{U} = \mathbf{U}_{prior} + \Delta\mathbf{U}$，其中 $\mathbf{U}_{prior}$ 来自 ILC 记忆库、在本周期内是常量。MPPI 实际求解的变为

$$
\Delta\mathbf{U}^* = \arg\min_{\Delta\mathbf{U}} \; \mathbb{E}_{\boldsymbol{\epsilon}}\left[J(\mathbf{U}_{prior} + \Delta\mathbf{U} + \boldsymbol{\epsilon})\right]
$$

即 MPPI 优化的是**相对于 ILC 先验的残差修正量** $\Delta\mathbf{U}$。这构成了一种**残差学习架构**：ILC 提供空间经验驱动的粗粒度前馈（基于历史圈次的跟踪误差积累），MPPI 在此基础上执行精细的在线残差优化（基于当前状态与代价函数）。

当 ILC 先验准确时，MPPI 的最优残差 $\Delta\mathbf{U}^*$ 会相应减小。从外部观察，总控制输出

$$
\mathbf{u}_{total} = \underbrace{\Delta\mathbf{u}^*}_{\text{MPPI 残差}} + \underbrace{\mathbf{b}_{ILC}}_{\text{ILC 先验}}
$$

可能与不使用 ILC 时的纯 MPPI 输出 $\mathbf{u}^*_{MPPI}$ 差异不大——ILC 的贡献被 MPPI 残差的等量减小所"吸收"。这一现象源于残差架构的自平衡特性，而非 ILC 的无效。考虑两种极端情况：

1. **ILC 先验完全准确（$\mathbf{b}_{ILC} \to \boldsymbol{\delta}_{true}$）**。最优残差趋近零，$\Delta\mathbf{U}^* \to \mathbf{0}$。MPPI 的 $K$ 条样本集中分布在低代价区域附近，权重分布高度均匀（ESS 接近 $K$）。虽然总输出与纯 MPPI 可能类似，但 MPPI 到达该解所需的"搜索工作量"大幅减少。

2. **ILC 先验完全错误（$\mathbf{b}_{ILC}$ 与实际扰动方向相反）**。含偏置的样本在代价评估中获得高代价，权重集中于纯噪声中偶然落入正确区域的少数样本上。MPPI 通过大幅的 $\Delta\mathbf{U}^*$ 自动抵消错误偏置，最终输出仍然合理，但 ESS 显著降低，控制品质可能不如无 ILC 的 baseline。

因此，ILC 的价值**不应仅通过比较最终跟踪误差来评估**（在采样数充足时 MPPI 可独立达到高精度），更有意义的评估维度包括：(1) 在采样数受限时 ILC 带来的性能提升；(2) ESS 随迭代圈数的增长趋势；(3) 达到目标精度所需的圈数（收敛速度）。

### 5.7 输出阶段的整合

MPPI 通过对所有采样轨迹按代价加权平均，得到最优残差控制序列 $\Delta\mathbf{U}^*$。最终输出需叠加回 ILC 先验：

$$
\mathbf{u}_{output} = \mathcal{F}\!\left(\Delta\mathbf{U}^* + \mathbf{U}_{prior}\right) + \mathbf{u}_{slip}
$$

其中 $\mathcal{F}(\cdot)$ 为可选的 Savitzky-Golay 平滑滤波，$\mathbf{u}_{slip}$ 为在线滑移补偿项（见 §10）。叠加步骤保证最终执行的控制指令同时包含 MPPI 的在线残差优化与 ILC 的空间经验前馈。

需要指出，这里的"叠加"与§5.1 中批判的"路径 A"并不矛盾。区别在于：路径 A 中 MPPI 对偏置毫不知情，而路径 B 中 MPPI 已在含偏置的采样中心附近完成了优化——偏置的效果已被代价函数充分评估过，此处的叠加只是将 MPPI 内部的"残差视角"还原为外部的"绝对控制视角"。

---

## 6 代价函数

代价函数是 MPPI 采样评估的核心，其设计直接决定了控制器的跟踪性能与行驶品质。本系统的代价函数由**阶段代价** $c_t$ 和**终端代价** $c_T$ 两部分构成，第 $k$ 条采样轨迹的总代价为

$$
J^{(k)} = \sum_{t=0}^{T-1} c_t\!\left(\mathbf{x}_t^{(k)},\; \mathbf{u}_t^{(k)},\; \mathbf{u}_{t-1}^{(k)}\right) + c_T\!\left(\mathbf{x}_T^{(k)},\; \mathbf{x}_{goal}\right)
$$

阶段代价按功能可分为四类：路径跟踪代价、滑移感知代价、控制平滑代价和信息论代价。下面逐一给出形式化定义。

### 6.1 路径跟踪代价

路径跟踪是控制器的首要目标。系统通过预计算的栅格地图 (Grid Map) 将连续的参考路径离散化为两张图层：距离误差图 $d(\mathbf{x})$ 记录任意位置到最近路径点的欧氏距离，参考航向图 $\psi_{ref}(\mathbf{x})$ 记录最近路径点处的切线方向角。基于这两张图层，路径跟踪代价由三个子项构成。

**位置跟踪**。以距离误差的平方作为主跟踪目标：

$$
c_{pos} = w_d \cdot d(\mathbf{x})^2
$$

其中 $w_d$ 为位置误差权重。该项在所有代价中权重最大，驱动采样轨迹贴近参考路径。

**航向跟踪**。航向偏差 $\Delta\psi = \mathrm{wrap}_{[-\pi,\pi]}(\psi - \psi_{ref})$ 采用 Huber 型代价函数，在小误差区域保持二次灵敏度，在大误差区域切换为线性增长以避免极端代价值导致数值问题：

$$
c_{head} = w_\psi \cdot \ell_H(\Delta\psi), \qquad
\ell_H(\Delta\psi) = \begin{cases}
\Delta\psi^2, & |\Delta\psi| < \delta_H \\
|\Delta\psi| - \frac{\delta_H}{2}, & |\Delta\psi| \geq \delta_H
\end{cases}
$$

其中 $\delta_H = 0.5\;\text{rad}$ 为 Huber 阈值。

**速度跟踪**。将机器人沿参考路径方向的投影速度与参考速度 $v_{ref}$ 做差，以二次代价鼓励匀速行驶：

$$
v_{aligned} = v_x \cos\Delta\psi + v_y \sin\Delta\psi, \qquad c_{vel} = w_v \cdot (v_{aligned} - v_{ref})^2
$$

路径跟踪代价的合项为 $c_{track} = c_{pos} + c_{head} + c_{vel}$。

### 6.2 滑移感知代价

为抑制高速转弯时的侧向滑移风险，系统引入两类与滑移因子 $K_s$ 耦合的代价项。

**滑移风险惩罚**。对预测滑移量的二次惩罚，直接抑制高风险的速度-角速度组合输入：

$$
c_{slip} = w_{slip} \cdot (K_s \cdot v_x \cdot \omega)^2
$$

**曲率自适应限速**。根据路径局部曲率 $\kappa$ 和有效摩擦系数 $\mu_{eff}$ 计算安全过弯速度上限，对超速采样施加二次惩罚。曲率通过参考航向图的前向差分估计：$\kappa \approx |\Delta\psi_{ahead}|/l_a$。有效摩擦系数考虑了滑移退化效应：$\mu_{eff} = \mu_0 \cdot \max(0.3,\; 1 - 2K_s)$。对曲率做软饱和处理后，安全过弯速度与代价为

$$
v_{safe} = \sqrt{\frac{\mu_{eff} \cdot g}{\kappa_{eff}}}, \qquad c_{curv} = w_{curv} \cdot \max\!\left(0,\; v - v_{safe}(1 + m_v)\right)^2
$$

**弯道偏航率跟踪**。仅在弯道（$\kappa > \kappa_{th}$）中激活，引导 MPPI 生成与路径曲率匹配的转向动作：

$$
c_{yaw} = w_{yaw} \cdot \min\!\left(1,\; \frac{\kappa - \kappa_{th}}{0.5}\right) \cdot (\omega - \omega_{des})^2
$$

滑移感知代价的合项为 $c_{slip\_aware} = c_{slip} + c_{curv} + c_{yaw}$。

### 6.3 控制平滑代价

为抑制控制指令在相邻时间步之间的剧烈跳变，系统在两个层次上施加平滑惩罚。

**体速度层平滑**。对三维体速度指令的变化量施加二次惩罚：

$$
c_{smooth,body} = \sum_{j \in \{v_x, v_y, \omega\}} w_{j}^{cmd} \cdot (\Delta u_j)^2, \qquad \Delta\mathbf{u}_t = \mathbf{u}_t - \mathbf{u}_{t-1}
$$

**车轮指令层平滑**。通过逆运动学将体速度映射为八维车轮指令（四个转向角 + 四个轮速），对八维变化量施加独立的二次惩罚：

$$
c_{smooth,wheel} = \sum_{j=1}^{8} w_{j}^{wheel} \cdot (\Delta q_j)^2, \qquad \Delta\mathbf{q}_t = h(\mathbf{u}_t) - h(\mathbf{u}_{t-1})
$$

其中 $h(\cdot)$ 为三维到八维的逆运动学映射（详见 §9）。双层平滑设计的动机是：体速度层平滑保证指令的总体连续性，车轮指令层平滑则直接约束实际执行机构的变化率。

### 6.4 控制代价与终端代价

路径积分控制理论要求代价函数包含二次控制代价，惩罚控制输入的幅值。在采样实现中，其有效形式为

$$
c_{ctrl} = \lambda(1-\alpha)\;\bar{\mathbf{u}}^\top\,\boldsymbol{\Sigma}^{-1}\,\mathbf{u}
$$

其中 $\bar{\mathbf{u}}$ 为当前采样中心，$\alpha \in [0,1]$ 为插值系数，默认取 $\alpha = 0.975$。

终端代价评估预测时域末端状态与目标点的距离偏差，仅在距目标较远时激活：

$$
c_T = \begin{cases}
w_T \cdot \|\mathbf{p}_T - \mathbf{p}_{goal}\|^2, & \|\mathbf{p}_T - \mathbf{p}_{goal}\| > 0.5\;\text{m} \\
0, & \text{otherwise}
\end{cases}
$$

### 6.5 总代价汇总

综合以上各项，第 $k$ 条采样轨迹在第 $t$ 步的阶段代价为

$$
c_t = \underbrace{c_{pos} + c_{head} + c_{vel}}_{\text{路径跟踪}} + \underbrace{c_{slip} + c_{curv} + c_{yaw}}_{\text{滑移感知}} + \underbrace{c_{smooth,body} + c_{smooth,wheel}}_{\text{控制平滑}} + \underbrace{c_{ctrl}}_{\text{控制代价}}
$$

各代价项的权重及其默认值汇总于表 2。

| 符号 | 代价项 | 默认值 | 说明 |
|:---:|:---|:---:|:---|
| $w_d$ | 位置跟踪 | 40.0 | 距离误差的平方权重 |
| $w_\psi$ | 航向跟踪 | 30.0 | Huber 型航向偏差权重 |
| $w_v$ | 速度跟踪 | 10.0 | 沿路径方向的速度误差权重 |
| $w_T$ | 终端代价 | 10.0 | 预测末端到目标点的距离惩罚 |
| $w_{slip}$ | 滑移风险 | 15.0 | $K_s v_x \omega$ 的二次惩罚 |
| $w_{curv}$ | 曲率限速 | 60.0 | 超过安全过弯速度时的惩罚 |
| $w_{yaw}$ | 偏航率跟踪 | 25.0 | 弯道中偏航率偏差的惩罚 |
| $\mathbf{w}^{wheel}$ | 车轮平滑 | $(1.4)^4,(0.1)^4$ | 四转向角 + 四轮速变化量惩罚 |

---

## 7 在线滑移因子估计

滑移因子 $K_s$ 并非预先标定的常数，而是由在线自适应估计器通过梯度下降法实时学习。估计器基于以下预测模型：

$$
\hat{v}_y = v_y^{cmd} - K_s \cdot v_x \cdot \omega
$$

定义预测误差为

$$
e_s = v_{y,actual} - \hat{v}_y = v_{y,actual} - v_y^{cmd} + K_s \cdot v_x \cdot \omega
$$

以最小化 $J_s = \frac{1}{2}e_s^2$ 为目标，对 $K_s$ 求梯度并更新：

$$
\frac{\partial J_s}{\partial K_s} = e_s \cdot v_x \cdot \omega, \qquad K_s \leftarrow K_s - \eta \cdot \frac{\partial J_s}{\partial K_s}
$$

其中 $\eta$ 为学习率（默认 $0.01$）。为保证估计的可靠性，仅当激励信号 $|v_x \cdot \omega|$ 超过阈值 $\epsilon_{exc}$ 时才触发更新（持续性激励条件）。估计值被限制在 $[K_{s,min},\; K_{s,max}]$ 内。

估计器对实际速度信号施加一阶低通滤波以抑制测量噪声，并维护残差的滑动窗口统计量用于收敛性判断。估计得到的 $K_s$ 在当前周期内被所有 $K$ 条采样轨迹的前向仿真共享，同时也被前馈滑移补偿器使用。

---

## 8 信息论加权与最优序列更新

### 8.1 代价归一化与权重计算

在 $K$ 条轨迹的代价 $\{J^{(1)}, J^{(2)}, \ldots, J^{(K)}\}$ 计算完毕后，通过 softmax 变换将代价转化为归一化权重。首先找到最小代价 $J_{min} = \min_k J^{(k)}$，然后计算

$$
w^{(k)} = \frac{\exp\!\left(-\frac{1}{\lambda}(J^{(k)} - J_{min})\right)}{\sum_{j=1}^{K}\exp\!\left(-\frac{1}{\lambda}(J^{(j)} - J_{min})\right)}
$$

温度参数 $\lambda$ 控制权重分布的"锐度"：$\lambda \to 0$ 时退化为贪心选择（仅取最低代价轨迹），$\lambda \to \infty$ 时退化为均匀平均。默认 $\lambda = 100$。减去 $J_{min}$ 的操作在数学上不影响归一化结果，但在数值实现中至关重要——它防止了指数函数的溢出。

### 8.2 最优控制序列更新

最优控制序列通过噪声的加权平均进行更新：

$$
\mathbf{u}_t^{opt} \leftarrow \mathbf{u}_t^{opt} + \sum_{k=1}^{K} w^{(k)} \cdot \boldsymbol{\epsilon}_t^{(k)}, \quad t = 0, 1, \ldots, T-1
$$

注意这里加权的是**噪声**而非完整控制量。这一设计保证了更新的方向性。更新后对控制量做速度限幅。

### 8.3 控制输出与平滑滤波

更新后的最优序列与 ILC 先验偏置叠加，得到总控制序列：

$$
\mathbf{u}_t^{total} = \mathbf{u}_t^{opt} + \mathbf{b}_{i_t}, \quad t = 0, 1, \ldots, T-1
$$

取序列的第一个元素 $\mathbf{u}_0^{total}$ 作为当前周期的输出。在输出前，可选地通过 Savitzky-Golay 多项式滤波器对控制序列进行平滑。该滤波器在以当前时刻为中心的 $(2n+1)$ 窗口内拟合 $p$ 阶多项式，取中心点的拟合值作为平滑输出：

$$
\mathbf{u}_{smooth} = \sum_{i=-n}^{n} c_i \cdot \mathbf{u}_i
$$

其中 $\{c_i\}$ 为预计算的 SG 滤波系数。默认半窗口宽度 $n = 4$，多项式阶数 $p = 6$。

---

## 9 从三维体速度到八维轮指令的升维映射

### 9.1 四轮全向转向的运动学结构

四轮全向转向 (Swerve Drive) 平台的每个车轮拥有两个独立自由度：转向角 $\phi_i$ 和轮速 $\Omega_i$（$i \in \{fl, fr, rl, rr\}$）。因此，底层执行空间为八维向量

$$
\mathbf{q} = \begin{bmatrix} \phi_{fl} & \phi_{fr} & \phi_{rl} & \phi_{rr} & \Omega_{fl} & \Omega_{fr} & \Omega_{rl} & \Omega_{rr} \end{bmatrix}^\top \in \mathbb{R}^8
$$

而 MPPI 的控制输出仅为三维体速度 $(v_x, v_y, \omega)$。从三维到八维的映射由**逆运动学**完成。

### 9.2 各轮速度分量的推导

设车体质心为原点，四轮相对于质心的几何参数为：前轴距质心 $l_f$，后轴距质心 $l_r$，左轮距质心 $d_l$，右轮距质心 $d_r$。各轮在体坐标系中的速度分量为

$$
\begin{aligned}
v_{x,fl} &= v_x - \omega\, d_l, &\quad v_{y,fl} &= v_y + \omega\, l_f \\
v_{x,fr} &= v_x + \omega\, d_r, &\quad v_{y,fr} &= v_y + \omega\, l_f \\
v_{x,rl} &= v_x - \omega\, d_l, &\quad v_{y,rl} &= v_y - \omega\, l_r \\
v_{x,rr} &= v_x + \omega\, d_r, &\quad v_{y,rr} &= v_y - \omega\, l_r
\end{aligned}
$$

### 9.3 转向角与轮速的计算

各轮的转向角由双参数反正切函数给出：$\phi_i = \mathrm{atan2}(v_{y,i},\; v_{x,i})$。各轮的合速度幅值除以轮胎半径 $r$，得到轮轴角速度：$\Omega_i = \sqrt{v_{x,i}^2 + v_{y,i}^2}/r$。

### 9.4 完整映射的矩阵表示

从体速度到各轮速度分量的线性映射为

$$
\begin{bmatrix}
v_{x,fl} \\ v_{y,fl} \\ v_{x,fr} \\ v_{y,fr} \\ v_{x,rl} \\ v_{y,rl} \\ v_{x,rr} \\ v_{y,rr}
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 & -d_l \\
0 & 1 & +l_f \\
1 & 0 & +d_r \\
0 & 1 & +l_f \\
1 & 0 & -d_l \\
0 & 1 & -l_r \\
1 & 0 & +d_r \\
0 & 1 & -l_r
\end{bmatrix}
\begin{bmatrix}
v_x \\ v_y \\ \omega
\end{bmatrix}
$$

该映射是**线性**的，但后续从 $(v_{x,i}, v_{y,i})$ 到 $(\phi_i, \Omega_i)$ 的极坐标变换是非线性的。整体的三维到八维映射表示为 $\mathbf{q} = h(\mathbf{u}): \mathbb{R}^3 \to \mathbb{R}^8$。

### 9.5 升维映射的执行位置

在系统架构中，三维到八维的映射在两个位置实现。在 MPPI 核心控制器内部，该映射用于计算八维车辆指令的变化量代价。在实际执行层，该映射由 Gazebo 仿真环境中的 `vel_driver` 节点完成。这种分层设计使得 MPPI 始终在紧凑的三维空间中进行采样与优化——避免了在八维空间中采样导致的维度灾难——而将确定性的升维映射推迟到执行层，不增加优化的计算负担。

---

## 10 滑移补偿层

在 MPPI 求解出最优三维速度指令后、发送给执行层之前，系统通过滑移补偿层对指令进行修正，抵消预测的侧向滑移。

### 10.1 前馈补偿

基于当前估计的滑移因子 $K_s$，前馈补偿项为

$$
\Delta v_{y,ff} = -\gamma_{ff} \cdot K_s \cdot v_x \cdot \omega
$$

其中 $\gamma_{ff}$ 为前馈增益（默认 $0.7$）。增益 $\gamma_{ff} < 1$ 提供了一定的保守裕度，避免因 $K_s$ 估计不精确导致的过补偿。

### 10.2 闭环反馈补偿

前馈补偿依赖于模型的准确性。为消除模型残差导致的稳态跟踪误差，系统叠加闭环反馈修正：

$$
\Delta v_{y,fb} = -K_p^{lat} \cdot e_{lat} - K_i \cdot \int e_{lat}\, dt
$$

$$
\Delta\omega_{fb} = -K_p^{head} \cdot e_{head}
$$

其中 $K_p^{lat}$、$K_p^{head}$、$K_i$ 分别为比例和积分增益。积分项消除恒定偏置导致的稳态误差，并设有抗饱和限幅和接近目标时的衰减机制。

### 10.3 总输出

最终发送到执行层的三维速度指令为

$$
\mathbf{u}_{output} = \mathbf{u}_{MPPI} + \begin{bmatrix} 0 \\ \Delta v_{y,ff} + \Delta v_{y,fb} \\ \Delta\omega_{fb} \end{bmatrix}
$$

经速度限幅后，由 vel_driver 映射为八维轮指令并执行。

---

## 11 收敛性与稳定性分析

### 11.1 单点收敛条件

考虑路径点 $i$ 处的 ILC 学习动力学。在近似线性化条件下，假设从偏置到跟踪误差的简化传递关系为

$$
e^{(k)}(i) = e_0(i) - G \cdot b^{(k)}(i)
$$

其中 $e_0$ 为无偏置时的稳态误差，$G > 0$ 为系统增益。代入更新律可得

$$
b^{(k+1)} = (\gamma - \alpha G) \cdot b^{(k)} + \alpha \, e_0
$$

此一阶差分方程的收敛条件为特征根的绝对值小于 1：

$$
|\gamma - \alpha G| < 1 \qquad \Longleftrightarrow \qquad \alpha < \frac{1 + \gamma}{G}
$$

收敛后的稳态残余误差为

$$
e_\infty = e_0 \cdot \frac{1 - \gamma}{1 - \gamma + \alpha G}
$$

当 $\gamma \to 1$ 时 $e_\infty \to 0$，可实现渐近零稳态误差。当 $\gamma < 1$ 时残余误差与 $(1-\gamma)/(\alpha G)$ 成正比。这再次印证了遗忘因子在鲁棒性与精度之间的折中角色。

### 11.2 与 MPPI 的协同稳定性

§11.1 的分析假设系统增益 $G$ 为常数。当 MPPI 介入后，这一假设需要重新审视：MPPI 在每个控制周期基于代价函数重新优化，其残差 $\Delta\mathbf{U}^*$ 会主动响应 ILC 偏置的变化，使得 ILC 所"看到"的等效系统增益不再是固定值。

**MPPI 作为自适应增益调节器。** ILC 偏置首先进入 MPPI 的采样中心，MPPI 在含偏置的样本中重新求解最优残差 $\Delta u^*$，实际执行的控制量为 $u_{total} = \Delta u^* + b$。关键观察是 $\Delta u^*$ 本身是 $b$ 的函数。ILC 所感知的等效增益 $G_{eff}$ 满足

$$
G_{eff} = G_{plant} \cdot (1 - \rho)
$$

其中 $G_{plant}$ 为被控对象的原始增益，$\rho \in [0,1)$ 为 MPPI 的"吸收率"。这一特性对 ILC 的收敛有两面性：$G_{eff} < G_{plant}$ 使收敛条件 $\alpha < (1+\gamma)/G_{eff}$ 更容易满足——MPPI 的存在扩大了学习率的稳定范围，充当了 ILC 学习过程的**隐式稳定器**；但 $G_{eff}$ 的降低同时减缓了收敛速度。

**融合架构的三重安全保障：**

1. **代价函数审核。** ILC 偏置仅改变采样分布的均值，不修改代价函数。若偏置将采样中心推向碰撞区域或速度约束边界，相关样本在代价评估中获得极高代价，在 softmax 加权中被自然抑制。

2. **探索样本兜底。** 探索组（默认 $10\%$）以 ILC 偏置为中心、不依赖上周期最优解。即使 ILC 先验与 $\mathbf{U}^{opt}$ 都偏离了正确方向，探索样本的大范围采样仍有概率覆盖低代价区域。

3. **先验正则化的柔性。** 默认 $w_{prior} = 0$，ILC 先验为纯粹的软建议。即使设置 $w_{prior} > 0$，其效果也仅是一个二次软约束，可被更强的碰撞代价或跟踪代价推翻。

### 11.3 多点耦合效应的讨论

在实际系统中，路径点 $i$ 的偏置通过系统动力学影响相邻点的跟踪误差，形成点间耦合。然而以下因素使耦合效应得到有效抑制：(1) 每个控制周期 MPPI 均从当前状态重新优化，隐含地吸收了偏置引入的状态偏移；(2) 偏置饱和防止了局部偏置的过度积累与传播；(3) 遗忘因子的持续衰减确保无偏置的无限增长。因此，在实际运行中可将各路径点视为弱耦合的独立学习单元进行分析。

---

## 12 本章小结

本章完整阐述了基于迭代学习控制先验的模型预测路径积分控制器的理论框架与系统设计。核心贡献概括如下：

1. **空间索引 ILC**（§4）。以路径点索引（等价于累积弧长）为记忆库的键，解决了时间索引在变速运动中的对齐困难。通过遗忘因子、误差死区、曲率衰减、单步限幅、偏置饱和等多级机制，保证了学习过程在复杂运行条件下的安全性与稳定性。

2. **MPPI 先验注入**（§5）。将 ILC 偏置作为 MPPI 采样分布的均值偏移而非直接控制叠加，在保留避障安全性的前提下实现了经验驱动的采样效率提升，形成隐式残差学习架构。从信息论角度论证了先验注入通过缩小提议分布与最优控制分布之间的 KL 散度来提升有效采样尺寸。

3. **三维滑移运动模型**（§3）。在名义运动学基础上引入一阶耦合滑移项 $-K_s v_x \omega$，以标量参数 $K_s$ 紧凑地刻画侧向滑移的速度-角速度耦合效应。

4. **统一的多目标代价函数**（§6）。代价函数由路径跟踪、滑移感知、控制平滑和信息论项四类构成，各项权重独立可调。

5. **在线滑移估计与多层补偿**（§7, §10）。通过梯度下降法实时学习滑移因子 $K_s$，结合前馈-反馈双通道滑移补偿，实现高速工况下的厘米级跟踪精度。

6. **三维到八维升维映射**（§9）。通过逆运动学将体速度指令确定性地映射为四轮的转角与轮速，使 MPPI 始终在紧凑的三维空间中优化，避免维度灾难。

7. **闭合路径全支持**（§4.3）。弧长环绕、窗口搜索环绕、移动目标点、圈次自动检测与多维度指标统计，构成了适用于循环跟踪任务的完整工程方案。

8. **收敛性保证**（§11）。给出了单点收敛条件的解析推导和稳态残余误差的闭式表达，并论证了 ILC-MPPI 融合架构的协同稳定性。
