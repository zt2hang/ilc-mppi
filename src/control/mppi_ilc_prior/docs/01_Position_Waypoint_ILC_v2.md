# 基于位置路径点的迭代学习控制

---

## 1 引言与动机

在移动机器人的高速轨迹跟踪任务中，名义运动学模型通常假设车轮与地面之间不存在滑移。然而，实际运行时地面摩擦力的空间不均匀性、车体惯性效应以及执行器延迟等因素，会使机器人在特定空间位置处反复出现相似的跟踪偏差。这类偏差的显著特征是**空间重复性**——同一弯道、同一段坡面在每次经过时均会产生类似的误差模式——这一特征恰好为迭代学习控制 (Iterative Learning Control, ILC) 提供了天然的适用前提。

传统反馈控制器（如 PID、标准 MPC）只能在误差发生之后进行修正。ILC 则利用重复任务的结构性，从历史迭代中学习前馈补偿信号，使系统在下一次经过同一位置时能够**提前**抵消可预测的扰动。

在本系统中，ILC 的学习成果并不直接叠加在最终控制输出上——那样会绕过 MPPI 的代价评估与安全性保障。取而代之的设计是：将 ILC 偏置注入为 MPPI 采样分布的**均值偏移 (Mean Shift)**，即作为一种控制先验 (Control Prior)。该策略在保留避障安全性的前提下，使采样分布集中于经验指示的高质量区域，显著提升采样效率与跟踪精度。

---

## 2 问题建模

### 2.1 系统描述

本系统控制一台四轮全向转向 (Swerve Drive) 移动机器人。状态向量定义为

$$
\mathbf{x} = \begin{bmatrix} x & y & \psi & v_x & v_y & \omega \end{bmatrix}^\top \in \mathbb{R}^6
$$

其中 $(x, y)$ 为全局坐标系下的位置，$\psi$ 为航向角，$(v_x, v_y, \omega)$ 为体坐标系下的前向速度、侧向速度与角速度。控制输入为体坐标系速度指令

$$
\mathbf{u} = \begin{bmatrix} v_x^{cmd} & v_y^{cmd} & \omega^{cmd} \end{bmatrix}^\top \in \mathbb{R}^3
$$

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

## 3 空间索引 ILC 的建立

### 3.1 空间索引 vs. 时间索引

经典 ILC 以时间步 $t$ 为索引。然而，移动机器人每次通过同一空间位置的时刻 $t$ 因速度不同而变化，导致时间索引下同一物理位置的经验无法对齐。相比之下，地面摩擦特性、坡度等物理量与**空间位置**强相关，因此本系统采用基于路径点索引 $i$（等价于弧长 $s_i$）的空间 ILC。

### 3.2 ILC 记忆库

ILC 记忆库 $\mathcal{M}$ 是一个与参考路径等长的查找表 (Lookup Table)，每个条目存储一个三维速度偏置向量：

$$
\mathcal{M} = \left\{\mathbf{b}_i = \begin{bmatrix} \delta v_{x,i} & \delta v_{y,i} & \delta\omega_i \end{bmatrix}^\top\right\}_{i=0}^{N-1}
$$

初始化时所有偏置为零向量。当系统收到新的参考路径且路径几何发生实质性变化时，可根据策略选择清空记忆库或保留已有学习成果。

### 3.3 最近路径点索引的确定

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

---

## 4 跟踪误差的定义

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

---

## 5 ILC 更新律

### 5.1 带遗忘因子的一阶更新律

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

### 5.2 遗忘因子与稳态分析

遗忘因子 $\gamma$ 控制历史偏置的衰减速率。考虑路径点 $i$ 处持续存在恒定扰动导致的稳态误差 $e_0$，令稳态偏置 $b_\infty = b^{(k+1)} = b^{(k)}$，可解得

$$
b_\infty = \frac{\alpha \cdot e_0}{1 - \gamma}
$$

偏置从任意初始值衰减到 $1/e$ 所需的更新次数为 $n_{1/e} = -1/\ln\gamma$。以默认值 $\gamma = 0.995$ 为例，$n_{1/e} \approx 200$ 次；在 $50\;\text{Hz}$ 控制频率下对应约 $4\;\text{s}$ 的记忆半衰期。

$\gamma$ 的选取体现了**稳态精度**与**环境适应性**之间的折中。当 $\gamma \to 1$ 时遗忘极慢、稳态误差趋于零，但对环境变化的响应也极慢；当 $\gamma$ 较小时遗忘快、对新扰动适应迅速，但会引入与 $(1-\gamma)/(\alpha G)$ 成比例的残余稳态误差。

### 5.3 门控与限幅机制

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

## 6 ILC 先验注入 MPPI

本节阐述 ILC 偏置如何作为控制先验注入 MPPI 优化器，这是本系统区别于传统 ILC 直接前馈叠加的核心设计。

### 6.1 预测时域内的索引映射

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

### 6.2 采样分布的偏移

在标准 MPPI 中，第 $k$ 条采样轨迹的第 $t$ 步控制量以上一周期最优解为中心进行高斯扰动：

$$
\mathbf{u}_t^{(k)} = \mathbf{u}_t^{opt} + \boldsymbol{\epsilon}_t^{(k)}, \quad \boldsymbol{\epsilon}_t^{(k)} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})
$$

注入 ILC 先验后，采样中心偏移为

$$
\mathbf{u}_{center,t} = \mathbf{u}_t^{opt} + \mathbf{b}_{i_t}
$$

从而采样变为

$$
\mathbf{u}_t^{(k)} = \mathbf{u}_{center,t} + \boldsymbol{\epsilon}_t^{(k)} = \mathbf{u}_t^{opt} + \mathbf{b}_{i_t} + \boldsymbol{\epsilon}_t^{(k)}
$$

这一偏移的效果是：MPPI 不再在"上次最优解"附近盲目采样，而是在"上次最优解 + ILC 空间记忆修正"附近进行探索。若 ILC 偏置方向正确，则高质量样本的密度显著增大，采样效率随之提高。在概念上，MPPI 此时优化的是**相对于 ILC 先验的残差修正量**，而非绝对控制量，形成一种隐式的残差学习架构。

### 6.3 Exploitation 与 Exploration 采样的差异

MPPI 将 $K$ 条采样轨迹分为两组：exploitation 样本（占比 $1 - r$，默认 $90\%$）与 exploration 样本（占比 $r$，默认 $10\%$）。ILC 先验对两组的影响不同。

对于 exploitation 样本，控制量以偏移后的中心为均值：

$$
\mathbf{u}_t^{(k)} = \left(\mathbf{u}_t^{opt} + \mathbf{b}_{i_t}\right) + \boldsymbol{\epsilon}_t^{(k)}
$$

对于 exploration 样本，当先验探索注入启用时，以 ILC 偏置本身为均值：

$$
\mathbf{u}_t^{(k)} = \mathbf{b}_{i_t} + \boldsymbol{\epsilon}_t^{(k)}
$$

这意味着即使是大范围的随机探索，也围绕"经验合理区域"而非控制空间原点展开，减少了在无效区域的算力浪费。

此外，系统支持可选的先验正则化代价。当正则化权重 $w_{prior} > 0$ 时，在代价函数中增加二次惩罚项

$$
J_{prior} = w_{prior} \sum_{t=0}^{T-1} \|\mathbf{u}_t^{(k)} - \mathbf{u}_{center,t}\|^2
$$

鼓励采样控制量靠近先验偏移后的均值。默认 $w_{prior} = 0$，即 ILC 仅作为软建议影响采样分布，不改变代价函数结构。

### 6.4 输出阶段的叠加

MPPI 通过对所有采样轨迹按代价加权平均，得到最优残差控制序列 $\mathbf{u}^{opt*}$。最终输出为

$$
\mathbf{u}_{output} = \mathcal{F}\!\left(\mathbf{u}^{opt*} + \mathbf{U}_{prior}\right) + \mathbf{u}_{slip}
$$

其中 $\mathcal{F}(\cdot)$ 为可选的 Savitzky-Golay 平滑滤波，$\mathbf{u}_{slip}$ 为在线滑移补偿项。ILC 先验在此处被再次叠加，保证最终输出同时包含 MPPI 的优化残差与 ILC 的经验修正。

---

## 7 闭合路径处理

当机器人在闭合赛道上循环运行时，需要额外的周期性逻辑。

**闭合检测**。路径首尾点距离小于阈值 $d_{close}$（默认 $0.6\;\text{m}$）时，判定为闭合路径：

$$
\|\mathbf{p}_0 - \mathbf{p}_{N-1}\| < d_{close} \;\Longrightarrow\; \text{闭合}
$$

**索引环绕**。在最近点的加窗搜索中，若窗口超出 $[0, N-1]$ 的边界，则将超出部分从路径的另一端继续搜索。先验索引构建中，弧长预测值取模 $\hat{s}_t \leftarrow \hat{s}_t \bmod L$，保证索引正确环绕。

**移动目标点**。对于闭合路径，MPPI 的目标点不固定在路径末尾（否则机器人在起终重合区域会触发"到达目标"而停车），而是沿路径前方动态推进：

$$
s_{goal} = s_{i^*} + \max\!\left(v \cdot T \cdot \Delta t,\; d_{min}\right), \qquad s_{goal} \leftarrow s_{goal} \bmod L
$$

**圈次检测**。通过监测弧长坐标的跳变实现：当 $s_{current} + \delta_m < s_{previous}$（默认 $\delta_m = 0.5\;\text{m}$）时，判定完成一圈。每圈结束后系统记录横向 RMSE（区分直道与弯道）、航向 RMSE、ILC 偏置 RMS、饱和次数以及更新增量 RMS 等指标，并输出至日志与 CSV 文件，用于离线分析 ILC 的圈间收敛行为。

---

## 8 收敛性与稳定性分析

### 8.1 单点收敛条件

考虑路径点 $i$ 处的 ILC 学习动力学。假设从偏置到跟踪误差的简化传递关系为

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

当 $\gamma \to 1$ 时 $e_\infty \to 0$，可实现渐近零稳态误差。当 $\gamma < 1$ 时残余误差与 $(1-\gamma)/(\alpha G)$ 成正比。这再次印证了遗忘因子在鲁棒性（快速适应新扰动）与精度（消除稳态误差）之间的折中角色。

### 8.2 与 MPPI 的协同稳定性

ILC 作为先验注入 MPPI 后，不会破坏 MPPI 自身的最优性保证。其原因有三：

1. **代价函数独立性**。ILC 偏置仅改变采样分布的均值，不改变代价函数的定义。若 ILC 建议的方向存在碰撞风险，碰撞惩罚将使该区域的采样获得极高代价，在加权平均中被自然抑制。

2. **探索样本提供纠错能力**。探索组（默认 $10\%$）的采样保证了即使 ILC 先验完全错误，MPPI 仍有概率发现正确的控制策略。

3. **先验正则化的柔性**。默认 $w_{prior} = 0$，ILC 先验为纯粹的软建议。即使设置 $w_{prior} > 0$，其效果也仅是在代价函数中增加一个二次软约束，可被更强的碰撞代价或跟踪代价所推翻。

### 8.3 多点耦合效应的讨论

在实际系统中，路径点 $i$ 的偏置通过系统动力学影响相邻点的跟踪误差，形成点间耦合。然而以下因素使耦合效应得到有效抑制：(1) 每个控制周期 MPPI 均从当前状态重新优化，隐含地吸收了偏置引入的状态偏移；(2) 偏置饱和防止了局部偏置的过度积累与传播；(3) 遗忘因子的持续衰减确保无偏置的无限增长。因此，在实际运行中可将各路径点视为弱耦合的独立学习单元进行分析。

---

## 9 本节小结

本节详细阐述了基于位置路径点索引的迭代学习控制模块的完整理论框架。核心贡献概括如下：

1. **空间索引 ILC**。以路径点索引（等价于累积弧长）为记忆库的键，解决了时间索引在变速运动中的对齐困难。

2. **鲁棒更新律**。通过遗忘因子、误差死区、曲率衰减、单步限幅、偏置饱和等多级机制，保证了学习过程在复杂运行条件下的安全性与稳定性。

3. **MPPI 先验注入**。将 ILC 偏置作为 MPPI 采样分布的均值偏移而非直接控制叠加，在保留避障安全性的前提下实现了经验驱动的采样效率提升，形成隐式残差学习架构。

4. **闭合路径全支持**。弧长环绕、窗口搜索环绕、移动目标点、圈次自动检测与多维度指标统计，构成了适用于循环跟踪任务的完整工程方案。

5. **收敛性保证**。给出了单点收敛条件的解析推导和稳态残余误差的闭式表达，并论证了 ILC-MPPI 融合架构的协同稳定性。
