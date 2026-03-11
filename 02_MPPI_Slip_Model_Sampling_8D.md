# MPPI 控制器：引入滑移的运动模型、采样优化与八维执行

---

## 1 引言

模型预测路径积分 (Model Predictive Path Integral, MPPI) 控制器是本系统轨迹跟踪的核心决策层。与传统基于梯度的模型预测控制 (MPC) 不同，MPPI 通过蒙特卡洛采样与信息论加权实现无梯度的随机最优控制，天然适用于非凸代价函数和非线性动力学。

本章围绕 MPPI 控制器的完整工作流程展开。§2 在体坐标系三维速度空间 $(v_x, v_y, \omega)$ 中建立引入侧向滑移的运动学模型；§3 阐述 ILC 先验注入下的采样策略与前向仿真；§4 给出代价函数的统一形式化定义；§5 描述在线滑移因子估计器；§6 推导信息论加权与最优序列更新；§7 将三维速度指令经逆运动学映射升维为四轮全向转向平台的八维执行指令；§8 给出滑移补偿层的前馈-反馈双通道结构。

---

## 2 引入滑移的三维运动模型

### 2.1 名义运动学

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

### 2.2 滑移耦合效应

在高速转弯时，由于离心惯性力与轮胎-地面接触力的非线性耦合，机器人的实际侧向速度会偏离指令值。这种耦合效应的物理本质是：当机器人同时执行前向运动 ($v_x \neq 0$) 和转向运动 ($\omega \neq 0$) 时，轮胎的侧偏刚度有限，在离心力作用下产生不可忽视的侧向滑移。

本系统采用一阶耦合滑移模型来描述这一效应。定义标量滑移因子 $K_s \geq 0$，引入滑移修正后的有效侧向速度为

$$
v_{y,eff} = v_y + \underbrace{(-K_s \cdot v_x \cdot \omega)}_{\text{滑移修正项}} = v_y - K_s \, v_x \, \omega
$$

滑移修正项的符号和形式具有明确的物理含义：当机器人向前行驶 ($v_x > 0$) 并左转 ($\omega > 0$) 时，离心力使车体向右侧滑，表现为负的附加侧向速度 $-K_s v_x \omega < 0$。

### 2.3 滑移修正运动学

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

此模型在 MPPI 的前向仿真（轨迹展开）中充当状态转移函数。滑移因子 $K_s$ 由在线估计器自适应学习（见 §5），在每个控制周期的所有采样轨迹中共享同一估计值。

---

## 3 外部控制先验与采样策略

上一章从 ILC 的角度阐述了空间偏置的学习过程与先验注入的设计动机。本节从 MPPI 控制器的角度出发，说明 MPPI 的采样框架如何自然地容纳外部控制先验，以及先验注入对优化过程的具体影响。

### 3.1 MPPI 采样框架中的先验位置

标准 MPPI 在每个控制周期维护一个最优控制序列 $\mathbf{U}^{opt}$，并以其作为下一周期的采样中心。这一序列可视为 MPPI 对"当前应执行何种控制"的**最佳猜测**——它源自上一周期的加权平均更新，携带了关于目标方向、避障约束和速度规划的综合信息。

然而，$\mathbf{U}^{opt}$ 仅反映了上一个控制周期末的在线优化结果。它不包含对**当前空间位置处特定扰动**的显式记忆——即使机器人在上一圈的相同弯道处观测到了相似的侧向偏移，这一经验不会被 $\mathbf{U}^{opt}$ 自动携带到下一圈。MPPI 的"记忆"仅限于上一控制周期到当前周期之间的序列平移，而非跨圈次的空间经验。

外部控制先验 $\mathbf{U}_{prior}$ 填补的正是这一空白。在本系统中，$\mathbf{U}_{prior}$ 来自 ILC 记忆库（上一章 §3–5），长度为预测时域 $T$。先验注入后，采样中心变为

$$
\bar{\mathbf{u}}_t = \mathbf{u}_t^{opt} + \mathbf{b}_{i_t}, \quad t = 0, 1, \ldots, T-1
$$

其中 $\mathbf{b}_{i_t}$ 为路径索引 $i_t$ 处的 ILC 偏置（索引映射详见上一章 §6.3）。至此，MPPI 的采样中心同时包含了两类信息：$\mathbf{U}^{opt}$ 提供的在线优化记忆，与 $\mathbf{U}_{prior}$ 提供的跨圈次空间记忆。

### 3.2 先验注入对采样分布的影响

先验注入在 MPPI 的两组采样中以不同方式生效。Exploitation 样本（占比 $1-r$）以 $\bar{\mathbf{u}}_t$ 为中心扰动，搜索范围由噪声协方差 $\boldsymbol{\Sigma}$ 控制；Exploration 样本（占比 $r$）以 $\mathbf{b}_{i_t}$ 本身为中心，不依赖上周期最优解。各采样的完整控制量为

$$
\mathbf{u}_t^{(k)} = \mathrm{clamp}\!\left(\mathbf{u}_{base,t}^{(k)} + \boldsymbol{\epsilon}_t^{(k)}\right), \quad
\boldsymbol{\epsilon}_t^{(k)} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})
$$

从 MPPI 优化器的角度，先验注入的关键效应可归纳为两点。

**第一，提高有效采样密度。** MPPI 的 softmax 权重计算（§6.1）使得仅代价接近最低值的样本获得非零权重。在标准 MPPI 中，采样中心 $\mathbf{U}^{opt}$ 与真实最优解之间的偏移完全依赖 MPPI 自身的迭代更新来消除。ILC 先验通过缩小这一偏移量，使更多样本落入低代价区域，从而在固定 $K$ 下获得更多有效样本。

**第二，保留代价函数的裁决权。** 先验注入仅改变采样分布的均值，不修改代价函数的任何项（碰撞惩罚、速度约束、滑移风险等）。在前向仿真和代价累加阶段，每条样本仍需经过完整的代价评估。这保证了即使 ILC 偏置方向错误，代价函数也能将其影响抑制——从权重计算的角度看，错误偏置导致的高代价样本在 softmax 归一化后权重趋近于零。

### 3.3 与信息论加权的联系

将先验注入置于 §6 的信息论框架下考察。MPPI 的加权更新本质上是用提议分布 $q(\mathbf{U})$ 对最优控制分布 $p^*(\mathbf{U}) \propto \exp(-J/\lambda)$ 做重要性采样。重要性采样的方差与 $q$ 和 $p^*$ 之间的 KL 散度正相关。先验注入将 $q$ 的均值从 $\mathbf{U}^{opt}$ 偏移到 $\mathbf{U}^{opt} + \mathbf{U}_{prior}$，当 $\mathbf{U}_{prior}$ 的方向正确时，$q$ 的均值更接近 $p^*$ 的众数，KL 散度减小，重要性采样方差随之降低。

这解释了为什么 ILC 先验的注入位置必须在采样阶段而非输出阶段：只有在采样前修改 $q$ 的参数，才能影响所有 $K$ 条样本的生成过程，从而在信息论意义上提升 MPPI 的求解质量。若在输出端叠加（不改变 $q$），$K$ 条样本的采样和评估不受影响，ILC 贡献完全脱离了 MPPI 的优化框架。

### 3.4 前向仿真

给定初始状态 $\mathbf{x}_0$ 和第 $k$ 条采样控制序列 $\{\mathbf{u}_t^{(k)}\}_{t=0}^{T-1}$，利用 §2.3 的离散动力学模型逐步前向仿真，生成对应的状态轨迹：

$$
\mathbf{x}_{t+1}^{(k)} = f\!\left(\mathbf{x}_t^{(k)},\; \mathbf{u}_t^{(k)},\; K_s\right), \quad t = 0, 1, \ldots, T-1
$$

前向仿真是 MPPI 计算的主要瓶颈。由于 $K$ 条轨迹之间相互独立，系统采用 OpenMP 多线程并行加速，将 $K$ 条轨迹分配到多个 CPU 核心同时计算。

---

## 4 代价函数

代价函数是 MPPI 采样评估的核心，其设计直接决定了控制器的跟踪性能与行驶品质。本系统的代价函数由**阶段代价** $c_t$ 和**终端代价** $c_T$ 两部分构成，第 $k$ 条采样轨迹的总代价为

$$
J^{(k)} = \sum_{t=0}^{T-1} c_t\!\left(\mathbf{x}_t^{(k)},\; \mathbf{u}_t^{(k)},\; \mathbf{u}_{t-1}^{(k)}\right) + c_T\!\left(\mathbf{x}_T^{(k)},\; \mathbf{x}_{goal}\right)
$$

阶段代价按功能可分为四类：路径跟踪代价、滑移感知代价、控制平滑代价和信息论代价。下面逐一给出形式化定义。

### 4.1 路径跟踪代价

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

投影速度的定义使得仅当机器人沿路径切线方向行驶时才被视为"有效速度"，避免了侧向漂移时的虚假速度匹配。

路径跟踪代价的合项为

$$
c_{track} = c_{pos} + c_{head} + c_{vel}
$$

### 4.2 滑移感知代价

为抑制高速转弯时的侧向滑移风险，系统引入两类与滑移因子 $K_s$ 耦合的代价项。

**滑移风险惩罚**。对预测滑移量的二次惩罚，直接抑制高风险的速度-角速度组合输入：

$$
c_{slip} = w_{slip} \cdot (K_s \cdot v_x \cdot \omega)^2
$$

该项的物理含义是：当 $v_x$ 和 $\omega$ 同时较大时（即高速转弯），预测的侧向滑移量 $K_s v_x \omega$ 增大，代价相应上升，迫使优化器选择更保守的控制策略。

**曲率自适应限速**。根据路径局部曲率 $\kappa$ 和有效摩擦系数 $\mu_{eff}$ 计算安全过弯速度上限，对超速采样施加二次惩罚。曲率通过参考航向图的前向差分估计：在当前位置沿参考航向前看距离 $l_a$ 处查询航向变化量，得到

$$
\kappa \approx \frac{|\Delta\psi_{ahead}|}{l_a}
$$

有效摩擦系数考虑了滑移退化效应：

$$
\mu_{eff} = \mu_0 \cdot \max(0.3,\; 1 - 2K_s)
$$

其中 $\mu_0$ 为标称摩擦系数。对于全向机器人，由于其运动学结构允许任意方向平移，引入曲率下界 $\kappa_{floor}$ 以防止直线段的安全速度趋于无穷。进一步采用 Sigmoid 函数对曲率做软饱和处理，避免阈值附近的不连续性：

$$
\kappa_{eff} = \kappa_{floor} + \sigma(\kappa) \cdot (\kappa - \kappa_{floor}), \qquad \sigma(\kappa) = \frac{1}{1 + e^{-10(\kappa - 0.3)}}
$$

安全过弯速度与代价为

$$
v_{safe} = \sqrt{\frac{\mu_{eff} \cdot g}{\kappa_{eff}}}, \qquad c_{curv} = w_{curv} \cdot \max\!\left(0,\; v - v_{safe}(1 + m_v)\right)^2
$$

其中 $g$ 为重力加速度，$m_v$ 为速度裕度系数。

**弯道偏航率跟踪**。仅在弯道（$\kappa > \kappa_{th}$）中激活，引导 MPPI 生成与路径曲率匹配的转向动作。期望角速度由航向偏差经比例控制器生成，代价权重随曲率强度线性增长：

$$
\omega_{des} = \mathrm{clamp}\!\left(-2\Delta\psi_{ahead},\; \pm 2.0\right)
$$

$$
c_{yaw} = w_{yaw} \cdot \min\!\left(1,\; \frac{\kappa - \kappa_{th}}{0.5}\right) \cdot (\omega - \omega_{des})^2
$$

曲率门控的设计意图在于：直线段上航向对齐已由 $c_{head}$ 负责，无需额外的偏航率约束；弯道中才需要主动引导角速度以实现预瞄式转向。

滑移感知代价的合项为

$$
c_{slip\_aware} = c_{slip} + c_{curv} + c_{yaw}
$$

### 4.3 控制平滑代价

为抑制控制指令在相邻时间步之间的剧烈跳变，系统在两个层次上施加平滑惩罚。

**体速度层平滑**。对三维体速度指令的变化量施加二次惩罚：

$$
c_{smooth,body} = \sum_{j \in \{v_x, v_y, \omega\}} w_{j}^{cmd} \cdot (\Delta u_j)^2, \qquad \Delta\mathbf{u}_t = \mathbf{u}_t - \mathbf{u}_{t-1}
$$

**车轮指令层平滑**。通过逆运动学将体速度映射为八维车轮指令（四个转向角 + 四个轮速），对八维变化量施加独立的二次惩罚：

$$
c_{smooth,wheel} = \sum_{j=1}^{8} w_{j}^{wheel} \cdot (\Delta q_j)^2, \qquad \Delta\mathbf{q}_t = h(\mathbf{u}_t) - h(\mathbf{u}_{t-1})
$$

其中 $h(\cdot)$ 为三维到八维的逆运动学映射（详见 §7）。双层平滑设计的动机是：体速度层平滑保证指令的总体连续性，车轮指令层平滑则直接约束实际执行机构的变化率，避免了因逆运动学的非线性放大效应导致的转向关节抖动。

### 4.4 控制代价

路径积分控制理论要求代价函数包含二次控制代价 $\frac{\lambda}{2}\,\mathbf{u}^\top \boldsymbol{\Sigma}^{-1} \mathbf{u}$，惩罚控制输入的幅值。该项与噪声协方差共享参数 $\boldsymbol{\Sigma}$，是 HJB 方程通过指数变换线性化的数学前提，因此不可省略。在采样实现中，原始二次代价按 $\mathbf{u} = \bar{\mathbf{u}} + \boldsymbol{\epsilon}$ 展开后，常数项与噪声二次项在归一化权重时消去，保留下来的有效形式为

$$
c_{ctrl} = \lambda(1-\alpha)\;\bar{\mathbf{u}}^\top\,\boldsymbol{\Sigma}^{-1}\,\mathbf{u}
$$

其中 $\bar{\mathbf{u}}$ 为当前采样中心（§3.1），$\alpha \in [0,1]$ 为插值系数，$\alpha = 1$ 时控制代价消失，$\alpha = 0$ 时完全生效。默认取 $\alpha = 0.975$，仅保留 $2.5\%$ 的强度，在理论完备性与工程实用性之间折中。

### 4.5 终端代价

终端代价评估预测时域末端状态与目标点的距离偏差，仅在距目标较远时激活，以避免在目标邻域内产生不必要的梯度：

$$
c_T = \begin{cases}
w_T \cdot \|\mathbf{p}_T - \mathbf{p}_{goal}\|^2, & \|\mathbf{p}_T - \mathbf{p}_{goal}\| > 0.5\;\text{m} \\
0, & \text{otherwise}
\end{cases}
$$

### 4.6 总代价汇总

综合以上各项，第 $k$ 条采样轨迹在第 $t$ 步的阶段代价为

$$
c_t = \underbrace{c_{pos} + c_{head} + c_{vel}}_{\text{路径跟踪}} + \underbrace{c_{slip} + c_{curv} + c_{yaw}}_{\text{滑移感知}} + \underbrace{c_{smooth,body} + c_{smooth,wheel}}_{\text{控制平滑}} + \underbrace{c_{ctrl}}_{\text{控制代价}}
$$

各代价项的权重及其默认值汇总于表 1。

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

## 5 在线滑移因子估计

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

估计器对实际速度信号施加一阶低通滤波以抑制测量噪声，并维护残差的滑动窗口统计量用于收敛性判断。

估计得到的 $K_s$ 在当前周期内被所有 $K$ 条采样轨迹的前向仿真共享，同时也被前馈滑移补偿器使用。

---

## 6 信息论加权与最优序列更新

### 6.1 代价归一化与权重计算

在 $K$ 条轨迹的代价 $\{J^{(1)}, J^{(2)}, \ldots, J^{(K)}\}$ 计算完毕后，通过 softmax 变换将代价转化为归一化权重。首先找到最小代价 $J_{min} = \min_k J^{(k)}$，然后计算

$$
w^{(k)} = \frac{\exp\!\left(-\frac{1}{\lambda}(J^{(k)} - J_{min})\right)}{\sum_{j=1}^{K}\exp\!\left(-\frac{1}{\lambda}(J^{(j)} - J_{min})\right)}
$$

温度参数 $\lambda$ 控制权重分布的"锐度"：$\lambda \to 0$ 时退化为贪心选择（仅取最低代价轨迹），$\lambda \to \infty$ 时退化为均匀平均。默认 $\lambda = 100$。

减去 $J_{min}$ 的操作在数学上不影响归一化结果，但在数值实现中至关重要——它防止了指数函数的溢出。

### 6.2 最优控制序列更新

最优控制序列通过噪声的加权平均进行更新：

$$
\mathbf{u}_t^{opt} \leftarrow \mathbf{u}_t^{opt} + \sum_{k=1}^{K} w^{(k)} \cdot \boldsymbol{\epsilon}_t^{(k)}, \quad t = 0, 1, \ldots, T-1
$$

注意这里加权的是**噪声**而非完整控制量。这一设计保证了更新的方向性：如果某个方向的噪声普遍导致低代价，则最优序列沿该方向偏移；反之则保持不变。更新后对控制量做速度限幅。

### 6.3 控制输出与平滑滤波

更新后的最优序列与 ILC 先验偏置（§3.1）叠加，得到总控制序列：

$$
\mathbf{u}_t^{total} = \mathbf{u}_t^{opt} + \mathbf{b}_{i_t}, \quad t = 0, 1, \ldots, T-1
$$

取序列的第一个元素 $\mathbf{u}_0^{total}$ 作为当前周期的输出。在输出前，可选地通过 Savitzky-Golay 多项式滤波器对控制序列进行平滑。该滤波器在以当前时刻为中心的 $(2n+1)$ 窗口内拟合 $p$ 阶多项式，取中心点的拟合值作为平滑输出：

$$
\mathbf{u}_{smooth} = \sum_{i=-n}^{n} c_i \cdot \mathbf{u}_i
$$

其中 $\{c_i\}$ 为预计算的 SG 滤波系数，窗口包含 $n$ 个历史时刻和 $n$ 个未来预测时刻（来自最优序列）。默认半窗口宽度 $n = 4$，多项式阶数 $p = 6$。

---

## 7 从三维体速度到八维轮指令的升维映射

### 7.1 四轮全向转向的运动学结构

四轮全向转向 (Swerve Drive) 平台的每个车轮拥有两个独立自由度：转向角 $\phi_i$ 和轮速 $\Omega_i$（$i \in \{fl, fr, rl, rr\}$）。因此，底层执行空间为八维向量

$$
\mathbf{q} = \begin{bmatrix} \phi_{fl} & \phi_{fr} & \phi_{rl} & \phi_{rr} & \Omega_{fl} & \Omega_{fr} & \Omega_{rl} & \Omega_{rr} \end{bmatrix}^\top \in \mathbb{R}^8
$$

而 MPPI 的控制输出仅为三维体速度 $(v_x, v_y, \omega)$。从三维到八维的映射由**逆运动学**完成，是一个从操作空间到执行空间的升维过程。

### 7.2 各轮速度分量的推导

设车体质心为原点，体坐标系下 $x$ 轴指向前方、$y$ 轴指向左方、$\omega$ 以逆时针为正。四轮相对于质心的几何参数为：前轴距质心 $l_f$，后轴距质心 $l_r$，左轮距质心 $d_l$，右轮距质心 $d_r$。由此可写出各轮毂中心在体坐标系中的位置矢量：

$$
\mathbf{r}_{fl} = \begin{pmatrix} l_f \\ d_l \end{pmatrix}, \quad
\mathbf{r}_{fr} = \begin{pmatrix} l_f \\ -d_r \end{pmatrix}, \quad
\mathbf{r}_{rl} = \begin{pmatrix} -l_r \\ d_l \end{pmatrix}, \quad
\mathbf{r}_{rr} = \begin{pmatrix} -l_r \\ -d_r \end{pmatrix}
$$

当车体以速度 $(v_x, v_y)$ 平移并以角速度 $\omega$ 绕质心旋转时，各轮毂中心的速度由刚体运动学给出：

$$
\mathbf{v}_i = \begin{bmatrix} v_x \\ v_y \end{bmatrix} + \omega \times \mathbf{r}_i
$$

在二维平面中，叉乘 $\omega \times \mathbf{r}_i$ 展开为 $(-\omega\, r_{y,i},\; \omega\, r_{x,i})^\top$。以前左轮 (FL) 为例，$\mathbf{r}_{fl} = (l_f,\; d_l)^\top$，代入得

$$
\mathbf{v}_{fl} = \begin{bmatrix} v_x - \omega\, d_l \\ v_y + \omega\, l_f \end{bmatrix}
$$

对四个轮依次展开，各轮在体坐标系中的速度分量为

$$
\begin{aligned}
v_{x,fl} &= v_x - \omega\, d_l, &\quad v_{y,fl} &= v_y + \omega\, l_f \\
v_{x,fr} &= v_x + \omega\, d_r, &\quad v_{y,fr} &= v_y + \omega\, l_f \\
v_{x,rl} &= v_x - \omega\, d_l, &\quad v_{y,rl} &= v_y - \omega\, l_r \\
v_{x,rr} &= v_x + \omega\, d_r, &\quad v_{y,rr} &= v_y - \omega\, l_r
\end{aligned}
$$

其中 $v_{x,i}$ 为第 $i$ 轮的前向速度分量，$v_{y,i}$ 为侧向速度分量。可以观察到：同侧两轮（FL 与 RL、FR 与 RR）的 $v_x$ 分量相同（因为它们与质心的横向距离相同），同轴两轮（FL 与 FR、RL 与 RR）的 $v_y$ 分量相同（因为它们与质心的纵向距离相同）。

### 7.3 转向角与轮速的计算

**转向角**。各轮的转向角定义为轮速矢量相对于车体 $x$ 轴（前向）的偏转角，由双参数反正切函数给出：

$$
\phi_i = \mathrm{atan2}(v_{y,i},\; v_{x,i})
$$

当 $v_{y,i} > 0$（轮速矢量偏向左方）时 $\phi_i > 0$，对应逆时针偏转，与角速度 $\omega$ 的正方向一致。

**轮速**。各轮的合速度幅值除以轮胎半径 $r$，得到轮轴角速度：

$$
\Omega_i = \frac{\sqrt{v_{x,i}^2 + v_{y,i}^2}}{r}
$$

### 7.4 完整映射的矩阵表示

将上述关系整理为从体速度 $(v_x, v_y, \omega)$ 到各轮速度分量的线性映射：

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

该映射是**线性**的，但后续从 $(v_{x,i}, v_{y,i})$ 到 $(\phi_i, \Omega_i)$ 的极坐标变换是非线性的。因此，整体的三维到八维映射可以表示为

$$
\mathbf{q} = h(\mathbf{u}): \quad \mathbb{R}^3 \to \mathbb{R}^8
$$

其中 $h$ 由线性矩阵乘法加逐轮的 $\arctan$ 与范数运算复合而成。

### 7.5 升维映射的执行位置

在系统架构中，三维到八维的映射在两个位置实现。在 MPPI 核心控制器内部（`dynamics.hpp` 的 `bodyToWheelCommands`），该映射用于计算八维车辆指令的变化量代价（控制平滑代价中的八维分量）。而在实际执行层，该映射由 Gazebo 仿真环境中的 `vel_driver` 节点完成：该节点订阅三维 `Twist` 消息 $(v_x, v_y, \omega)$，经过上述逆运动学计算后，将八个标量指令分别发布到 Gazebo 的 `ros_control` 插件，驱动仿真机器人的四个转向关节和四个驱动关节。

这种分层设计使得 MPPI 始终在紧凑的三维空间中进行采样与优化——避免了在八维空间中采样导致的维度灾难——而将确定性的升维映射推迟到执行层，不增加优化的计算负担。

---

## 8 滑移补偿层

在 MPPI 求解出最优三维速度指令后、发送给执行层之前，系统通过滑移补偿层对指令进行修正，抵消预测的侧向滑移。

### 8.1 前馈补偿

基于当前估计的滑移因子 $K_s$，前馈补偿项为

$$
\Delta v_{y,ff} = -\gamma_{ff} \cdot K_s \cdot v_x \cdot \omega
$$

其中 $\gamma_{ff}$ 为前馈增益（默认 $0.7$）。该项本质上是对 §2.2 中滑移修正项的直接反向抵消：模型预测机器人会产生 $-K_s v_x \omega$ 的侧向漂移，因此在指令中预先加入等量反向补偿。增益 $\gamma_{ff} < 1$ 提供了一定的保守裕度，避免因 $K_s$ 估计不精确导致的过补偿。

### 8.2 闭环反馈补偿

前馈补偿依赖于模型的准确性。为消除模型残差导致的稳态跟踪误差，系统叠加闭环反馈修正。反馈补偿量由比例-积分结构生成：

$$
\Delta v_{y,fb} = -K_p^{lat} \cdot e_{lat} - K_i \cdot \int e_{lat}\, dt
$$

$$
\Delta\omega_{fb} = -K_p^{head} \cdot e_{head}
$$

其中 $e_{lat}$ 为横向跟踪误差，$e_{head}$ 为航向误差，$K_p^{lat}$、$K_p^{head}$、$K_i$ 分别为比例和积分增益。积分项消除恒定偏置导致的稳态误差，并设有抗饱和限幅和接近目标时的衰减机制。

### 8.3 总输出

最终发送到执行层的三维速度指令为

$$
\mathbf{u}_{output} = \mathbf{u}_{MPPI} + \begin{bmatrix} 0 \\ \Delta v_{y,ff} + \Delta v_{y,fb} \\ \Delta\omega_{fb} \end{bmatrix}
$$

经速度限幅后，由 vel_driver 映射为八维轮指令并执行。

---

## 9 本节小结

本章完整阐述了 MPPI 控制器从建模、采样、代价评估到执行的全链路。核心要点概括如下：

1. **三维滑移运动模型**（§2）。在名义运动学基础上引入一阶耦合滑移项 $-K_s v_x \omega$，以标量参数 $K_s$ 紧凑地刻画侧向滑移的速度-角速度耦合效应，在保持模型简洁性的同时提升高速转弯场景的预测精度。

2. **ILC 先验注入的采样策略**（§3）。ILC 偏置作为 MPPI 采样分布的均值偏移注入，形成 exploitation/exploration 分组策略。MPPI 求解相对于 ILC 先验的残差修正量，构成隐式残差学习架构。

3. **统一的多目标代价函数**（§4）。代价函数由路径跟踪（位置、航向、速度）、滑移感知（滑移风险、曲率限速、偏航率跟踪）、控制平滑（体速度层 + 车轮指令层）和信息论项四类构成，各项权重独立可调。

4. **在线滑移估计**（§5）。通过梯度下降法实时学习滑移因子 $K_s$，无需预先标定，使控制器能够自适应不同地面条件。

5. **信息论加权与 Savitzky-Golay 平滑**（§6）。采样轨迹经 softmax 加权平均更新最优序列，输出前施加多项式平滑滤波以抑制采样噪声。

6. **三维到八维升维映射**（§7）。通过逆运动学将体速度指令确定性地映射为四轮的转角与轮速，执行层实现在 vel_driver 节点中。这一分层设计使 MPPI 始终在紧凑的三维空间中优化，避免维度灾难，同时保留了全向转向平台的完整运动能力。

7. **多层滑移补偿**（§8）。MPPI 最优指令在输出前经过前馈-反馈双通道滑移补偿，融合模型预测（前馈）与实测误差（反馈），实现高速工况下的厘米级跟踪精度。
