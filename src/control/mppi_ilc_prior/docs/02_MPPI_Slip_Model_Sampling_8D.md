# MPPI 控制器：引入滑移的运动模型、采样优化与八维执行

---

## 1 引言

模型预测路径积分 (Model Predictive Path Integral, MPPI) 控制器是本系统轨迹跟踪的核心决策层。与传统基于梯度的 MPC 不同，MPPI 通过蒙特卡洛采样与信息论加权实现无梯度的随机最优控制，天然适用于非凸代价函数和非线性动力学。

本章围绕 MPPI 控制器的三个层次展开：第一，在体坐标系三维速度空间 $(v_x, v_y, \omega)$ 中建立引入侧向滑移的运动学模型（§2）；第二，以该模型为前向仿真器，通过随机采样、代价评估与信息论加权求解最优控制序列（§3–§5）；第三，将三维速度指令经逆运动学映射升维为四轮全向转向平台的八维执行指令——四个转角加四个轮速（§6）。

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

此模型在 MPPI 的前向仿真（轨迹展开）中充当状态转移函数。滑移因子 $K_s$ 由在线估计器自适应学习（见 §4），在每个控制周期的所有采样轨迹中共享同一估计值。

---

## 3 MPPI 采样与轨迹展开

### 3.1 控制序列的表示

MPPI 在长度为 $T$ 的预测时域上工作。系统维护一个最优控制序列

$$
\mathbf{U}^{opt} = \left[\mathbf{u}_0^{opt},\; \mathbf{u}_1^{opt},\; \ldots,\; \mathbf{u}_{T-1}^{opt}\right]
$$

该序列在控制周期之间持续更新，作为下一次采样的中心。

### 3.2 噪声采样

在每个控制周期中，系统生成 $K$ 条随机采样轨迹。第 $k$ 条轨迹的第 $t$ 步噪声向量独立采自零均值高斯分布：

$$
\boldsymbol{\epsilon}_t^{(k)} \sim \mathcal{N}(\mathbf{0},\; \boldsymbol{\Sigma}), \quad
\boldsymbol{\Sigma} = \mathrm{diag}(\sigma_{v_x}^2,\; \sigma_{v_y}^2,\; \sigma_\omega^2)
$$

$K$ 条样本被分为两组：exploitation 组（占比 $1 - r$）与 exploration 组（占比 $r$），默认 $r = 0.1$。

**Exploitation 样本**（$k < K(1-r)$）以上一周期最优解为中心扰动：

$$
\mathbf{u}_t^{(k)} = \mathbf{u}_t^{opt} + \boldsymbol{\epsilon}_t^{(k)}
$$

**Exploration 样本**（$k \geq K(1-r)$）仅包含纯噪声：

$$
\mathbf{u}_t^{(k)} = \boldsymbol{\epsilon}_t^{(k)}
$$

当存在外部控制先验 $\mathbf{b}_t$（如来自 ILC 的偏置）时，exploitation 的中心变为 $\mathbf{u}_t^{opt} + \mathbf{b}_t$，exploration 的中心变为 $\mathbf{b}_t$（见第一章 §6）。所有样本在采样后做速度限幅。

### 3.3 前向仿真

给定初始状态 $\mathbf{x}_0$ 和第 $k$ 条采样控制序列 $\{\mathbf{u}_t^{(k)}\}_{t=0}^{T-1}$，利用 §2.3 的离散动力学模型逐步前向仿真，生成对应的状态轨迹：

$$
\mathbf{x}_{t+1}^{(k)} = f\!\left(\mathbf{x}_t^{(k)},\; \mathbf{u}_t^{(k)},\; K_s\right), \quad t = 0, 1, \ldots, T-1
$$

前向仿真是 MPPI 计算的主要瓶颈。由于 $K$ 条轨迹之间相互独立，系统采用 OpenMP 多线程并行加速，将 $K$ 条轨迹分配到多个 CPU 核心同时计算。

---

## 4 代价函数与滑移估计

### 4.1 阶段代价

每条采样轨迹的总代价由逐步累加的阶段代价 $c_t$ 和终端代价 $c_T$ 构成：

$$
J^{(k)} = \sum_{t=0}^{T-1} c_t\!\left(\mathbf{x}_t^{(k)}, \mathbf{u}_t^{(k)}\right) + c_T\!\left(\mathbf{x}_T^{(k)}\right)
$$

阶段代价包含以下成分：

**路径跟踪代价**。利用栅格地图 (Grid Map) 查询当前状态点的距离误差 $d(\mathbf{x})$ 和参考航向 $\psi_{ref}(\mathbf{x})$，以及速度误差 $\Delta v$：

$$
c_{track} = w_d \cdot d(\mathbf{x})^2 + w_\psi \cdot \Delta\psi(\mathbf{x})^2 + w_v \cdot \Delta v^2
$$

其中 $\Delta\psi = \mathrm{wrap}_{[-\pi,\pi]}(\psi - \psi_{ref})$，$\Delta v = \|\mathbf{v}\| - v_{ref}$。

**碰撞惩罚**。从碰撞代价栅格图中查询碰撞风险值 $c_{col}(\mathbf{x})$，以大权重 $w_{col}$ 施加惩罚。

**滑移风险代价**。对高滑移风险的控制输入施加惩罚。滑移量的预测值为 $|K_s \cdot v_x \cdot \omega|$，通过二次代价抑制过激的高速转弯操作：

$$
c_{slip} = w_{slip} \cdot (K_s \cdot v_x \cdot \omega)^2
$$

**曲率限速代价**。根据路径曲率和有效摩擦系数计算安全过弯速度上限 $v_{safe}$，对超速的采样施加惩罚：

$$
v_{safe} = \sqrt{\frac{\mu_{eff} \cdot g}{\max(\kappa,\; \kappa_{floor})}}, \qquad c_{curv} = w_{curv} \cdot \max(0,\; v - v_{safe} + m_v)^2
$$

其中 $\mu_{eff} = \mu_0(1 - 2K_s)$ 为考虑滑移退化后的等效摩擦系数，$\kappa_{floor}$ 为全向机器人的最小等效曲率下界，$m_v$ 为速度裕度。

**转速跟踪代价**。在弯道中将期望角速度与实际角速度的偏差纳入代价，引导 MPPI 生成与路径曲率匹配的转向行为：

$$
c_{yaw} = w_{yaw} \cdot (\omega - \omega_{des})^2
$$

**控制平滑代价**。惩罚相邻时间步的控制量变化 $\Delta\mathbf{u}_t = \mathbf{u}_t - \mathbf{u}_{t-1}$，抑制指令的剧烈波动。

### 4.2 信息论代价项

MPPI 算法源自路径积分控制理论，其最优性推导依赖于将控制代价项嵌入自由能变分框架。具体地，在阶段代价中加入信息论项

$$
c_{info,t} = \lambda(1-\alpha)\; \mathbf{u}_t^{opt\top} \boldsymbol{\Sigma}^{-1} \mathbf{u}_t^{(k)}
$$

其中 $\lambda$ 为温度参数，$\alpha$ 为控制平滑系数（默认 $0.975$）。该项的物理意义是在自由能最小化的框架下保证采样分布与最优控制分布之间 KL 散度的有界性。

### 4.3 终端代价

终端代价评估预测时域末端状态与目标的偏差：

$$
c_T = w_T \cdot \left[\|\mathbf{p}_T - \mathbf{p}_{goal}\|^2 + \Delta\psi_T^2\right]
$$

### 4.4 在线滑移因子估计

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

## 5 信息论加权与最优序列更新

### 5.1 代价归一化与权重计算

在 $K$ 条轨迹的代价 $\{J^{(1)}, J^{(2)}, \ldots, J^{(K)}\}$ 计算完毕后，通过 softmax 变换将代价转化为归一化权重。首先找到最小代价 $J_{min} = \min_k J^{(k)}$，然后计算

$$
w^{(k)} = \frac{\exp\!\left(-\frac{1}{\lambda}(J^{(k)} - J_{min})\right)}{\sum_{j=1}^{K}\exp\!\left(-\frac{1}{\lambda}(J^{(j)} - J_{min})\right)}
$$

温度参数 $\lambda$ 控制权重分布的"锐度"：$\lambda \to 0$ 时退化为贪心选择（仅取最低代价轨迹），$\lambda \to \infty$ 时退化为均匀平均。默认 $\lambda = 100$。

减去 $J_{min}$ 的操作在数学上不影响归一化结果，但在数值实现中至关重要——它防止了指数函数的溢出。

### 5.2 最优控制序列更新

最优控制序列通过噪声的加权平均进行更新：

$$
\mathbf{u}_t^{opt} \leftarrow \mathbf{u}_t^{opt} + \sum_{k=1}^{K} w^{(k)} \cdot \boldsymbol{\epsilon}_t^{(k)}, \quad t = 0, 1, \ldots, T-1
$$

注意这里加权的是**噪声**而非完整控制量。这一设计保证了更新的方向性：如果某个方向的噪声普遍导致低代价，则最优序列沿该方向偏移；反之则保持不变。更新后对控制量做速度限幅。

### 5.3 控制输出与平滑滤波

更新后的最优序列与控制先验叠加，得到总控制序列：

$$
\mathbf{u}_t^{total} = \mathbf{u}_t^{opt} + \mathbf{b}_t, \quad t = 0, 1, \ldots, T-1
$$

取序列的第一个元素 $\mathbf{u}_0^{total}$ 作为当前周期的输出。在输出前，可选地通过 Savitzky-Golay 多项式滤波器对控制序列进行平滑。该滤波器在以当前时刻为中心的 $(2n+1)$ 窗口内拟合 $p$ 阶多项式，取中心点的拟合值作为平滑输出：

$$
\mathbf{u}_{smooth} = \sum_{i=-n}^{n} c_i \cdot \mathbf{u}_i
$$

其中 $\{c_i\}$ 为预计算的 SG 滤波系数，窗口包含 $n$ 个历史时刻和 $n$ 个未来预测时刻（来自最优序列）。默认半窗口宽度 $n = 4$，多项式阶数 $p = 6$。

---

## 6 从三维体速度到八维轮指令的升维映射

### 6.1 四轮全向转向的运动学结构

四轮全向转向 (Swerve Drive) 平台的每个车轮拥有两个独立自由度：转向角 $\phi_i$ 和轮速 $\Omega_i$（$i \in \{fl, fr, rl, rr\}$）。因此，底层执行空间为八维向量

$$
\mathbf{q} = \begin{bmatrix} \phi_{fl} & \phi_{fr} & \phi_{rl} & \phi_{rr} & \Omega_{fl} & \Omega_{fr} & \Omega_{rl} & \Omega_{rr} \end{bmatrix}^\top \in \mathbb{R}^8
$$

而 MPPI 的控制输出仅为三维体速度 $(v_x, v_y, \omega)$。从三维到八维的映射由**逆运动学**完成，是一个从操作空间到执行空间的升维过程。

### 6.2 各轮速度分量的推导

设车体质心为原点，前轴距质心 $l_f$，后轴距质心 $l_r$，左轮距质心 $d_l$，右轮距质心 $d_r$。体坐标系下前向为 $x$ 轴、左向为 $y$ 轴。

当车体以速度 $(v_x, v_y)$ 平移并以角速度 $\omega$ 绕质心旋转时，各轮毂中心的速度为质心速度加上旋转贡献。以前左轮 (FL) 为例，其位置矢量为 $\mathbf{r}_{fl} = (-d_l,\; l_f)^\top$，旋转贡献为 $\omega \times \mathbf{r}_{fl}$。在二维平面中，

$$
\mathbf{v}_{fl} = \begin{bmatrix} v_x \\ v_y \end{bmatrix} + \omega \begin{bmatrix} -l_f \\ -d_l \end{bmatrix} = \begin{bmatrix} v_x - \omega\, l_f \\ v_y - \omega\, d_l \end{bmatrix}
$$

> **坐标系说明**。在实际的 Gazebo 仿真 URDF 模型中，车体坐标系与上述体坐标系存在轴向差异：URDF 模型的前向为 $y$ 轴、右向为 $x$ 轴。因此在底层驱动 (vel_driver) 中，接收到的 `Twist` 消息中 `linear.x` 对应车体前向 $v_f$，`linear.y` 的取反对应右向 $v_r$。换算关系为 $v_r = -v_y^{msg}$，$v_f = v_x^{msg}$。在此坐标系下，各轮速度分量为

$$
\begin{aligned}
v_{x,fl} &= v_r - \omega\, l_f, &\quad v_{y,fl} &= v_f - \omega\, d_l \\
v_{x,fr} &= v_r - \omega\, l_f, &\quad v_{y,fr} &= v_f + \omega\, d_r \\
v_{x,rl} &= v_r + \omega\, l_r, &\quad v_{y,rl} &= v_f - \omega\, d_l \\
v_{x,rr} &= v_r + \omega\, l_r, &\quad v_{y,rr} &= v_f + \omega\, d_r
\end{aligned}
$$

此处 $v_{x,i}$ 为第 $i$ 轮在其安装平面内的横向分量，$v_{y,i}$ 为纵向分量。前左与前右的 $v_x$ 分量相同（因为它们与质心的纵向距离 $l_f$ 相同），后左与后右同理。左侧与右侧的 $v_y$ 分量因横向臂长 $d_l$、$d_r$ 不同而差异化。

### 6.3 转向角与轮速的计算

**转向角**。各轮的转向角由其速度分量的反正切得到：

$$
\phi_i = \arctan\!\left(\frac{-v_{x,i}}{v_{y,i}}\right)
$$

取负号是因为在 URDF 关节定义中，正转向角对应的偏转方向与 $v_x$ 分量的正方向相反。

**轮速**。各轮的合速度幅值除以轮胎半径 $r$，得到轮轴角速度：

$$
\Omega_i = \frac{\sqrt{v_{x,i}^2 + v_{y,i}^2}}{r}
$$

### 6.4 完整映射的矩阵表示

将上述关系整理为从体速度 $(v_r, v_f, \omega)$ 到各轮速度分量的线性映射：

$$
\begin{bmatrix}
v_{x,fl} \\ v_{y,fl} \\ v_{x,fr} \\ v_{y,fr} \\ v_{x,rl} \\ v_{y,rl} \\ v_{x,rr} \\ v_{y,rr}
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 & -l_f \\
0 & 1 & -d_l \\
1 & 0 & -l_f \\
0 & 1 & +d_r \\
1 & 0 & +l_r \\
0 & 1 & -d_l \\
1 & 0 & +l_r \\
0 & 1 & +d_r
\end{bmatrix}
\begin{bmatrix}
v_r \\ v_f \\ \omega
\end{bmatrix}
$$

该映射是**线性**的，但后续从 $(v_{x,i}, v_{y,i})$ 到 $(\phi_i, \Omega_i)$ 的极坐标变换是非线性的。因此，整体的三维到八维映射可以表示为

$$
\mathbf{q} = h(\mathbf{u}): \quad \mathbb{R}^3 \to \mathbb{R}^8
$$

其中 $h$ 由线性矩阵乘法加逐轮的 $\arctan$ 与范数运算复合而成。

### 6.5 升维映射的执行位置

在系统架构中，三维到八维的映射在两个位置实现。在 MPPI 核心控制器内部（`dynamics.hpp` 的 `bodyToWheelCommands`），该映射用于计算八维车辆指令的变化量代价（控制平滑代价中的八维分量）。而在实际执行层，该映射由 Gazebo 仿真环境中的 `vel_driver` 节点完成：该节点订阅三维 `Twist` 消息 $(v_x, v_y, \omega)$，经过上述逆运动学计算后，将八个标量指令分别发布到 Gazebo 的 `ros_control` 插件，驱动仿真机器人的四个转向关节和四个驱动关节。

这种分层设计使得 MPPI 始终在紧凑的三维空间中进行采样与优化——避免了在八维空间中采样导致的维度灾难——而将确定性的升维映射推迟到执行层，不增加优化的计算负担。

---

## 7 滑移补偿层

在 MPPI 求解出最优三维速度指令后、发送给执行层之前，系统通过滑移补偿层对指令进行修正，抵消预测的侧向滑移。

### 7.1 前馈补偿

基于当前估计的滑移因子 $K_s$，前馈补偿项为

$$
\Delta v_{y,ff} = -\gamma_{ff} \cdot K_s \cdot v_x \cdot \omega
$$

其中 $\gamma_{ff}$ 为前馈增益（默认 $0.7$）。该项本质上是对 §2.2 中滑移修正项的直接反向抵消：模型预测机器人会产生 $-K_s v_x \omega$ 的侧向漂移，因此在指令中预先加入等量反向补偿。增益 $\gamma_{ff} < 1$ 提供了一定的保守裕度，避免因 $K_s$ 估计不精确导致的过补偿。

### 7.2 闭环反馈补偿

前馈补偿依赖于模型的准确性。为消除模型残差导致的稳态跟踪误差，系统叠加闭环反馈修正。反馈补偿量由比例-积分结构生成：

$$
\Delta v_{y,fb} = -K_p^{lat} \cdot e_{lat} - K_i \cdot \int e_{lat}\, dt
$$

$$
\Delta\omega_{fb} = -K_p^{head} \cdot e_{head}
$$

其中 $e_{lat}$ 为横向跟踪误差，$e_{head}$ 为航向误差，$K_p^{lat}$、$K_p^{head}$、$K_i$ 分别为比例和积分增益。积分项消除恒定偏置导致的稳态误差，并设有抗饱和限幅和接近目标时的衰减机制。

### 7.3 总输出

最终发送到执行层的三维速度指令为

$$
\mathbf{u}_{output} = \mathbf{u}_{MPPI} + \begin{bmatrix} 0 \\ \Delta v_{y,ff} + \Delta v_{y,fb} \\ \Delta\omega_{fb} \end{bmatrix}
$$

经速度限幅后，由 vel_driver 映射为八维轮指令并执行。

---

## 8 本节小结

本章完整阐述了 MPPI 控制器从建模、优化到执行的全链路。核心要点概括如下：

1. **三维滑移运动模型**。在名义运动学基础上引入一阶耦合滑移项 $-K_s v_x \omega$，以标量参数 $K_s$ 紧凑地刻画侧向滑移的速度-角速度耦合效应，在保持模型简洁性的同时提升高速转弯场景的预测精度。

2. **蒙特卡洛采样优化**。在三维体速度空间中生成 $K$ 条采样轨迹，利用滑移修正运动学前向仿真，通过包含路径跟踪、碰撞安全、滑移风险、曲率限速等多目标的代价函数评估，并以信息论 softmax 加权实现无梯度最优控制。

3. **在线滑移估计**。通过梯度下降法实时学习滑移因子 $K_s$，无需预先标定，使控制器能够自适应不同地面条件。

4. **Savitzky-Golay 平滑**。在最优序列输出前施加多项式平滑滤波，抑制采样带来的指令噪声。

5. **三维到八维升维映射**。通过逆运动学将体速度指令确定性地映射为四轮的转角与轮速，执行层实现在 vel_driver 节点中。这一分层设计使 MPPI 始终在紧凑的三维空间中优化，避免维度灾难，同时保留了全向转向平台的完整运动能力。

6. **多层滑移补偿**。MPPI 最优指令在输出前经过前馈-反馈双通道滑移补偿，融合模型预测（前馈）与实测误差（反馈），实现高速工况下的厘米级跟踪精度。
