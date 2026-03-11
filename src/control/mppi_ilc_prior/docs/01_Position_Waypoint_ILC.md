# 第一章 基于位置路径点的迭代学习控制 (Position-Waypoint-Indexed ILC)

> **所属系统**: `mppi_ilc_prior` — 将迭代学习控制 (ILC) 作为采样先验注入 MPPI 优化器的控制架构  
> **文档版本**: v1.0 | 2026-02

---

## 目录

- [1.1 引言与动机](#11-引言与动机)
- [1.2 问题建模：全向移动机器人的轨迹跟踪](#12-问题建模全向移动机器人的轨迹跟踪)
- [1.3 迭代学习控制的基本原理](#13-迭代学习控制的基本原理)
- [1.4 空间索引 ILC 的建立](#14-空间索引-ilc-的建立)
  - [1.4.1 参考路径离散化与累积弧长计算](#141-参考路径离散化与累积弧长计算)
  - [1.4.2 ILC 记忆库的数据结构](#142-ilc-记忆库的数据结构)
  - [1.4.3 最近点索引查找与连续性保持](#143-最近点索引查找与连续性保持)
- [1.5 跟踪误差的定义与计算](#15-跟踪误差的定义与计算)
  - [1.5.1 横向误差 $e_{lat}$](#151-横向误差-e_lat)
  - [1.5.2 航向误差 $e_{head}$](#152-航向误差-e_head)
  - [1.5.3 路径曲率 $\kappa$ 的估计](#153-路径曲率-kappa-的估计)
- [1.6 ILC 更新律的推导与设计](#16-ilc-更新律的推导与设计)
  - [1.6.1 一阶 ILC 更新律](#161-一阶-ilc-更新律)
  - [1.6.2 遗忘因子 $\gamma$ 的作用与分析](#162-遗忘因子-gamma-的作用与分析)
  - [1.6.3 学习增益 $\alpha$ 的选择](#163-学习增益-alpha-的选择)
  - [1.6.4 更新门控机制](#164-更新门控机制)
  - [1.6.5 饱和限幅](#165-饱和限幅)
- [1.7 ILC 先验注入 MPPI 优化器](#17-ilc-先验注入-mppi-优化器)
  - [1.7.1 预测时域内的索引序列构建](#171-预测时域内的索引序列构建)
  - [1.7.2 先验控制序列的提取](#172-先验控制序列的提取)
  - [1.7.3 采样分布偏移机制](#173-采样分布偏移机制)
  - [1.7.4 先验在 Exploitation 与 Exploration 中的角色](#174-先验在-exploitation-与-exploration-中的角色)
- [1.8 闭合路径的特殊处理](#18-闭合路径的特殊处理)
  - [1.8.1 路径闭合检测](#181-路径闭合检测)
  - [1.8.2 索引环绕与弧长取模](#182-索引环绕与弧长取模)
  - [1.8.3 圈次检测与指标统计](#183-圈次检测与指标统计)
- [1.9 参数配置与工程调优指南](#19-参数配置与工程调优指南)
- [1.10 收敛性与稳定性分析](#110-收敛性与稳定性分析)
- [1.11 本章小结](#111-本章小结)

---

## 1.1 引言与动机

在移动机器人的高速轨迹跟踪任务中（例如赛道循环跟踪、自动巡检路线），系统面临一个核心矛盾：**名义运动学模型假设无滑移**，但实际运行中由于地面摩擦力的不均匀性、车体惯性效应、执行器延迟等因素，机器人在特定位置处总会出现类似的偏差模式。

这种偏差具有以下特征：

| 特征 | 说明 |
|------|------|
| **空间重复性** | 同一个弯道、同一段坡面每次经过时产生相似的误差模式 |
| **确定性主导** | 地面摩擦系数、路面倾斜角度等物理量与位置强相关 |
| **模型不可知** | 这些微观扰动难以被简单的运动学/动力学模型精确捕获 |

传统的反馈控制器（PID、MPC 等）只能在误差**发生之后**进行修正，这在高速运行时会导致不可忽视的滞后。而如果我们能够**记住**在路径上每个位置曾经遇到的误差并提前补偿，就可以实现前馈式的误差消除。

**迭代学习控制 (Iterative Learning Control, ILC)** 恰好提供了这样一种框架：它利用重复任务的结构性，从历史迭代中学习一个前馈补偿信号，使得系统在下一次迭代中能够提前抵消可预测的扰动。

在本系统中，我们将 ILC 与 MPPI (Model Predictive Path Integral) 控制器深度融合，形成 `mppi_ilc_prior` 架构。ILC 的学习成果**不是**直接叠加在控制输出上（那样会绕过 MPPI 的安全性保障），而是作为 MPPI 采样分布的**均值偏移 (Mean Shift)**，从而在保留避障能力的同时获得经验驱动的采样效率提升。

---

## 1.2 问题建模：全向移动机器人的轨迹跟踪

### 系统状态

本系统控制的是一台**四轮全向转向 (Swerve Drive)** 移动机器人。其状态向量定义为：

$$
\mathbf{x} = \begin{bmatrix} x \\ y \\ \psi \\ v_x \\ v_y \\ \omega \end{bmatrix} \in \mathbb{R}^6
$$

其中 $(x, y)$ 为全局坐标下的位置，$\psi$ 为航向角，$(v_x, v_y, \omega)$ 为体坐标系 (Body Frame) 下的前向速度、侧向速度和角速度。

### 控制输入

控制量为体坐标系下的速度指令：

$$
\mathbf{u} = \begin{bmatrix} v_x^{cmd} \\ v_y^{cmd} \\ \omega^{cmd} \end{bmatrix} \in \mathbb{R}^3
$$

该指令经过逆运动学映射后转化为四个独立转向轮的转角和轮速。

### 名义运动学模型

在理想条件下（无滑移），状态转移方程为：

$$
\begin{cases}
\dot{x} = v_x \cos\psi - v_y \sin\psi \\
\dot{y} = v_x \sin\psi + v_y \cos\psi \\
\dot{\psi} = \omega \\
\dot{v}_x = f_{vx}(\mathbf{u}) \\
\dot{v}_y = f_{vy}(\mathbf{u}) \\
\dot{\omega} = f_{\omega}(\mathbf{u})
\end{cases}
$$

然而在实际运行中，地面摩擦力的空间变化、执行器响应延迟等因素使得实际的速度响应偏离指令值。这种偏差正是 ILC 要学习和补偿的对象。

### 参考路径

参考路径以有序路径点序列的形式给出：

$$
\mathcal{P} = \{(\mathbf{p}_0, \psi_0), (\mathbf{p}_1, \psi_1), \ldots, (\mathbf{p}_{N-1}, \psi_{N-1})\}
$$

其中 $\mathbf{p}_i = (x_i, y_i)^\top$ 为第 $i$ 个路径点的二维坐标，$\psi_i$ 为该点处路径的切线方向角。路径点总数 $N$ 由 ROS 话题 `reference_path` 提供。

---

## 1.3 迭代学习控制的基本原理

### 经典 ILC 公式

考虑一个重复性任务，系统在第 $k$ 次迭代中执行任务，产生跟踪误差 $\mathbf{e}^{(k)}$。经典的一阶 ILC (First-order ILC，也称 P-type ILC) 更新律为：

$$
\mathbf{u}_{ff}^{(k+1)}(j) = \mathbf{u}_{ff}^{(k)}(j) + \mathbf{K}_l \cdot \mathbf{e}^{(k)}(j), \quad j = 0, 1, \ldots, N-1
$$

其中：
- $\mathbf{u}_{ff}^{(k)}(j)$ 是第 $k$ 次迭代在索引 $j$ 处的前馈控制量
- $\mathbf{K}_l$ 是学习增益矩阵
- $\mathbf{e}^{(k)}(j)$ 是第 $k$ 次迭代在索引 $j$ 处的跟踪误差

### 传统 ILC 的局限性

经典 ILC 存在以下问题：

1. **时间索引依赖**：传统 ILC 使用时间 $t$ 作为索引。但移动机器人每次通过同一位置的时间可能不同（因速度变化），导致时间索引无法正确对齐同一空间位置。

2. **非重复扰动的脆弱性**：如果环境发生微小变化（如地面状态改变），纯累积的 ILC 会导致旧经验"过拟合"，产生有害的前馈信号。

3. **与采样优化的割裂**：传统 ILC 将前馈直接叠加在反馈控制器输出上。如果前馈信号指向障碍物区域，反馈控制器无法阻止碰撞。

本系统针对上述三个问题分别提出了解决方案：**空间索引替代时间索引**、**指数遗忘因子**、以及**将 ILC 嵌入 MPPI 的采样先验**。

---

## 1.4 空间索引 ILC 的建立

### 1.4.1 参考路径离散化与累积弧长计算

当 ROS 节点收到参考路径消息 (`nav_msgs/Path`) 后，系统首先计算**累积弧长** (Cumulative Arc-length)，为每个路径点分配一个单调递增的弧长标量：

$$
s_0 = 0, \quad s_i = s_{i-1} + \|\mathbf{p}_i - \mathbf{p}_{i-1}\|_2, \quad i = 1, 2, \ldots, N-1
$$

在实现中（`mppi_ilc_prior_ros.cpp`），这一计算在 `refPathCallback` 中完成：

```cpp
ref_path_cum_s_.resize(ref_path_.poses.size(), 0.0);
for (std::size_t i = 1; i < ref_path_.poses.size(); ++i) {
    const auto& p0 = ref_path_.poses[i - 1].pose.position;
    const auto& p1 = ref_path_.poses[i].pose.position;
    ref_path_cum_s_[i] = ref_path_cum_s_[i - 1] + std::hypot(p1.x - p0.x, p1.y - p0.y);
}
```

路径总长度为 $L = s_{N-1}$。该弧长数组在后续的先验索引构建和圈次检测中起核心作用。

### 1.4.2 ILC 记忆库的数据结构

ILC 记忆库 (`ILCMemory`) 是一个与参考路径等长的查找表 (Lookup Table)，每个条目存储一个三维速度偏置向量：

$$
\mathcal{M} = \{(\delta v_x^{(i)}, \delta v_y^{(i)}, \delta \omega^{(i)})\}_{i=0}^{N-1}
$$

在 C++ 实现中，它被定义为 `std::vector<BodyVelocity>`：

```cpp
class ILCMemory {
    std::vector<BodyVelocity> bias_;  // 大小 = N（路径点数）
    // ...
};
```

每当收到新的参考路径时，记忆库通过 `resizeILC(N)` 重新分配大小。如果参数 `reset_on_new_path` 为 `true`，则所有偏置归零；否则保留前一条路径的学习成果（适用于路径仅微调的场景）。

**设计哲学**：记忆库的关键洞察是——扰动与**空间位置**绑定，而非与时间绑定。无论机器人以何种速度通过路径上的某一点，该点处的地面摩擦特性是不变的。因此，以路径点索引 $i$（或等价地以弧长 $s_i$）为键来存储偏置，比以时间步 $t$ 为键更加自然和鲁棒。

### 1.4.3 最近点索引查找与连续性保持

在每个控制周期，系统需要确定机器人当前位于参考路径的哪个位置索引 $i^*$。这看似是一个简单的最近邻搜索问题，但在以下情况中会变得复杂：

1. **自交叉路径 (Self-intersecting Paths)**：例如"8"字形赛道在交叉点处有两个候选最近点，分别属于不同的路径段。如果每次都选全局最近点，索引可能在两段之间频繁跳跃。

2. **闭合路径的起/终点附近**：在路径的起终点重合区域，全局最近点搜索可能在 $i = 0$ 和 $i = N-1$ 之间剧烈振荡。

为解决这些问题，系统采用了**加窗搜索 + 全局回退**的两级策略：

**第一级：窗口搜索 (Windowed Search)**

基于上一周期的最近点索引 $i_{prev}$，仅在 $[i_{prev} - W_b, \ i_{prev} + W_f]$ 的窗口内搜索：

$$
i^* = \arg\min_{i \in \text{window}} \left[ \|\mathbf{p}_i - \mathbf{p}_{robot}\|_2 + \lambda_h \cdot |e_{head}(i)| \right]
$$

其中 $\lambda_h$ 是航向权重 (`idx_heading_weight`，默认 0.4 m/rad)，引入航向一致性约束以区分自交叉点处的两个候选段。窗口参数为：
- $W_b = 25$（向后看 25 个索引）
- $W_f = 60$（向前看 60 个索引）

对于闭合路径，窗口会进行环绕 (wrap-around) 处理：

```cpp
if (path_is_closed_ && idx_allow_wraparound_) {
    if (last_closest_idx_ + fwd >= n) {
        // 从路径头部继续搜索
        const int wrap_max = (last_closest_idx_ + fwd) - (n - 1);
        for (int i = 0; i <= std::min(n - 1, wrap_max); ++i)
            evalCandidate(i, ...);
    }
    if (last_closest_idx_ - back < 0) {
        // 从路径尾部继续搜索
        const int wrap_min = n + (last_closest_idx_ - back);
        for (int i = std::max(0, wrap_min); i < n; ++i)
            evalCandidate(i, ...);
    }
}
```

**第二级：全局回退 (Global Fallback)**

窗口搜索的结果需要通过两项检验：
1. **航向门限** (Heading Gate): $|e_{head}(i^*)| \leq \theta_{gate}$（默认 1.2 rad）
2. **距离比率** (Distance Ratio): $d_{window} \leq \eta \cdot d_{global}$（默认 $\eta = 1.6$）

若任一检验不通过，则放弃窗口结果，回退至全局最近点。这保证了在机器人被外力推离路径后能够快速恢复到正确的路径段。

```cpp
const bool heading_bad = (std::abs(win_best_head) > idx_heading_gate_);
const bool dist_bad = (win_best_dist > idx_global_fallback_factor_ * global_dist);
if (!heading_bad && !dist_bad) {
    best_idx = win_best_idx;     // 使用窗口结果
} else {
    best_idx = global_idx;       // 回退到全局最近点
}
```

---

## 1.5 跟踪误差的定义与计算

在确定当前最近路径点索引 $i^*$ 后，系统计算两类关键跟踪误差。

### 1.5.1 横向误差 $e_{lat}$

横向误差定义为机器人位置在路径切线法线方向上的投影距离。设路径在 $i^*$ 处的切线方向角为 $\psi_{path}$，机器人相对该点的位移为 $\Delta \mathbf{p} = \mathbf{p}_{robot} - \mathbf{p}_{i^*}$，则：

$$
e_{lat} = -\Delta x \cdot \sin(\psi_{path}) + \Delta y \cdot \cos(\psi_{path})
$$

其中 $\Delta x = x_{robot} - x_{i^*}$，$\Delta y = y_{robot} - y_{i^*}$。

**符号约定**：$e_{lat} > 0$ 表示机器人位于路径切线方向的**左侧**（路径坐标系下），$e_{lat} < 0$ 表示在右侧。

```cpp
double path_yaw = tf2::getYaw(ref_path_.poses[best_idx].pose.orientation);
double dx = current_state_.x - ref_path_.poses[best_idx].pose.position.x;
double dy = current_state_.y - ref_path_.poses[best_idx].pose.position.y;
error.lateral_error = -dx * std::sin(path_yaw) + dy * std::cos(path_yaw);
```

### 1.5.2 航向误差 $e_{head}$

航向误差定义为机器人航向与路径切线方向之差：

$$
e_{head} = \text{wrap}_{[-\pi, \pi]}(\psi_{robot} - \psi_{path})
$$

其中 $\text{wrap}_{[-\pi, \pi]}(\cdot)$ 将角度归一化到 $[-\pi, \pi]$ 区间。

```cpp
error.heading_error = wrapAngle(current_state_.yaw - path_yaw);
```

### 1.5.3 路径曲率 $\kappa$ 的估计

路径曲率通过以 $i^*$ 为中心的三点法（Menger 曲率）估计。取前后各偏移 3 个索引的点 $\mathbf{p}_{i_0}, \mathbf{p}_{i^*}, \mathbf{p}_{i_2}$，利用三角形面积与外接圆半径的关系：

$$
\kappa = \frac{2 \cdot A(\mathbf{p}_{i_0}, \mathbf{p}_{i^*}, \mathbf{p}_{i_2})}{\|\mathbf{p}_{i_0} - \mathbf{p}_{i^*}\| \cdot \|\mathbf{p}_{i^*} - \mathbf{p}_{i_2}\| \cdot \|\mathbf{p}_{i_2} - \mathbf{p}_{i_0}\|}
$$

其中 $A$ 为三点构成的三角形面积（通过叉积计算），曲率符号由叉积的方向决定（正值为左转）。

```cpp
double area2 = std::abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
double denom = d01 * d12 * d20;
error.path_curvature = area2 / denom;
double cross = (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1);
error.path_curvature = std::copysign(error.path_curvature, cross);
```

曲率在 ILC 更新中起**门控**作用：高曲率区域的学习增益被衰减，以避免在弯道处的瞬态误差被过度学习（详见 §1.6.4）。

### 路径末端衰减

在路径的最后 5 个点附近，所有误差量按线性因子衰减：

$$
f_{end} = \frac{N - i^*}{5}, \quad e_{lat} \leftarrow e_{lat} \cdot f_{end}, \quad e_{head} \leftarrow e_{head} \cdot f_{end}
$$

这避免了在非闭合路径的末端产生不必要的大幅修正。

---

## 1.6 ILC 更新律的推导与设计

### 1.6.1 一阶 ILC 更新律

本系统采用**带遗忘因子的一阶 ILC 更新律** (Forgetting-factor P-type ILC)。在每个控制周期，系统根据当前最近路径点索引 $i^*$ 及对应的跟踪误差，就地更新记忆库。更新律的向量形式为：

$$
\mathbf{b}^{(k+1)}(i^*) = \gamma \cdot \mathbf{b}^{(k)}(i^*) + \mathbf{K}_l \cdot \mathbf{e}^{(k)}(i^*)
$$

展开到各分量：

$$
\delta v_x(i^*) \leftarrow \gamma \cdot \delta v_x(i^*)
$$

$$
\delta v_y(i^*) \leftarrow \gamma \cdot \delta v_y(i^*) + \alpha_{lat} \cdot e_{lat}(i^*)
$$

$$
\delta \omega(i^*) \leftarrow \gamma \cdot \delta \omega(i^*) + \alpha_{head} \cdot e_{head}(i^*)
$$

**关键观察**：

1. **$v_x$ 分量仅衰减不学习**：在当前实现中，前向速度偏置 $\delta v_x$ 仅受遗忘因子影响而逐渐衰减，不直接从跟踪误差中学习。这是因为前向速度主要由 MPPI 的参考速度 (`ref_velocity`) 控制，ILC 主要负责补偿侧向滑移和转向不足/过度。

2. **在线 (Online) 更新**：与传统 ILC 在每次迭代结束后批量更新不同，本系统在每个控制周期内实时更新。这意味着同一圈内后半段的学习成果会立即影响当前圈的控制（如果机器人已经绕过该点一次）。

3. **就地更新 (In-place Update)**：更新直接修改 `bias_[idx]`，无需存储完整的历史轨迹。

```cpp
void ILCMemory::update(std::size_t idx, double lateral_error, double heading_error)
{
    bias_[idx].vx    *= config_.decay;           // 衰减
    bias_[idx].vy    *= config_.decay;           // 衰减
    bias_[idx].omega *= config_.decay;           // 衰减

    bias_[idx].vy    += config_.k_lateral  * lateral_error;   // 学习
    bias_[idx].omega += config_.k_heading  * heading_error;   // 学习

    // 限幅（见 §1.6.5）
    bias_[idx].vx    = std::clamp(bias_[idx].vx,    -config_.max_bias_v,     config_.max_bias_v);
    bias_[idx].vy    = std::clamp(bias_[idx].vy,    -config_.max_bias_v,     config_.max_bias_v);
    bias_[idx].omega = std::clamp(bias_[idx].omega, -config_.max_bias_omega, config_.max_bias_omega);
}
```

### 1.6.2 遗忘因子 $\gamma$ 的作用与分析

遗忘因子 $\gamma \in (0, 1)$ 控制历史经验的衰减速率。其物理含义和数学性质如下：

**稳态偏置分析**：假设系统在路径点 $i$ 处持续遭受恒定扰动 $e_0$，则偏置的稳态值为：

$$
b_\infty = \lim_{k \to \infty} b^{(k)} = \frac{\alpha \cdot e_0}{1 - \gamma}
$$

这是因为在稳态时 $b^{(k+1)} = b^{(k)} = b_\infty$：

$$
b_\infty = \gamma \cdot b_\infty + \alpha \cdot e_0 \quad \Longrightarrow \quad b_\infty = \frac{\alpha \cdot e_0}{1 - \gamma}
$$

**时间常数**：偏置衰减到初始值 $1/e$ 所需的更新次数为：

$$
n_{1/e} = \frac{-1}{\ln \gamma}
$$

对于 $\gamma = 0.995$（默认值），$n_{1/e} \approx 200$ 次更新。在 50 Hz 控制频率下，这意味着约 4 秒的记忆半衰期。

**参数选择指南**：

| $\gamma$ 范围 | 适用场景 | 记忆特性 |
|:---:|---|---|
| $0.99 \sim 1.0$ | 环境高度稳定、扰动几乎不变 | 长记忆，慢适应 |
| $0.95 \sim 0.99$ | 环境可能缓慢变化 | 中等记忆 |
| $0.90 \sim 0.95$ | 环境频繁变化 | 短记忆，快适应 |
| $< 0.90$ | 通常不推荐 | 过度遗忘，ILC 效果甚微 |

### 1.6.3 学习增益 $\alpha$ 的选择

系统使用两个独立的学习增益：
- $\alpha_{lat}$ (`k_lateral`)：横向误差到侧向速度偏置的转换增益，默认 0.15
- $\alpha_{head}$ (`k_heading`)：航向误差到角速度偏置的转换增益，默认 0.05

**增益的物理单位**：
- $\alpha_{lat}$: $[\text{(m/s)} / \text{m}] = [\text{s}^{-1}]$——每米横向误差产生多少 m/s 的侧向速度补偿
- $\alpha_{head}$: $[\text{(rad/s)} / \text{rad}] = [\text{s}^{-1}]$——每弧度航向误差产生多少 rad/s 的角速度补偿

**稳定性约束**：ILC 学习律的收敛要求满足：

$$
0 < \alpha < \frac{2(1 - \gamma)}{G_{max}}
$$

其中 $G_{max}$ 是被控对象从控制输入到跟踪误差的最大增益。在实践中，由于 $G_{max}$ 难以精确获得，通常通过保守选择小学习增益来保证收敛。

### 1.6.4 更新门控机制

原始的 ILC 更新律在所有路径点上均匀应用。然而，某些情况下的更新可能是有害的，因此系统引入了多级门控：

**1) 误差死区 (Error Deadband)**

当误差绝对值小于死区阈值时，不进行学习更新：

$$
|e_{lat}| < \epsilon_{db} \Longrightarrow e_{lat} \leftarrow 0
$$

默认 $\epsilon_{db} = 0.005$ m。这避免了在误差已经足够小时继续积累微小偏置，防止数值漂移。

**2) 曲率衰减 (Curvature Attenuation)**

在高曲率区域（弯道），瞬态误差的主要来源是动态效应（如离心力引起的侧滑），而非稳态扰动。过度学习这些瞬态误差会导致在下一圈进入弯道时产生过大的预补偿。因此：

$$
|\kappa| > \kappa_{th} \Longrightarrow \begin{cases} e_{lat} \leftarrow 0.3 \cdot e_{lat} \\ e_{head} \leftarrow 0.3 \cdot e_{head} \end{cases}
$$

默认 $\kappa_{th} = 0.10$ m$^{-1}$。

```cpp
if (std::abs(error.path_curvature) > ilc_cfg_.curvature_threshold) {
    lat_err  *= 0.3;
    head_err *= 0.3;
}
```

**3) 单步更新限幅 (Per-step Update Clamp)**

每次更新的误差输入被限制在一个最大范围内，防止异常大误差（如碰撞后的突发偏移）导致偏置瞬间跳变：

$$
e_{lat} \leftarrow \text{clamp}(e_{lat}, -\Delta_{max,lat}, \Delta_{max,lat})
$$

默认 $\Delta_{max,lat} = 0.02$ m，$\Delta_{max,head} = 0.02$ rad。

**4) 低速防护 (Low-speed Guard)**

当机器人速度低于阈值 ($v < 0.3$ m/s) 时，跟踪误差按速度比例缩放：

$$
v < v_{th} \Longrightarrow \begin{cases} e_{lat} \leftarrow \frac{v}{v_{th}} \cdot e_{lat} \\ e_{head} \leftarrow \frac{v}{v_{th}} \cdot e_{head} \end{cases}
$$

这避免了在启动或停车阶段因低速导致的大航向偏差被错误地学习。

**5) 旋转防护 (Spin Guard)**

当角速度与线速度的比率过大时（机器人可能在原地旋转），横向误差被进一步衰减：

$$
\frac{|\omega|}{v + 0.01} > 2.0 \Longrightarrow e_{lat} \leftarrow \min\left(1, \frac{2.0}{\text{ratio}}\right) \cdot e_{lat}
$$

### 1.6.5 饱和限幅

偏置值被硬限制在物理可行范围内：

$$
|\delta v_x|, |\delta v_y| \leq v_{bias,max} = 0.6 \ \text{m/s}
$$

$$
|\delta \omega| \leq \omega_{bias,max} = 0.8 \ \text{rad/s}
$$

限幅的作用是双重的：
1. **安全性**：防止 ILC 学习到过大的补偿量，导致机器人突然侧移或急转
2. **数值稳定性**：在遗忘因子接近 1 时，无限幅的情况下偏置可能缓慢增长到不合理的数值

系统还会统计饱和事件的次数 (`metrics_ilc_saturation_v_`, `metrics_ilc_saturation_omega_`)，用于监控 ILC 是否处于"能力极限"。如果频繁饱和，可能说明学习增益过大或系统存在超出 ILC 补偿能力的扰动。

---

## 1.7 ILC 先验注入 MPPI 优化器

这是本系统区别于传统 ILC 的核心创新。ILC 学到的偏置不是直接叠加在控制输出上，而是作为 MPPI 优化器的**采样先验 (Sampling Prior)** 注入。

### 1.7.1 预测时域内的索引序列构建

MPPI 需要在预测时域 $T$ 步内的每一步都有对应的 ILC 偏置。这要求将未来 $T$ 步的时间步映射到路径上的空间索引。

系统支持两种索引模式：

**模式一：弧长-速度预测法 (Arclength-velocity Prediction)**（默认启用）

基于当前速度 $v$ 和预测步长 $\Delta t$，预测未来第 $t$ 步的弧长位置：

$$
s_t = s_{base} + v \cdot \Delta t \cdot t, \quad t = 0, 1, \ldots, T-1
$$

其中 $s_{base}$ 是当前最近点的弧长 $s_{i^*}$，$v = \sqrt{v_x^2 + v_y^2}$（如果当前速度低于阈值 $v_{min} = 0.2$ m/s，则使用参考速度 $v_{ref}$）。

然后通过二分查找 (`std::lower_bound`) 将弧长 $s_t$ 映射回路径点索引：

$$
i_t = \arg\min_{j} \{ s_j \geq s_t \}
$$

```cpp
for (int t = 0; t < T; ++t) {
    double target_s = base_s + speed * config_.mppi.step_dt * static_cast<double>(t);
    if (path_is_closed_ && path_length > 1e-6) {
        target_s = std::fmod(target_s, path_length);  // 弧长取模
    }
    auto it = std::lower_bound(ref_path_cum_s_.begin(), ref_path_cum_s_.end(), target_s);
    int idx = static_cast<int>(std::distance(ref_path_cum_s_.begin(), it));
    indices[t] = std::clamp(idx, 0, n - 1);
}
```

**模式二：固定步长法 (Fixed Index Step)**

简单地以固定步长递增索引：

$$
i_t = i^* + t \cdot \Delta_{step}, \quad t = 0, 1, \ldots, T-1
$$

默认 $\Delta_{step} = 1$。此模式在路径点分布不均匀或速度变化剧烈时可能不准确，但计算成本极低。

### 1.7.2 先验控制序列的提取

给定索引序列 $\{i_0, i_1, \ldots, i_{T-1}\}$，从 ILC 记忆库中提取对应的偏置向量，组装为先验控制序列：

$$
\mathbf{U}_{prior} = \left[ \mathbf{b}(i_0), \ \mathbf{b}(i_1), \ \ldots, \ \mathbf{b}(i_{T-1}) \right]
$$

```cpp
void MPPIILCPriorCore::applyILCPriorFromIndices(const std::vector<int>& indices) {
    mppi_hc::ControlSequence prior_seq;
    prior_seq.resize(T);
    for (int t = 0; t < T; ++t) {
        prior_seq[t] = ilc_memory_.getBias(static_cast<std::size_t>(idx));
    }
    mppi_core_.setControlPrior(prior_seq, ilc_cfg_.prior_weight, 
                                ilc_cfg_.prior_apply_to_exploration);
}
```

### 1.7.3 采样分布偏移机制

在标准 MPPI 中，第 $k$ 条采样轨迹的第 $t$ 步控制量为：

$$
\mathbf{u}_t^{(k)} = \mathbf{u}_{t}^{opt} + \boldsymbol{\epsilon}_t^{(k)}, \quad \boldsymbol{\epsilon}_t^{(k)} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})
$$

其中 $\mathbf{u}^{opt}$ 是上一周期的最优控制序列。注入 ILC 先验后，采样中心偏移为：

$$
\mathbf{u}_{center, t} = \mathbf{u}_{t}^{opt} + \underbrace{\mathbf{b}(i_t)}_{\text{ILC Prior}}
$$

$$
\mathbf{u}_t^{(k)} = \mathbf{u}_{center, t} + \boldsymbol{\epsilon}_t^{(k)}
$$

这意味着 MPPI 不再在"上次最优解"附近采样，而是在"上次最优解 + ILC 经验修正"附近采样。其效果是：

1. **提高采样效率**：如果 ILC 的偏置方向正确，则大量采样点自然集中在高质量区域，减少了无效采样的浪费。

2. **隐式残差学习**：MPPI 优化的对象变成了**相对于 ILC 先验的残差修正量** $\Delta \mathbf{u} = \mathbf{u}^{opt} - \mathbf{0}$，而非绝对控制量。这与深度学习中的 ResNet 思想类似——让优化器只需学习"还差多少"，而不是"完整答案是什么"。

### 1.7.4 先验在 Exploitation 与 Exploration 中的角色

MPPI 的采样分为两组：
- **Exploitation 样本**（默认 90%）：在上次最优解附近小幅扰动
- **Exploration 样本**（默认 10%）：大范围随机探索

ILC 先验对两组的影响不同：

**Exploitation 样本**：

$$
\mathbf{u}_t^{(k)} = (\mathbf{u}_{t}^{opt} + \mathbf{b}(i_t)) + \boldsymbol{\epsilon}_t^{(k)}
$$

先验叠加在最优解上，使得微调区域已经包含了经验修正。

**Exploration 样本**（当 `prior_apply_to_exploration = true` 时）：

$$
\mathbf{u}_t^{(k)} = \mathbf{b}(i_t) + \boldsymbol{\epsilon}_t^{(k)}
$$

探索样本以 ILC 先验为中心，而非以零为中心。这意味着即使完全随机的探索采样，也是在"合理的经验区域"附近进行，避免了在控制空间的无效区域浪费算力。

**输出阶段的再叠加**：

最终的控制输出在 MPPI 优化完成后，还会再次叠加 ILC 先验：

$$
\mathbf{u}_{output} = \text{Filter}\left(\mathbf{u}^{opt*} + \mathbf{b}\right) + \mathbf{u}_{slip\_comp}
$$

其中 $\mathbf{u}^{opt*}$ 是 MPPI 加权平均后的最优残差，`Filter` 是可选的 Savitzky-Golay 平滑滤波，$\mathbf{u}_{slip\_comp}$ 是在线滑移补偿项。

---

## 1.8 闭合路径的特殊处理

当机器人在闭合赛道上循环运行时，需要额外的逻辑来处理路径的周期性。

### 1.8.1 路径闭合检测

系统通过比较路径首尾点的距离来自动判断路径是否闭合：

$$
\|\mathbf{p}_0 - \mathbf{p}_{N-1}\| < d_{close} \quad \Longrightarrow \quad \text{path\_is\_closed} = \text{true}
$$

默认 $d_{close} = 0.6$ m。

```cpp
path_is_closed_ = (std::hypot(pN.x - p0.x, pN.y - p0.y) < idx_closed_path_threshold_);
```

### 1.8.2 索引环绕与弧长取模

对于闭合路径：

1. **弧长取模**：先验索引构建时，弧长预测值对路径总长取模：
   $$s_t \leftarrow s_t \mod L$$

2. **索引窗口环绕**：最近点搜索的窗口在路径末尾时"绕回"到开头（反之亦然），如 §1.4.3 所述。

3. **移动目标点**：对于闭合路径，MPPI 的目标点不再固定在路径末尾（否则会在起终点附近触发"到达目标"条件而停车），而是沿路径前方动态推进：

$$
s_{goal} = s_{base} + \max(v \cdot T \cdot \Delta t, \ d_{min})
$$

$$
s_{goal} \leftarrow s_{goal} \mod L
$$

```cpp
double target_s = base_s + std::max(horizon_lookahead, min_lookahead);
target_s = std::fmod(target_s, path_length);
```

### 1.8.3 圈次检测与指标统计

系统通过监测弧长的跳变来检测是否完成了一圈。当弧长从接近 $L$ 突然回到接近 0 时，判定为完成一圈：

$$
s_{current} + \delta_{margin} < s_{previous} \quad \Longrightarrow \quad \text{Lap completed}
$$

默认 $\delta_{margin} = 0.5$ m。

每圈结束时，系统输出以下统计指标：

| 指标 | 公式 | 说明 |
|------|------|------|
| 横向 RMSE | $\sqrt{\frac{1}{n}\sum e_{lat}^2}$ | 整圈平均横向误差 |
| 直道横向 RMSE | 仅 $\|\kappa\| < \kappa_{th}$ 的点 | 直道段跟踪精度 |
| 弯道横向 RMSE | 仅 $\|\kappa\| \geq \kappa_{th}$ 的点 | 弯道段跟踪精度 |
| 航向 RMSE | $\sqrt{\frac{1}{n}\sum e_{head}^2}$ (deg) | 航向跟踪精度 |
| ILC 偏置 RMS | $\sqrt{\frac{1}{N}\sum b_{vy}^2}$ | 已学习偏置的大小 |
| 偏置饱和次数 | 累计 | ILC 达到限幅上限的频率 |
| 更新增量 RMS | $\sqrt{\frac{1}{m}\sum \Delta b^2}$ | ILC 每步更新量的大小 |

这些指标会输出到 ROS 日志和 CSV 文件（路径由 `metrics/log_dir` 配置），用于离线分析 ILC 的学习过程和收敛行为。

---

## 1.9 参数配置与工程调优指南

以下是 ILC 相关参数的完整列表及调优建议，对应配置文件 `config/mppi_ilc_prior.yaml` 的 `ilc:` 节：

| 参数 | 默认值 | 范围建议 | 作用 |
|------|:------:|:--------:|------|
| `enabled` | `true` | bool | 总开关 |
| `reset_on_new_path` | `false` | bool | 新路径时是否清空记忆库 |
| `k_lateral` | 0.15 | 0.05 ~ 0.3 | 横向学习增益 $\alpha_{lat}$ |
| `k_heading` | 0.05 | 0.02 ~ 0.15 | 航向学习增益 $\alpha_{head}$ |
| `decay` | 0.995 | 0.90 ~ 0.999 | 遗忘因子 $\gamma$ |
| `max_bias_v` | 0.6 | 0.2 ~ 1.0 | 速度偏置饱和限幅 [m/s] |
| `max_bias_omega` | 0.8 | 0.3 ~ 1.5 | 角速度偏置饱和限幅 [rad/s] |
| `prior_weight` | 0.0 | 0.0 ~ 10.0 | 先验正则化权重（0 = 仅偏移采样中心） |
| `prior_apply_to_exploration` | `true` | bool | 探索样本是否也使用先验 |
| `prior_use_arclength` | `true` | bool | 使用弧长-速度法构建索引 |
| `prior_speed_min` | 0.2 | 0.1 ~ 0.5 | 低速时的索引速度下限 [m/s] |
| `prior_index_step` | 1 | 1 ~ 5 | 固定步长法的索引间隔 |
| `curvature_threshold` | 0.10 | 0.05 ~ 0.3 | 弯道检测曲率阈值 [1/m] |
| `error_deadband` | 0.005 | 0.001 ~ 0.02 | 误差死区 [m, rad] |
| `max_update_lateral` | 0.02 | 0.005 ~ 0.05 | 单步横向误差输入限幅 [m] |
| `max_update_heading` | 0.02 | 0.005 ~ 0.05 | 单步航向误差输入限幅 [rad] |

### 典型调优流程

1. **初始验证**：先将 `enabled` 设为 `false`，确认纯 MPPI 的跟踪性能基线。
2. **保守启动**：开启 ILC，使用默认参数。观察 2~3 圈后的 RMSE 变化趋势。
3. **增益调整**：
   - 如果横向 RMSE 下降缓慢 → 适当增大 `k_lateral`
   - 如果出现振荡（RMSE 圈间交替上升/下降）→ 减小 `k_lateral` 或增大 `decay`
4. **弯道调整**：如果弯道 RMSE 恶化 → 降低 `curvature_threshold` 或减小弯道衰减系数
5. **饱和监控**：如果 CSV 日志中 `sat_vy` 或 `sat_omega` 频繁计数 → 增大 `max_bias_v` / `max_bias_omega`，或检查是否存在超出 ILC 能力的系统性问题

---

## 1.10 收敛性与稳定性分析

### 单点收敛分析

考虑单个路径点 $i$ 处的 ILC 动力学。假设系统从控制偏置到跟踪误差的简化传递关系为：

$$
e^{(k)}(i) = e_0(i) - G \cdot b^{(k)}(i)
$$

其中 $e_0(i)$ 是无偏置时的稳态误差，$G > 0$ 是系统增益。代入更新律：

$$
b^{(k+1)} = \gamma \cdot b^{(k)} + \alpha \cdot (e_0 - G \cdot b^{(k)}) = (\gamma - \alpha G) \cdot b^{(k)} + \alpha e_0
$$

**收敛条件**：要求特征根的绝对值小于 1：

$$
|\gamma - \alpha G| < 1
$$

即：

$$
\frac{\gamma - 1}{G} < \alpha < \frac{\gamma + 1}{G}
$$

由于 $\gamma < 1$ 且 $\alpha, G > 0$，左侧总为负数，故实际约束为：

$$
\alpha < \frac{1 + \gamma}{G}
$$

**稳态误差**：收敛后的残余误差为：

$$
e_\infty = e_0 \cdot \frac{1 - \gamma}{1 - \gamma + \alpha G}
$$

当 $\gamma \to 1$（无遗忘），$e_\infty \to 0$，实现零稳态误差。当 $\gamma < 1$ 时，存在与 $\frac{1-\gamma}{\alpha G}$ 成正比的残余误差。这体现了遗忘因子带来的鲁棒性（适应环境变化）与稳态精度之间的折中。

### 多点耦合效应

在实际系统中，路径点 $i$ 的偏置不仅影响该点的跟踪误差，还会通过系统动力学影响相邻点的误差。然而，由于以下因素，点间耦合被有效抑制：

1. **MPPI 的重新优化**：每个控制周期 MPPI 都会根据当前状态重新求解，隐含地吸收了 ILC 偏置引入的扰动。
2. **偏置饱和**：限幅防止了局部偏置的过度积累和传播。
3. **遗忘因子**：持续的衰减确保了不会出现偏置的无限增长。

### 与 MPPI 的协同稳定性

ILC 作为先验注入 MPPI 时，不会破坏 MPPI 本身的最优性保证。原因是：

1. **ILC 偏置仅影响采样分布的均值，不改变代价函数**。即使 ILC 的偏置指向危险方向，MPPI 代价函数中的碰撞惩罚会使这些采样获得极高的代价，被加权平均所抑制。
2. **Exploration 样本提供了纠错能力**。10% 的探索样本保证了即使 ILC 先验完全错误，MPPI 仍有概率发现正确的控制策略。
3. **先验权重 $w_{prior}$ 提供了额外的柔性**。当 $w_{prior} = 0$（默认），ILC 先验是"软建议"；当 $w_{prior} > 0$，它变成"软约束"，但仍可被足够大的代价差异所推翻。

---

## 1.11 本章小结

本章详细阐述了 `mppi_ilc_prior` 系统中基于位置路径点索引的迭代学习控制模块的完整理论框架和实现细节。核心贡献可以概括为：

1. **空间索引 ILC**：以路径点（或等价的累积弧长）作为记忆库的索引键，解决了传统时间索引 ILC 在变速运动中的时间对齐问题。

2. **多级门控更新律**：通过误差死区、曲率衰减、单步限幅、低速/旋转防护等机制，保证了 ILC 学习过程的鲁棒性和安全性。

3. **MPPI 先验注入**：将 ILC 偏置作为 MPPI 的采样分布偏移而非直接控制叠加，在保留避障安全性的前提下提高了采样效率。

4. **闭合路径完整支持**：包括弧长环绕、窗口搜索环绕、移动目标点、圈次自动检测与指标统计等一整套机制。

5. **可配置的参数体系**：学习增益、遗忘因子、饱和限幅、门控阈值等参数均可通过 YAML 文件灵活调整。

在后续章节中，我们将进一步讨论：
- **第二章**：ILC 先验与 MPPI 采样优化的深度融合机制（包括正则化代价、多层滑移补偿的协同）
- **第三章**：实验评估与圈间收敛特性分析（基于 CSV 指标日志的定量分析方法）

---

> **代码参考**：
> - 记忆库数据结构：`include/mppi_ilc_prior/ilc_memory.hpp`
> - 核心求解器封装：`include/mppi_ilc_prior/mppi_ilc_prior_core.hpp` → `src/mppi_ilc_prior_core.cpp`
> - ROS 集成与控制循环：`include/mppi_ilc_prior/mppi_ilc_prior_ros.hpp` → `src/mppi_ilc_prior_ros.cpp`
> - 参数配置：`config/mppi_ilc_prior.yaml`
