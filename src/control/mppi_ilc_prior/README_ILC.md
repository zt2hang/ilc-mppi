# MPPI-ILC: Iterative Learning Control as a Prior

本文档详细介绍了 **ILC (Iterative Learning Control)** 如何与 **MPPI (Model Predictive Path Integral)** 控制器结合，形成一种具备“记忆”能力的鲁棒控制架构。

## 1. 核心理念：ILC 作为控制先验 (Control Prior)

传统的 ILC 通常作为一种附加的前馈项直接叠加在反馈控制器的输出上。然而，在 `mppi_ilc_prior` 架构中，我们采用了一种更深度的融合方式：**将 ILC 学习到的经验作为 MPPI 采样分布的均值（Mean）**。

这种方法的核心优势在于：
*   **引导采样 (Guided Sampling)**: MPPI 不再是盲目地在零附近采样，而是在“上次成功的经验”附近进行探索。
*   **保留避障能力**: ILC 仅提供一个建议（Prior），MPPI 仍然通过代价函数（Cost Function）评估所有轨迹。如果 ILC 的建议会导致碰撞（例如环境发生了变化），MPPI 会自动拒绝该建议并找到新的安全路径。

## 2. ILC 基础与原理 (Fundamentals)

### 2.1 什么是 ILC？
迭代学习控制 (ILC) 的核心思想是 **"Practice Makes Perfect" (熟能生巧)**。对于重复性任务（例如在固定赛道上跑圈），系统遇到的扰动往往也是重复的（例如某处地面总是倾斜，或者某个弯道的摩擦力总是较低）。

普通的反馈控制（如 PID）只能在误差发生 **后** 进行修正。而 ILC 通过记忆上一圈在同一位置的误差，可以在下一圈到达该位置 **前** 提前施加补偿，从而实现“零误差”跟踪。

### 2.2 空间索引 vs. 时间索引 (Spatial vs. Temporal)
传统的 ILC 通常基于时间 $t$（例如机械臂轨迹）。但在移动机器人导航中，我们采用 **基于空间路径索引 (Spatially Indexed)** 的 ILC。

*   **原因**: 机器人每次经过同一个弯道的速度可能不同，导致时间轴无法对齐。但弯道的几何形状和地面的摩擦特性是与 **位置 (Location)** 绑定的。
*   **实现**: 我们将整条参考路径离散化为 $N$ 个点。ILC 记忆库是一个长度为 $N$ 的查找表 (Lookup Table)，存储了路径上每一个点对应的控制偏置。

## 3. 详细的学习循环 (The Learning Loop)

学习循环负责在机器人运行时在线更新 ILC 记忆库。这是一个将“跟踪误差”转化为“控制偏置”的过程。

### 3.1 误差定义
在路径索引 $s$ 处，我们关注两个关键误差：
1.  **横向误差 ($e_{lat}$)**: 机器人中心偏离参考路径的垂直距离。
2.  **航向误差 ($e_{head}$)**: 机器人车头朝向与路径切线方向的夹角。

### 3.2 学习目标 (Bias Terms)
我们并不直接学习电机的电压或力矩，而是学习 **运动学层面的速度偏置**：
1.  **侧向速度偏置 ($\delta v_y$)**: 用于主动抵消侧滑。如果机器人在某处总是向外滑（$e_{lat}$ 变大），我们需要学习一个向内的 $v_y$ 来“拉”住车身。
2.  **角速度偏置 ($\delta \omega$)**: 用于修正转向特性。如果机器人在某处总是转向不足（$e_{head}$ 滞后），我们需要学习一个额外的 $\omega$ 来辅助转向。

### 3.3 更新律 (Update Law)
在每一个控制周期，系统根据当前位置 $s$ 更新记忆库。更新公式如下：

$$ \mathbf{u}_{bias}^{(k+1)}(s) = \gamma \cdot \mathbf{u}_{bias}^{(k)}(s) + \mathbf{K}_{learning} \cdot \mathbf{e}^{(k)}(s) $$

具体展开为：

$$ \delta v_y(s) \leftarrow \underbrace{\gamma \cdot \delta v_y(s)}_{\text{记忆保持}} + \underbrace{\alpha_{lat} \cdot e_{lat}(s)}_{\text{误差修正}} $$

$$ \delta \omega(s) \leftarrow \underbrace{\gamma \cdot \delta \omega(s)}_{\text{记忆保持}} + \underbrace{\alpha_{head} \cdot e_{head}(s)}_{\text{误差修正}} $$

其中：
*   **$\gamma$ (Decay/Forgetting Factor)**: 遗忘因子，通常取值 $0.95 \sim 0.99$。
    *   *作用*: 处理非重复性扰动。如果环境发生变化（例如路面变干了），旧的经验会逐渐淡忘，防止过拟合旧环境。
*   **$\alpha$ (Learning Rate)**: 学习率。决定了收敛速度。
    *   $\alpha$ 越大，误差消除越快，但可能导致震荡。
    *   $\alpha$ 越小，学习越平滑，但收敛慢。

## 4. 规划循环 (Planning Loop)

在每个控制周期，MPPI 利用 ILC 记忆库来生成控制指令。

1.  **提取先验 (Retrieve Prior)**:
    对于预测时域 $T$ 内的每一步 $t$，根据预测的路径索引，从 ILC 记忆库中提取对应的偏置：
    $$ U_{prior} = \{ u_{bias}(s_0), u_{bias}(s_1), ..., u_{bias}(s_T) \} $$

2.  **注入采样 (Injection into Sampling)**:
    MPPI 生成 $K$ 条随机轨迹。第 $k$ 条轨迹的控制输入序列 $V_k$ 由名义控制 $u_{nom}$、ILC 先验 $u_{prior}$ 和随机噪声 $\epsilon$ 组成：
    $$ v_{t}^{(k)} = u_{nom, t} + u_{prior, t} + \epsilon_{t}^{(k)} $$
    *   $u_{nom}$: 基于纯运动学模型计算的基准指令（无滑移假设）。
    *   $u_{prior}$: 来自 ILC 的经验补偿（包含滑移和动力学修正）。
    *   $\epsilon$: 随机探索噪声。

3.  **最优控制求解**:
    MPPI 根据代价函数对这些轨迹进行加权平均。由于采样分布的均值已经包含了 ILC 的修正，MPPI 实际上是在“正确的解”附近进行微调，从而极大地提高了采样效率和控制精度。

## 5. 总结

`mppi_ilc_prior` 通过将 ILC 嵌入到 MPPI 的优化循环内部，实现了一种 **"Residual Learning" (残差学习)** 框架：
1.  **物理模型** 解决大部分控制问题。
2.  **ILC** 通过“空间记忆”学习并补偿未建模的动力学残差（如复杂的地面摩擦、机械间隙）。
3.  **MPPI** 保证最终轨迹的最优性和安全性。

这种结合方式特别适合重复性高、动力学复杂且对安全性要求高的场景（如赛车、自动巡检机器人）。
