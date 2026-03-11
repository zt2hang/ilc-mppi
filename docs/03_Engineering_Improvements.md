# 第三章　工程改进：CUDA加速与执行层优化

## 1　引言

前两章从理论层面分别阐述了基于位置路径点的迭代学习控制（ILC）以及基于滑移模型的MPPI采样优化框架。然而，将上述算法部署到实际四轮独立转向驱动（FWIDS）机器人上时，存在两个核心工程瓶颈：

1. **计算实时性瓶颈**：MPPI算法需要在每个控制周期内完成 $K$ 条轨迹（典型值 $K = 512 \sim 4096$）、每条轨迹 $T$ 步（典型值 $T = 30 \sim 50$）的前向仿真与代价累积，总计 $K \times T$ 次运动学积分与代价计算。在CPU上即使采用OpenMP并行化，当 $K \geq 2048$ 时仍难以满足 20ms 的控制周期约束。

2. **执行层Sim-to-Real差距**：原始速度驱动节点（vel_driver）仅执行纯粹的逆运动学映射，未考虑物理电机的转速/转角速率限制、静摩擦起步困难、MPPI采样噪声导致的转向抖振以及各轮物理转向角度限幅等实际工程问题。

本章针对上述两个瓶颈，分别从GPU异构计算加速和执行层工程优化两方面展开详细论述。

---

## 2　MPPI采样过程的CUDA加速

### 2.1　GPU加速的动机与总体架构

MPPI算法的核心计算循环具有天然的并行结构：$K$ 条采样轨迹之间互相独立，每条轨迹内部的 $T$ 步前向积分虽然存在时间序列依赖，但不同采样间的积分完全可以并发执行。设单次阶段代价计算的复杂度为 $\mathcal{O}(P)$（其中 $P$ 为参考路径点数），则CPU端的总计算量为

$$
\mathcal{C}_{\text{CPU}} = K \times T \times \mathcal{O}(P)
$$

即使使用 $N_{\text{core}}$ 核CPU并行，计算时间下界为 $\mathcal{C}_{\text{CPU}} / N_{\text{core}}$，对于 $K = 2048,\, T = 40,\, P = 200$ 的典型参数组合，仍然需要约 $10 \sim 15$ ms。

GPU架构通过数千个CUDA核心的大规模并行来消除这一瓶颈。总体设计思路如下：

- **噪声生成内核**：一个CUDA kernel负责在设备端并行生成 $K \times T$ 组三维高斯噪声样本，避免主机-设备间的大规模噪声数据传输。
- **轨迹前滚与代价计算内核**：一个CUDA kernel为每条采样轨迹分配一个线程，在设备端独立完成 $T$ 步运动学积分、阶段代价累积与终端代价计算。
- **异步流水线**：全部设备端计算在非阻塞CUDA流上执行，仅在最终结果就绪时同步。

### 2.2　设备端数据结构设计

为在GPU上高效执行计算，需将CPU端的数据结构转换为适合GPU访问模式的紧凑浮点结构。定义以下设备端结构：

**路径点结构** $\mathbf{p}_i^{\text{GPU}}$：每个参考路径点包含四维浮点向量

$$
\mathbf{p}_i^{\text{GPU}} = (x_i,\, y_i,\, \psi_i,\, \kappa_i), \quad i = 1, \ldots, P
$$

其中 $\kappa_i$ 为该路径点处的局部曲率，通过相邻三点差分估计：

$$
\kappa_i \approx \frac{\Delta\psi_i}{\|\Delta\mathbf{r}_i\|}, \quad \Delta\psi_i = \text{atan2}(\Delta y_{i+1},\, \Delta x_{i+1}) - \text{atan2}(\Delta y_i,\, \Delta x_i)
$$

**车辆参数结构** $\Theta_v^{\text{GPU}}$：包含速度限幅与执行器时间常数

$$
\Theta_v^{\text{GPU}} = \bigl(v_{x,\max},\, v_{y,\max},\, \omega_{\max},\, \tau_{\text{act}}\bigr)
$$

其中执行器时间常数 $\tau_{\text{act}}$ 是工程改进中新增的参数（见第2.4节）。

**代价权重结构** $\mathbf{w}^{\text{GPU}}$：包含所有阶段代价分项的权重

$$
\mathbf{w}^{\text{GPU}} = \bigl(w_d,\, w_\psi,\, w_v,\, w_{\text{slip}},\, w_\kappa,\, w_{\dot\psi},\, w_{\text{term}},\, w_{\Delta u_1},\, w_{\Delta u_2},\, w_{\Delta u_3}\bigr)
$$

**MPPI超参数结构**：

$$
\Theta_{\text{MPPI}}^{\text{GPU}} = \bigl(K,\, T,\, \Delta t,\, \lambda,\, \alpha,\, \boldsymbol{\sigma},\, v_{\text{ref}}\bigr)
$$

### 2.3　噪声生成内核

GPU端噪声生成避免了在主机端生成 $K \times T \times 3$ 个随机数后再传输到设备的开销。每个线程对应一个 $(k, t)$ 索引对，利用cuRAND库的设备端API直接在全局内存中写入噪声：

设总线程数为 $N_{\text{thread}} = K \times T$，线程索引 $\text{idx} = k \cdot T + t$，则第 $\text{idx}$ 个线程生成的噪声为

$$
\boldsymbol{\epsilon}_{\text{idx}} = \bigl(\sigma_{v_x}\, z_1,\; \sigma_{v_y}\, z_2,\; \sigma_\omega\, z_3\bigr), \quad z_i \sim \mathcal{N}(0,1)
$$

其中 $z_i$ 由cuRAND状态机 $\mathcal{S}_{\text{idx}}$ 通过Box-Muller变换产生。每个线程维护独立的cuRAND状态以保证统计独立性。噪声写入全局内存的布局为

$$
\texttt{noise}[\text{idx} \times 3 + j] = \sigma_j \cdot z_{j+1}, \quad j \in \{0, 1, 2\}
$$

cuRAND状态在控制器初始化时通过专用初始化内核一次性设置种子。设块大小 $B = 256$，则需要 $\lceil N_{\text{thread}} / B \rceil$ 个线程块。

### 2.4　执行器延迟模型

工程改进中在GPU端动力学前向积分中引入了一阶执行器延迟模型。令 $\mathbf{u}_t^{\text{cmd}} = (v_x^{\text{cmd}},\, v_y^{\text{cmd}},\, \omega^{\text{cmd}})_t$ 为第 $t$ 步的指令值，$\mathbf{u}_t^{\text{act}}$ 为执行器实际输出值，则延迟动力学为

$$
\mathbf{u}_t^{\text{act}} = \mathbf{u}_{t-1}^{\text{act}} + \frac{\Delta t}{\tau_{\text{act}} + \Delta t}\bigl(\mathbf{u}_t^{\text{cmd}} - \mathbf{u}_{t-1}^{\text{act}}\bigr)
$$

其等效传递函数为一阶惯性环节

$$
G(s) = \frac{1}{\tau_{\text{act}}\, s + 1}
$$

离散化滤波系数为

$$
\alpha_{\text{act}} = \frac{\Delta t}{\tau_{\text{act}} + \Delta t}
$$

当 $\tau_{\text{act}} \to 0$ 时，$\alpha_{\text{act}} \to 1$，退化为理想无延迟情形。该模型使MPPI在采样阶段即考虑执行器惯性，避免生成执行器无法跟踪的激进控制序列。

### 2.5　轨迹前滚与代价计算内核

轨迹前滚内核为每条采样轨迹分配一个独立线程。设线程索引 $k \in \{0, \ldots, K-1\}$，则该线程的计算流程如下。

**初始化**：设当前时刻状态为 $\mathbf{x}_0 = (x_0,\, y_0,\, \psi_0)$，执行器状态初始化为 $(v_x^{\text{act}},\, v_y^{\text{act}},\, \omega^{\text{act}})_0 = (v_{x,0},\, v_{y,0},\, \omega_0)$。

**时间步循环**（$t = 0, \ldots, T-1$）：

1. 读取噪声索引：$\text{noise\_idx} = k \cdot T \cdot 3 + t \cdot 3$

2. 构造含噪控制量并限幅：
$$
\mathbf{u}_t^{(k)} = \text{clamp}\bigl(\bar{\mathbf{u}}_t + \boldsymbol{\epsilon}_{k,t},\; -\mathbf{u}_{\max},\; \mathbf{u}_{\max}\bigr)
$$

3. 执行器延迟滤波（第2.4节公式）

4. 运动学积分（含滑移耦合）：
$$
\begin{cases}
v_{y,\text{eff}} = v_y^{\text{act}} - K_s\, v_x^{\text{act}}\, \omega^{\text{act}} \\[4pt]
x_{t+1} = x_t + \Delta t\bigl(v_x^{\text{act}}\cos\psi_t - v_{y,\text{eff}}\sin\psi_t\bigr) \\[2pt]
y_{t+1} = y_t + \Delta t\bigl(v_x^{\text{act}}\sin\psi_t + v_{y,\text{eff}}\cos\psi_t\bigr) \\[2pt]
\psi_{t+1} = \psi_t + \Delta t\, \omega^{\text{act}}
\end{cases}
$$

5. 阶段代价累积：通过设备端函数计算距离误差、航向误差、速度跟踪误差、滑移风险、曲率限速与控制变化率等代价分项的加权和

6. 信息论控制代价项：
$$
J_{\text{ctrl},t}^{(k)} = \lambda\,(1 - \alpha)\, \bar{\mathbf{u}}_t^\top\, \Sigma^{-1}\, \mathbf{u}_t^{(k)}
$$

**终端代价**：

$$
J_{\text{term}}^{(k)} = \begin{cases}
w_{\text{term}}\, \|\mathbf{x}_T^{(k)} - \mathbf{x}_{\text{goal}}\|^2 & \text{if } \|\mathbf{x}_T^{(k)} - \mathbf{x}_{\text{goal}}\|^2 > 0.25 \\
0 & \text{otherwise}
\end{cases}
$$

最终第 $k$ 条轨迹的总代价写入设备内存：

$$
J^{(k)} = \sum_{t=0}^{T-1}\Bigl[J_{\text{stage},t}^{(k)} + J_{\text{ctrl},t}^{(k)}\Bigr] + J_{\text{term}}^{(k)}
$$

### 2.6　设备端最近邻路径搜索

阶段代价中的路径跟踪误差需要找到参考路径上距当前预测位置最近的点。在GPU端采用线性扫描策略：

$$
i^* = \arg\min_{i \in \{1, \ldots, P\}} \bigl\|(x_t, y_t) - (x_i^{\text{ref}}, y_i^{\text{ref}})\bigr\|^2
$$

由于该搜索在每个线程内独立执行且路径点数 $P$ 一般为数百量级，线性搜索的开销可接受。找到 $i^*$ 后，距离误差和参考航向可直接查表获取。

### 2.7　内存管理与异步传输策略

CUDA加速的关键性能瓶颈之一在于主机-设备间的数据传输延迟（PCIe带宽约 12 GB/s）。本实现采用以下策略最小化传输开销：

**设备端内存分配**：在控制器初始化时一次性分配以下设备内存块：

| 内存块 | 大小 | 用途 |
|:---:|:---:|:---:|
| $\texttt{d\_costs}$ | $K \times 4$ 字节 | 各轨迹总代价 |
| $\texttt{d\_noise}$ | $K \times T \times 3 \times 4$ 字节 | 采样噪声 |
| $\texttt{d\_mean}$ | $T \times 3 \times 4$ 字节 | 均值控制序列 |
| $\texttt{d\_rng}$ | $K \times T \times \text{sizeof(curandState)}$ 字节 | 随机数生成器状态 |
| $\texttt{d\_path}$ | $P_{\max} \times 16$ 字节 | 参考路径（预分配最大容量） |

**锁页内存（Pinned Memory）**：在主机端通过 `cudaMallocHost` 分配锁页缓冲区 $\texttt{h\_costs\_pinned}$、$\texttt{h\_noise\_pinned}$、$\texttt{h\_mean\_pinned}$。锁页内存相比普通分页内存可将DMA传输吞吐提升约 $2\sim 3$ 倍，因为操作系统不会将其换出到磁盘。

**非阻塞CUDA流**：全部内核启动与内存拷贝均提交到同一条非阻塞流 $\mathcal{S}$ 上，流内操作按提交顺序串行执行，但不阻塞CPU主线程。只在最终结果传回完成后执行一次流同步。

**动态缓冲区扩展**：当运行时MPPI参数（$K$ 或 $T$）变化时，通过容量检测自动释放并重新分配锁页缓冲区，避免越界写入。

### 2.8　计算流水线与性能剖析

完整的单次MPPI-CUDA求解流水线由以下五个阶段组成：

$$
\underbrace{\text{H2D}(\bar{\mathbf{u}})}_{\text{阶段1}} \;\to\; \underbrace{\text{NoiseKernel}}_{\text{阶段2}} \;\to\; \underbrace{\text{RolloutKernel}}_{\text{阶段3}} \;\to\; \underbrace{\text{D2H}(\mathbf{J})}_{\text{阶段4}} \;\to\; \underbrace{\text{D2H}(\boldsymbol{\epsilon})}_{\text{阶段5}}
$$

其中：
- **阶段1**（H2D均值序列）：将CPU端的 $T \times 3$ 均值控制序列通过异步拷贝传输到设备端，数据量 $T \times 3 \times 4$ 字节，典型时间 $< 0.01$ ms。
- **阶段2**（噪声生成内核）：$\lceil K \cdot T / 256 \rceil$ 个线程块并行生成 $K \times T \times 3$ 个高斯随机数。
- **阶段3**（轨迹前滚内核）：$\lceil K / 256 \rceil$ 个线程块，每个线程执行 $T$ 步前向积分与代价累积，是计算最密集阶段。
- **阶段4**（D2H代价传回）：$K \times 4$ 字节的代价向量异步传回主机端锁页缓冲区。
- **阶段5**（D2H噪声传回）：$K \times T \times 3 \times 4$ 字节的噪声矩阵传回，用于CPU端权重加权更新。

系统通过CUDA事件（`cudaEvent`）在每个阶段边界插入计时标记，精确测量各阶段耗时，输出形如

$$
\text{total} = t_{\text{H2D}} + t_{\text{noise}} + t_{\text{rollout}} + t_{\text{D2H,cost}} + t_{\text{D2H,noise}}
$$

的细粒度性能报告，支持运行时性能瓶颈诊断。

### 2.9　CPU-GPU混合求解架构

CUDA求解器并非完全替代CPU端的MPPI流程，而是替代其中计算最密集的三个步骤（噪声生成、轨迹前滚、代价计算），将结果传回CPU后继续执行权重计算与控制序列更新。完整的混合架构为：

$$
\boxed{\text{GPU}: \text{噪声生成} \to \text{轨迹前滚} \to \text{代价计算}} \;\xrightarrow{\text{D2H}}\; \boxed{\text{CPU}: \text{权重计算} \to \text{加权更新} \to \text{滤波} \to \text{补偿}}
$$

这种设计的理由如下：

1. 权重计算（softmax归一化）与加权更新（向量加权平均）的计算量为 $\mathcal{O}(K \cdot T)$，相比轨迹前滚的 $\mathcal{O}(K \cdot T \cdot P)$ 小一个量级，在CPU上即可快速完成。
2. Savitzky-Golay滤波和闭环滑移补偿依赖历史状态与实时传感器反馈，保留在CPU端更便于维护。
3. 通过pimpl模式（指针到实现）将CUDA求解器封装为可选组件，当GPU初始化失败时自动回退到CPU+OpenMP路径，保证系统的鲁棒性。

---

## 3　执行层优化：四轮独立转向驱动指令生成

### 3.1　问题分析

原始vel_driver节点接收整车级别的机体速度指令 $(v_x,\, v_y,\, \omega)$ 后，通过确定性逆运动学直接计算四个轮子的转向角和转速并发布，存在以下工程缺陷：

- **无速率限制**：指令跳变直接发送给电机，导致电流冲击和机械振动。
- **无死区滤波**：MPPI采样噪声中的微小横向/角速度分量导致四轮持续微幅转向抖振。
- **无起步过渡**：从静止起步时，控制量从零阶跃到目标值，静摩擦导致轮子卡滞。
- **无物理限位**：每个轮子的机械结构限制了转向角度范围，超限指令将导致堵转。
- **转向-驱动耦合**：当转向尚未到达目标角度时仍以全速驱动，车轮沿错误方向拖行地面。

### 3.2　逆运动学修正

原始逆运动学存在坐标约定错误。令机器人底盘中心为原点，前轴距 $l_f$，后轴距 $l_r$，左横向距 $d_l$，右横向距 $d_r$，在URDF/ROS标准坐标系（$X$ 轴向前，$Y$ 轴向左）下，四轮的安装位置向量为

$$
\mathbf{r}_{\text{FL}} = (l_f,\, d_l), \quad \mathbf{r}_{\text{FR}} = (l_f,\, -d_r), \quad \mathbf{r}_{\text{RL}} = (-l_r,\, d_l), \quad \mathbf{r}_{\text{RR}} = (-l_r,\, -d_r)
$$

由刚体运动学 $\mathbf{v}_{\text{wheel}} = \mathbf{v}_c + \boldsymbol{\omega} \times \mathbf{r}$（其中 $\boldsymbol{\omega} = \omega\,\hat{z}$），各轮在车体坐标系下的速度分量为

$$
\begin{pmatrix} v_{x,i} \\ v_{y,i} \end{pmatrix} = \begin{pmatrix} v_x - \omega\, r_{y,i} \\ v_y + \omega\, r_{x,i} \end{pmatrix}
$$

以前左轮为例：

$$
v_{x,\text{FL}} = v_x - \omega\, d_l, \quad v_{y,\text{FL}} = v_y + \omega\, l_f
$$

转向角和转速的计算为

$$
\delta_i = \text{atan2}(v_{y,i},\, v_{x,i}), \quad \Omega_i = \frac{\sqrt{v_{x,i}^2 + v_{y,i}^2}}{r_{\text{tire}}}
$$

原始实现中存在坐标轴映射错误（将 `linear.y` 取反映射为"车辆右向速度"），改进版本统一为ROS标准约定，消除了此歧义。

### 3.3　指令级速率限制

对整车级速度指令和轮级执行指令分别施加速率限制。令 $\Delta t$ 为相邻指令的时间间隔，$\dot{u}_{\max}$ 为对应的最大变化率，速率限制函数定义为

$$
\text{RateLimit}(u_{t-1},\, u_t^{\text{target}},\, \dot{u}_{\max},\, \Delta t) = \begin{cases}
u_t^{\text{target}} & \text{if } |u_t^{\text{target}} - u_{t-1}| \leq \dot{u}_{\max}\, \Delta t \\
u_{t-1} + \text{sign}(u_t^{\text{target}} - u_{t-1})\, \dot{u}_{\max}\, \Delta t & \text{otherwise}
\end{cases}
$$

该函数应用于以下四个层级：

| 层级 | 限制对象 | 最大变化率参数 | 典型值 |
|:---:|:---:|:---:|:---:|
| 整车线速度 | $v_x,\, v_y$ | $a_{\max}^{\text{vel}}$ | 1.0 m/s² |
| 整车角速度 | $\omega$ | $\dot{\omega}_{\max}^{\text{cmd}}$ | 2.0 rad/s² |
| 转向角 | $\delta_i$ | $\dot{\delta}_{\max}$ | 1.5 rad/s |
| 转子角速度 | $\Omega_i$ | $\dot{\Omega}_{\max}$ | 5.0 rad/s² |

### 3.4　低通滤波与死区抑制

**低通滤波**：在速率限制之前，对整车速度指令施加一阶指数移动平均滤波

$$
u_t^{\text{filt}} = u_{t-1}^{\text{filt}} + \alpha_s\,(u_t^{\text{raw}} - u_{t-1}^{\text{filt}})
$$

其中 $\alpha_s \in (0, 1]$ 为平滑系数。$\alpha_s$ 越小滤波越强但响应越慢。典型值 $\alpha_s = 0.3$。

**死区抑制**：对横向速度 $v_y$ 和角速度 $\omega$ 施加死区处理

$$
\text{Deadband}(u,\, \epsilon) = \begin{cases}
0 & \text{if } |u| < \epsilon \\
u & \text{otherwise}
\end{cases}
$$

同时对转向角变化量施加死区：若 $|\delta_t^{\text{new}} - \delta_{t-1}| < \epsilon_\delta$，则保持 $\delta_t = \delta_{t-1}$。这有效抑制MPPI采样噪声在直行场景中引起的四轮微幅转向抖振。

### 3.5　起步斜坡策略

从静止状态起步时，静摩擦力远大于动摩擦力，若以正常加速度限制缓慢增大驱动力矩，车轮可能长时间停留在静摩擦区间导致卡滞。本文引入起步斜坡（Startup Ramp）策略：

定义状态检测函数

$$
\text{NearZero}(\mathbf{v}) = \bigl(|v_x| < \epsilon_0\bigr) \land \bigl(|v_y| < \epsilon_0\bigr) \land \bigl(|\omega| < \epsilon_0\bigr)
$$

当检测到从静止状态($\text{NearZero}(\mathbf{v}_{t-1}) = \text{true}$)到运动状态($\text{NearZero}(\mathbf{v}_t^{\text{target}}) = \text{false}$)的切换时，在前 $N_{\text{ramp}}$ 步（典型值 10 步）内将加速度限制乘以放大系数 $\gamma_{\text{startup}}$（典型值 3.0）：

$$
\dot{u}_{\max}^{\text{eff}} = \gamma_{\text{startup}}\, \dot{u}_{\max}, \quad \text{for } n_{\text{cmd}} \leq N_{\text{ramp}}
$$

同时相应调大低通滤波系数

$$
\alpha_s^{\text{eff}} = \min\bigl(1,\; \alpha_s \cdot \gamma_{\text{startup}}\bigr)
$$

以加快滤波器对突变指令的响应。这一策略等效于在起步瞬间提供更大的力矩裕度以克服静摩擦。

### 3.6　物理转向限位与翻转优化

FWIDS机器人的各轮转向机构受物理结构限制，不能实现 $[-\pi,\, \pi]$ 全范围旋转。实际测量的各轮限位为

| 轮位 | 最小角（CW极限） | 最大角（CCW极限） |
|:---:|:---:|:---:|
| FL | $-50°$ | $+30°$ |
| FR | $-30°$ | $+50°$ |
| RL | $-30°$ | $+50°$ |
| RR | $-50°$ | $+30°$ |

注意各轮限位是非对称的，这由舵机安装偏置和底盘干涉决定。

当逆运动学计算得到的目标转向角 $\delta^*$ 超出限位 $[\delta_{\min},\, \delta_{\max}]$ 时，采用**翻转优化**（Flip Optimization）策略：

**直接方案**：$\delta_A = \delta^*$，转速 $\Omega_A = +\Omega$

**翻转方案**：$\delta_B = \delta^* + \pi$（归一化到 $(-\pi, \pi]$），转速 $\Omega_B = -\Omega$

两种方案产生相同的轮面速度向量。选择逻辑如下：

1. 若 $\delta_A \in [\delta_{\min},\, \delta_{\max}]$，选直接方案
2. 否则若 $\delta_B \in [\delta_{\min},\, \delta_{\max}]$，选翻转方案
3. 若两者均越限，进入智能钳位模式

**智能钳位**：当直接与翻转方案均越限时，分别将两者钳位到最近边界

$$
\delta_A^{\text{clamp}} = \text{clamp}(\delta_A,\, \delta_{\min},\, \delta_{\max}), \quad \delta_B^{\text{clamp}} = \text{clamp}(\delta_B,\, \delta_{\min},\, \delta_{\max})
$$

计算各方案钳位后的力矢量方向误差

$$
e_A = \bigl|\angle(\delta_A^{\text{clamp}},\, \delta^*)\bigr|, \quad e_B = \bigl|\angle(\delta_B^{\text{clamp}} + \pi,\, \delta^*)\bigr|
$$

选择误差更小的方案，并按误差余弦衰减转速以抑制侧向拖行力：

$$
\Omega_{\text{out}} = \pm\,\Omega \cdot \max\bigl(0,\, \cos(e)\bigr)
$$

其物理含义为：当轮面方向与期望方向偏差增大时，降低驱动转速，在 $90°$ 偏差处完全停转，避免产生垂直于期望方向的地面反力。

### 3.7　转向-驱动耦合衰减

由于转向角速率受限，在指令突变时转向角需要若干控制周期才能到达目标值。在转向过渡期间，若以全速驱动车轮，车轮将沿当前（而非目标）方向推动机器人，产生拖行效应。

本文引入转向-驱动耦合衰减机制：令第 $i$ 号轮的当前转向角为 $\delta_i^{\text{curr}}$，目标转向角为 $\delta_i^{\text{target}}$，则驱动转速乘以耦合衰减系数

$$
\eta_i = \max\bigl(0,\; \cos(\delta_i^{\text{curr}} - \delta_i^{\text{target}})\bigr)
$$

$$
\Omega_i^{\text{out}} = \eta_i \cdot \Omega_i^{\text{cmd}}
$$

当 $\delta_i^{\text{curr}} = \delta_i^{\text{target}}$ 时 $\eta_i = 1$（不衰减）；当转向偏差达到 $\pm 90°$ 时 $\eta_i = 0$（完全停转）。该机制确保车轮仅在转向基本对齐后才施加驱动力。

在时序上，耦合衰减在转向速率限制之后、转速速率限制之前执行：

$$
\delta_i^{\text{rate-lim}} \;\to\; \eta_i = \cos(\delta_i^{\text{rate-lim}} - \delta_i^{\text{target}}) \;\to\; \Omega_i^{\text{attenuated}} = \eta_i\, \Omega_i \;\to\; \Omega_i^{\text{rate-lim}}
$$

---

## 4　CUDA加速与ILC先验的协同

CUDA加速不仅服务于基础MPPI控制器，也与ILC先验机制协同工作。在第一章中，ILC提供的先验控制序列 $\bar{\mathbf{u}}_t^{\text{prior}}$ 被注入到MPPI采样分布的均值中。当使用CUDA求解器时，这一注入在CPU端完成后传输到GPU：

$$
\bar{\mathbf{u}}_t^{\text{GPU}} = \bar{\mathbf{u}}_t^{\text{opt}} + \bar{\mathbf{u}}_t^{\text{prior}}
$$

GPU端生成的噪声 $\boldsymbol{\epsilon}_{k,t}$ 叠加在此已偏移的均值上，从而在设备端自然实现了ILC先验引导的采样偏置。GPU端计算完成后，代价向量 $\{J^{(k)}\}_{k=1}^K$ 和噪声矩阵 $\{\boldsymbol{\epsilon}_{k,t}\}$ 传回CPU端，后续的softmax权重计算和控制序列更新仍在CPU端执行。

参考路径的GPU端传输通过 `setReferencePath` 方法实现，该方法在路径更新时（通常频率远低于控制频率）将 $P$ 个路径点批量传输到设备端，避免了每个控制周期的路径传输开销。

---

## 5　完整系统数据流

综合前三章内容，MPPI_ILC_PRIOR系统从感知输入到执行器指令的完整数据流如下：

$$
\underbrace{\text{里程计} \to \text{ILC位置查询} \to \text{先验注入}}_{\text{第一章}} \;\to\; \underbrace{\text{GPU采样/前滚/代价}}_{\text{本章\S2}} \;\to\; \underbrace{\text{CPU权重/更新}}_{\text{第二章}}
$$

$$
\to\; \underbrace{\text{SG滤波} \to \text{滑移补偿}}_{\text{第二章}} \;\to\; \underbrace{\text{逆运动学} \to \text{速率限制/耦合衰减}}_{\text{本章\S3}} \;\to\; \underbrace{8\text{D执行器}}_{\text{Gazebo}}
$$

其中ILC记忆更新（学习律）在每圈/每段路径完成后执行，更新频率远低于控制频率。

---

## 6　小结

本章从两个维度对MPPI_ILC_PRIOR系统进行了工程优化：

1. **CUDA加速**：将MPPI算法中计算最密集的噪声生成、轨迹前滚和代价计算三步卸载到GPU执行，通过锁页内存与异步流最小化传输延迟，同时引入执行器延迟模型提高仿真保真度。CPU-GPU混合架构通过pimpl模式保持模块化和故障回退能力。

2. **执行层优化**：在逆运动学映射之后、执行器指令发布之前，插入多层信号处理管线——低通滤波、死区抑制、速率限制、起步斜坡、物理限位翻转优化和转向-驱动耦合衰减——以弥合仿真环境与真实硬件之间的差距。

两项优化相互正交且互补：CUDA加速使得在相同控制周期内可以使用更大的采样数 $K$，提高最优解质量；执行层优化确保更高质量的最优解能被物理执行器忠实地执行，最终体现为闭环跟踪性能的提升。
