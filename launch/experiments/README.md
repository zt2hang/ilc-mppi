# 实验说明文档

本目录包含论文第三章（基于迭代学习控制先验的 MPPI 控制器）的全部仿真实验配置。

---

## 目录结构

```
launch/experiments/
├── README.md                     ← 本文件
├── config/                       ← YAML 参数配置
│   ├── ilc_enabled.yaml          # ILC 开启（基准配置）
│   ├── ilc_disabled.yaml         # ILC 关闭（消融基准）
│   ├── decay_0980.yaml           # γ = 0.980
│   ├── decay_0990.yaml           # γ = 0.990
│   ├── decay_0999.yaml           # γ = 0.999
│   ├── decay_1000.yaml           # γ = 1.000（无遗忘）
│   ├── samples_*_ilc_{on,off}.yaml       # K∈{32..4000}, ILC 开/关（标准模式）
│   ├── isolated_*_ilc_{on,off}.yaml      # K∈{32..2000}, 隔离模式（estimator frozen + feedback off）
│   ├── isolated_*_ilc_aggressive.yaml    # K∈{32..2000}, 隔离 + 激进ILC参数
│   ├── disturbance_ilc_{on,off}.yaml     # 空间扰动 + ILC 开/关
│   ├── disturbance_ilc_aggressive.yaml   # 空间扰动 + 激进ILC参数
│   └── disturbance_full_stack.yaml       # 空间扰动 + 全栈补偿（无ILC）
│
├── exp1_convergence.launch       # 实验 1 launch
├── exp2_ablation_ilc_on.launch   # 实验 2 launch (ILC ON)
├── exp2_ablation_ilc_off.launch  # 实验 2 launch (ILC OFF)
├── exp3_decay_sweep.launch       # 实验 3 launch (参数化)
├── exp4_generalization.launch    # 实验 4 launch (参数化)
├── exp5_sample_budget.launch     # 实验 5 launch (参数化)
├── exp6_sample_efficiency.launch # 实验 6 launch (参数化)
├── exp7_spatial_disturbance.launch # 实验 7 launch (参数化)
│
├── run_exp1_convergence.sh       # 实验 1 运行脚本
├── run_exp2_ablation.sh          # 实验 2 运行脚本
├── run_exp3_decay_sweep.sh       # 实验 3 运行脚本
├── run_exp4_generalization.sh    # 实验 4 运行脚本
├── run_exp5_sample_budget.sh     # 实验 5 运行脚本
├── run_exp6_sample_efficiency.sh  # 实验 6 运行脚本
├── run_exp7_spatial_disturbance.sh # 实验 7 运行脚本
│
├── plot_exp1_convergence.py      # 实验 1 画图
├── plot_exp2_ablation.py         # 实验 2 画图
├── plot_exp3_decay_sweep.py      # 实验 3 画图
├── plot_exp4_generalization.py   # 实验 4 画图
├── plot_exp5_sample_budget.py    # 实验 5 画图
├── plot_exp6_sample_efficiency.py # 实验 6 画图
├── plot_exp7_spatial_disturbance.py # 实验 7 画图
└── plot_font_utils.py            # 绘图公共字体工具
```

---

## 快速开始

```bash
cd ~/planner_code/mppi_swerve_drive_ros_ex

# 每个实验独立运行，互不依赖
bash launch/experiments/run_exp1_convergence.sh
bash launch/experiments/run_exp2_ablation.sh
bash launch/experiments/run_exp3_decay_sweep.sh
bash launch/experiments/run_exp4_generalization.sh
bash launch/experiments/run_exp5_sample_budget.sh
bash launch/experiments/run_exp6_sample_efficiency.sh
bash launch/experiments/run_exp7_spatial_disturbance.sh

# 所有脚本支持 --dry-run 预览（不实际启动仿真）
bash launch/experiments/run_exp1_convergence.sh --dry-run
```

> **提示**：运行中随时按 `Ctrl+C` 可安全中断，脚本会自动关闭 roslaunch 并保存已有日志。

---

## 实验详情

### 实验 1：ILC 收敛性验证

| 项目 | 值 |
|:---|:---|
| **脚本** | `run_exp1_convergence.sh` |
| **赛道** | 圆形闭合路径，radius = 10 m |
| **摩擦系数** | μ = 0.3 |
| **控制器** | mppi_ilc_prior，ILC 开启 (γ = 0.995) |
| **默认圈数** | 15 圈 |
| **预计时长** | ~10 分钟 |
| **对应章节** | §4–§5 |

**目标**：验证 ILC 偏置在多圈运行后是否收敛。

**观测指标**：
- 横向 RMSE 随圈数的下降曲线
- ILC 偏置 RMS 随圈数的变化
- 偏置饱和触发频率

**命令行参数**：
```bash
bash run_exp1_convergence.sh --laps 20   # 增加到 20 圈
```

**预期结果**：横向 RMSE 在前 3–5 圈内快速下降，之后趋于稳态；偏置 RMS 同步上升后饱和。

---

### 实验 2：ILC 消融实验

| 项目 | 值 |
|:---|:---|
| **脚本** | `run_exp2_ablation.sh` |
| **赛道** | 跑道形 (straight = 20 m, radius = 3 m) |
| **摩擦系数** | μ = 0.10（低摩擦） |
| **对比组** | ILC ON vs ILC OFF（same mppi_ilc_prior node） |
| **默认圈数** | 每组 10 圈 |
| **预计时长** | ~18 分钟（两组） |
| **对应章节** | §5.6 |

**目标**：量化 ILC 先验注入带来的跟踪精度提升。

**观测指标**：
- 横向 RMSE 对比（分直道/弯道）
- 航向 RMSE 对比
- ESS (有效采样尺寸) 对比

**命令行参数**：
```bash
bash run_exp2_ablation.sh --only on    # 仅运行 ILC ON
bash run_exp2_ablation.sh --only off   # 仅运行 ILC OFF
bash run_exp2_ablation.sh --laps 15    # 每组 15 圈
```

**预期结果**：ILC ON 在第 5 圈后的弯道横向 RMSE 比 ILC OFF 降低 30–50%。

---

### 实验 3：遗忘因子 γ 参数扫描

| 项目 | 值 |
|:---|:---|
| **脚本** | `run_exp3_decay_sweep.sh` |
| **赛道** | 圆形 (radius = 10 m) |
| **摩擦系数** | μ = 0.3 |
| **扫描变量** | γ ∈ {0.980, 0.990, 0.995, 0.999, 1.000} |
| **默认圈数** | 每组 15 圈 |
| **预计时长** | ~50 分钟（5 组） |
| **对应章节** | §4.5 |

**目标**：验证遗忘因子对收敛速度、稳态误差、鲁棒性的影响。

**观测指标**：
- 各 γ 值的收敛速度（达到目标 RMSE 的圈数）
- 稳态残余误差 e∞ ∝ (1−γ)/(αG)
- 偏置 RMS 饱和水平

**命令行参数**：
```bash
bash run_exp3_decay_sweep.sh --values 0980,0999      # 只跑 2 个值
bash run_exp3_decay_sweep.sh --laps 20                # 每组 20 圈
```

**预期结果**：
- γ = 0.980：收敛最快（~3 圈），但稳态残余误差最大
- γ = 1.000：稳态误差趋零，但收敛最慢且对环境变化敏感
- γ = 0.995（默认）：在两者之间取得平衡

---

### 实验 4：多赛道泛化验证

| 项目 | 值 |
|:---|:---|
| **脚本** | `run_exp4_generalization.sh` |
| **赛道** | circular, racetrack, figure8, square |
| **摩擦系数** | circular: μ=0.3，其余: μ=0.10 |
| **控制器** | mppi_ilc_prior，ILC 开启 |
| **默认圈数** | 每赛道 8 圈 |
| **预计时长** | ~30 分钟（4 组） |
| **对应章节** | §4.3（闭合路径处理） |

**目标**：验证 ILC 在不同路径几何上的适用性。

**四种赛道特征**：

| 赛道 | 曲率特征 | 重点验证 |
|:---|:---|:---|
| circular | 恒定曲率 | ILC 基准收敛 |
| racetrack | 直道 + 弯道混合 | 直/弯分段精度 |
| figure8 | 正/反弯道交替 | 自交叉点索引鲁棒性 |
| square | 急转弯（κ 很大） | 曲率门控机制有效性 |

**命令行参数**：
```bash
bash run_exp4_generalization.sh --scenarios circular,figure8   # 只跑 2 种
bash run_exp4_generalization.sh --laps 12                       # 每赛道 12 圈
```

---

### 实验 5：采样数预算分析

| 项目 | 值 |
|:---|:---|
| **脚本** | `run_exp5_sample_budget.sh` |
| **赛道** | racetrack (straight = 20 m, radius = 3 m) |
| **摩擦系数** | μ = 0.10 |
| **扫描变量** | K ∈ {256, 512, 1000, 2000, 4000} × {ILC ON, OFF} |
| **默认圈数** | 每组 10 圈 |
| **预计时长** | ~90 分钟（10 组） |
| **对应章节** | §5.5（采样效率） |

**目标**：验证 ILC 先验可作为计算资源受限时的性能补偿手段。

**核心问题**：K=256 + ILC ON 能否达到 K=2000 + ILC OFF 的精度？

**观测指标**：
- 各 K 值下 ILC ON vs OFF 的横向 RMSE
- ESS / K（有效采样比例）
- 控制频率（K 越大，每周期计算时间越长）

**命令行参数**：
```bash
bash run_exp5_sample_budget.sh --K 256,2000            # 只对比 2 个 K 值
bash run_exp5_sample_budget.sh --ilc on                 # 只跑 ILC ON
bash run_exp5_sample_budget.sh --K 256,512 --ilc on     # 组合过滤
```

**预期结果**：ILC ON + K=256 的第 10 圈 RMSE 接近甚至优于 ILC OFF + K=2000，证明 ILC 先验等效于 ~8× 的采样数放大。

---

### 实验 6：隔离采样效率分析

| 项目 | 值 |
|:---|:---|
| **脚本** | `run_exp6_sample_efficiency.sh` |
| **赛道** | racetrack (straight = 20 m, radius = 3 m) |
| **摩擦系数** | μ = 0.10 |
| **扫描变量** | K ∈ {32, 64, 128, 256, 512, 1000, 2000} × {ILC off, ILC on, ILC aggressive} |
| **隔离模式** | SlipEstimator 冻结 + 反馈补偿关闭 |
| **默认圈数** | 每组 15 圈 |
| **预计时长** | ~270 分钟（21 组） |
| **对应章节** | §5.5（采样效率 — 隔离 ILC 贡献） |

**目标**：在关闭在线估计和反馈补偿的条件下，纯隔离地验证 ILC 先验对 MPPI 采样效率的提升效果。

**核心论证**：ILC + K=64 ≈ 纯 MPPI + K=1000（ILC 先验等效于一个数量级的采样数放大）。

**与实验 5 的区别**：
- 实验 5 使用标准三层补偿栈（ILC + SlipEstimator + 反馈），ILC 效果被其他层掩盖
- 实验 6 冻结 SlipEstimator、关闭反馈补偿，纯隔离 ILC 贡献
- 实验 6 扩展到超低采样数 K=32/64/128，在此区间 MPPI 独立运行非常吃力
- 实验 6 增加 aggressive 参数组（k_lat=0.4, k_head=0.15, γ=0.999）

**观测指标**：
- 各 K 值下 off / on / aggressive 的横向 RMSE
- 收敛曲线：低 K 值下 ILC 的收敛速度优势
- "等效采样数"：ILC+K=N 达到纯 MPPI+K=? 的精度

**命令行参数**：
```bash
bash run_exp6_sample_efficiency.sh                        # 全部 21 组
bash run_exp6_sample_efficiency.sh --K 32,64,128          # 只跑低采样组
bash run_exp6_sample_efficiency.sh --ilc aggressive       # 只跑 aggressive
bash run_exp6_sample_efficiency.sh --laps 20              # 每组 20 圈
bash run_exp6_sample_efficiency.sh --dry-run              # 预览不运行
```

**预期结果**：
- K=32 ILC off 的 RMSE >> 0.05 m，几乎无法稳定跟踪
- K=64 ILC aggressive 的 RMSE 接近 K=1000 ILC off
- ILC on/aggressive 在所有 K 值下显著优于 ILC off，且 K 越小差距越大

---

### 实验 7：空间扰动补偿

| 项目 | 值 |
|:---|:---|
| **脚本** | `run_exp7_spatial_disturbance.sh` |
| **赛道** | racetrack (straight = 20 m, radius = 3 m) |
| **摩擦系数** | μ = 0.10 |
| **扰动** | d_vy(s) = A·sin(2π·n·s/L)，空间正弦侧向力 |
| **默认参数** | A = 0.10 m/s, n = 3 cycles/lap |
| **对比组** | ILC off / ILC on / ILC aggressive / Full stack（无ILC但开启估计+反馈） |
| **默认圈数** | 每组 20 圈 |
| **预计时长** | ~70 分钟（4 组） |
| **对应章节** | §5.7（空间先验的独特价值） |

**目标**：验证 ILC 的空间记忆能力——对于在路径上按固定空间分布重复出现的干扰，ILC 先验可以学习并补偿，而全局在线估计器和反馈控制器无法有效应对这类空间变化的扰动。

**扰动机制**：
- 在 MPPI 求解完成后、发布指令前，对 `cmd_vy` 叠加空间正弦扰动
- 扰动幅度和频率完全由路径进度 `s` 决定（空间域而非时间域）
- 每圈经过同一位置时扰动相同，ILC 可学习；但全局估计器只能感知平均效应

**四组对比**：

| 组别 | ILC | SlipEstimator | 反馈补偿 | 验证目的 |
|:---|:---|:---|:---|:---|
| ilc_off | 关 | 冻结 | 关 | 纯 MPPI 基线 |
| ilc_on | 开（标准） | 冻结 | 关 | ILC 标准学习效果 |
| ilc_aggressive | 开（激进） | 冻结 | 关 | ILC 快速学习效果 |
| full_stack | 关 | 开启 | 开启 | 全局补偿栈能否应对空间扰动 |

**命令行参数**：
```bash
bash run_exp7_spatial_disturbance.sh                           # 全部 4 组
bash run_exp7_spatial_disturbance.sh --amp 0.15 --wave 4       # 更强扰动
bash run_exp7_spatial_disturbance.sh --only ilc_aggressive     # 只跑一组
bash run_exp7_spatial_disturbance.sh --laps 25                 # 每组 25 圈
bash run_exp7_spatial_disturbance.sh --dry-run                 # 预览不运行
```

**预期结果**：
- ilc_off：RMSE 在所有圈次保持较高，无改善
- ilc_on/aggressive：RMSE 在前 3-5 圈快速下降（ILC 学到空间扰动模式）
- full_stack：RMSE 略低于 ilc_off（反馈可部分补偿），但远不如 ILC 组
- ILC aggressive 的偏置 RMS 显著高于标准组，证明学习增益的效果

---

## 日志输出与数据文件

所有实验日志输出到 `~/log/experiments/`，按 `<实验名>_<时间戳>/` 组织：

```
~/log/experiments/
├── exp1_convergence_20260311_183000/
│   ├── roslaunch_stdout.log                                    ← roslaunch 完整输出
│   └── exp1_convergence/                                       ← run_id 子目录
│       ├── ilc_lap_metrics__exp1_convergence.csv               ← 每圈汇总 (17列)
│       ├── mppi_eval__exp1_convergence__mppi_ilc_prior_eval.csv ← 每周期 (18列)
│       └── meta__exp1_convergence.txt                          ← 元数据
├── exp2_ablation_ilc_on_20260311_184500/
├── exp2_ablation_ilc_off_20260311_185400/
├── exp3_decay_0980_20260311_190000/
│   ...
```

### CSV 数据格式

**每圈汇总 CSV** (`ilc_lap_metrics__*.csv`) — 由 `mppi_ilc_prior` 节点在每圈结束时写入：

| 列名 | 说明 |
|:---|:---|
| `t` | 时间戳 (s) |
| `lap` | 圈次编号 |
| `path_length` | 路径总长 (m) |
| `lat_rmse` | 横向 RMSE (m) |
| `lat_rmse_straight` | 直道横向 RMSE (m) |
| `lat_rmse_corner` | 弯道横向 RMSE (m) |
| `head_rmse_deg` | 航向 RMSE (°) |
| `samples_total` | MPPI 采样数 |
| `bias_rms_vy` | 侧向偏置 RMS |
| `bias_max_abs_vy` | 侧向偏置最大绝对值 |
| `bias_rms_omega` | 角速度偏置 RMS |
| `bias_max_abs_omega` | 角速度偏置最大绝对值 |
| `ilc_updates` | 该圈 ILC 更新次数 |
| `delta_vy_rms` / `delta_omega_rms` | ILC 增量 RMS |
| `sat_vy` / `sat_omega` | 饱和触发次数 |

**每周期 CSV** (`mppi_eval__*__mppi_ilc_prior_eval.csv`) — 由 `mppi_eval_logger` 节点按控制频率写入：

| 列名 | 说明 |
|:---|:---|
| `t` | 时间戳 (s) |
| `state_cost` | 状态代价 (≈ k_slip) |
| `global_x`, `global_y`, `global_yaw` | 实际位姿 |
| `cmd_vx`, `cmd_vy`, `cmd_yawrate` | 速度指令 |
| `cmd_steer_fl/fr/rl/rr` | 转向角指令 |
| `cmd_rotor_fl/fr/rl/rr` | 驱动轮速指令 |
| `calc_time_ms` | MPPI 求解时间 (ms) |
| `goal_reached` | 是否到达目标 |

---

## 画图脚本

每个实验配有对应的 Python 画图脚本，实验完成后脚本会自动输出画图命令：

| 实验 | 画图脚本 | 用法 |
|:---|:---|:---|
| 实验 1 | `plot_exp1_convergence.py` | `python3 plot_exp1_convergence.py <LOG_DIR>` |
| 实验 2 | `plot_exp2_ablation.py` | `python3 plot_exp2_ablation.py <DIR_ON> <DIR_OFF>` |
| 实验 3 | `plot_exp3_decay_sweep.py` | `python3 plot_exp3_decay_sweep.py <DIR_γ1> <DIR_γ2> ...` |
| 实验 4 | `plot_exp4_generalization.py` | `python3 plot_exp4_generalization.py <DIR_circular> <DIR_racetrack> ...` |
| 实验 5 | `plot_exp5_sample_budget.py` | `python3 plot_exp5_sample_budget.py <DIR_K256_on> <DIR_K256_off> ...` |

**依赖**：`pip install matplotlib numpy pandas`

**输出**：每个脚本在日志父目录生成 PDF + PNG 图文件。

### 各脚本生成的图

| 脚本 | 输出图 | 内容 |
|:---|:---|:---|
| plot_exp1 | `exp1_convergence.*` | 3 子图：横向 RMSE、偏置 RMS、航向 RMSE 随圈数变化 |
| plot_exp2 | `exp2_ablation_curves.*` | ILC ON/OFF 逐圈 RMSE 曲线对比 |
| | `exp2_ablation_bar.*` | 稳态指标柱状图 |
| | `exp2_ablation_traj.*` | 实际轨迹 XY 对比 |
| plot_exp3 | `exp3_decay_sweep_curves.*` | 各 γ 值的 RMSE + 偏置收敛曲线 |
| | `exp3_decay_sweep_bar.*` | 稳态 RMSE 柱状图 |
| plot_exp4 | `exp4_generalization_traj.*` | 各赛道轨迹对比 (多面板) |
| | `exp4_generalization_curves.*` | 各赛道收敛曲线 |
| | `exp4_generalization_bar.*` | 稳态精度柱状图 |
| plot_exp5 | `exp5_sample_budget_bar.*` | K × ILC ON/OFF 分组柱状图 |
| | `exp5_sample_budget_calc_time.*` | 计算时间对比 |
| | `exp5_sample_budget_curves.*` | 各配置收敛曲线 |
| plot_exp6 | `exp6_rmse_vs_K.*` | RMSE vs K 曲线（对数横轴，off/on/aggressive 三线） |
| | `exp6_convergence.*` | 代表性 K 值的逐圈收敛曲线 |
| | `exp6_bar.*` | 稳态 RMSE 分组柱状图 |
| | `exp6_equivalent_K.*` | 等效采样数表格 |
| plot_exp7 | `exp7_convergence.*` | 四组模式的逐圈 RMSE 收敛曲线 |
| | `exp7_bias_growth.*` | ILC 偏置 RMS 随圈数增长 |
| | `exp7_bar.*` | 稳态指标汇总柱状图 |
| | `exp7_straight_vs_corner.*` | 直道/弯道分段 RMSE 对比 |

---

## 注意事项

1. **Gazebo 仿真时间**：脚本使用挂钟时间（wall-clock time）作为超时。如果系统负载高导致仿真变慢（real-time factor < 1.0），实际完成的圈数可能少于预期。可通过增加 `--laps` 参数来补偿。

2. **重复运行**：每次运行会生成带时间戳的独立日志目录，不会覆盖之前的结果。

3. **中断恢复**：`Ctrl+C` 会安全关闭 roslaunch。已完成圈次的 CSV 数据不会丢失。

4. **磁盘空间**：每组实验的日志约 5–20 MB，全部实验（约 35 组）总计约 300–700 MB。

5. **GPU / 显示器**：Gazebo 需要图形界面。如在无头服务器上运行，请使用 `xvfb-run` 或设置 `DISPLAY` 环境变量到虚拟 framebuffer。
