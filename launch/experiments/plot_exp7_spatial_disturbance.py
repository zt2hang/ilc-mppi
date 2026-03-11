#!/usr/bin/env python3
"""实验 7：空间扰动补偿 — 画图脚本

用法:
    python3 plot_exp7_spatial_disturbance.py  <DIR_ilc_off> <DIR_ilc_on> <DIR_ilc_aggressive> <DIR_full_stack>

目录名中应包含 'disturbance_ilc_off|on|aggressive|full_stack' 以自动提取模式。

输出:
    <公共父目录>/exp7_spatial_disturbance_*.pdf/png

核心图表:
  1. 逐圈 RMSE 收敛曲线（四条线）
  2. 按弧长展开的横向误差分布（选取第一圈 vs 最后三圈均值）
  3. ILC bias 沿弧长的分布（展示空间学习效果）
  4. 柱状图汇总
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_font_utils import configure_chinese_font, set_lap_ticks

configure_chinese_font(plt)

MODE_COLORS = {
    "disturbance_ilc_off": "#F44336",
    "disturbance_ilc_on": "#2196F3",
    "disturbance_ilc_aggressive": "#4CAF50",
    "disturbance_full_stack": "#FF9800",
}

MODE_LABELS = {
    "disturbance_ilc_off": "纯 MPPI (无补偿)",
    "disturbance_ilc_on": "ILC 标准",
    "disturbance_ilc_aggressive": "ILC 激进",
    "disturbance_full_stack": "全栈补偿 (无 ILC)",
}


def find_csv(log_dir, pattern):
    return sorted(glob.glob(os.path.join(log_dir, "**", pattern), recursive=True))


def load_lap_csv(log_dir):
    files = find_csv(log_dir, "ilc_lap_metrics__*.csv")
    if not files:
        print(f"[警告] 未在 {log_dir} 中找到 ilc_lap_metrics__*.csv，跳过")
        return None
    df = pd.read_csv(files[0])
    df.columns = df.columns.str.strip()
    return df


def extract_mode(dirname):
    """从目录名提取实验模式"""
    for mode in MODE_COLORS:
        if mode in dirname:
            return mode
    return None


def main():
    parser = argparse.ArgumentParser(description="实验 7: 空间扰动补偿画图")
    parser.add_argument("dirs", nargs="+", help="各模式的日志目录")
    args = parser.parse_args()

    data = {}
    for d in args.dirs:
        d = os.path.expanduser(d)
        mode = extract_mode(os.path.basename(d))
        if mode is None:
            print(f"[警告] 无法从 {d} 提取模式，跳过")
            continue
        lap_df = load_lap_csv(d)
        if lap_df is not None:
            data[mode] = lap_df

    if not data:
        sys.exit("[错误] 未找到有效数据")

    out_dir = os.path.dirname(os.path.expanduser(args.dirs[0]))

    # Ordered modes for consistent plotting
    mode_order = [m for m in MODE_COLORS if m in data]
    n_steady = 3

    # ═══════════════════════════════════════════════════════════════════════
    # 图1: 逐圈 RMSE 收敛曲线
    # ═══════════════════════════════════════════════════════════════════════
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    for mode in mode_order:
        df = data[mode]
        ax1.plot(df["lap"], df["lat_rmse"],
                 marker="o", markersize=4, linewidth=1.8,
                 color=MODE_COLORS[mode],
                 label=MODE_LABELS.get(mode, mode))

    ax1.set_xlabel("圈数 (lap)")
    ax1.set_ylabel("横向 RMSE (m)")
    ax1.set_title("空间扰动下的逐圈收敛 (d_vy = A·sin(2πns/L))")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    set_lap_ticks(ax1, *[data[m]["lap"] for m in mode_order])
    fig1.tight_layout()

    # ═══════════════════════════════════════════════════════════════════════
    # 图2: ILC bias 统计（RMS 和 Max 随圈数变化）
    # ═══════════════════════════════════════════════════════════════════════
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))

    for mode in mode_order:
        df = data[mode]
        if "bias_rms_vy" not in df.columns:
            continue
        ax2a.plot(df["lap"], df["bias_rms_vy"],
                  marker="o", markersize=3, linewidth=1.5,
                  color=MODE_COLORS[mode],
                  label=MODE_LABELS.get(mode, mode))
        ax2b.plot(df["lap"], df["bias_max_abs_vy"],
                  marker="s", markersize=3, linewidth=1.5,
                  color=MODE_COLORS[mode],
                  label=MODE_LABELS.get(mode, mode))

    ax2a.set_xlabel("圈数")
    ax2a.set_ylabel("ILC bias RMS (vy) [m/s]")
    ax2a.set_title("ILC 偏置 RMS 增长")
    ax2a.legend(fontsize=8)
    ax2a.grid(True, alpha=0.3)

    ax2b.set_xlabel("圈数")
    ax2b.set_ylabel("ILC bias max|vy| [m/s]")
    ax2b.set_title("ILC 偏置峰值增长")
    ax2b.legend(fontsize=8)
    ax2b.grid(True, alpha=0.3)

    fig2.suptitle("ILC 空间记忆积累过程", fontsize=14)
    fig2.tight_layout()

    # ═══════════════════════════════════════════════════════════════════════
    # 图3: 稳态性能汇总柱状图
    # ═══════════════════════════════════════════════════════════════════════
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

    modes_ordered = mode_order
    x = np.arange(len(modes_ordered))

    # RMSE
    rmse_vals = []
    for m in modes_ordered:
        tail = data[m].tail(min(n_steady, len(data[m])))
        rmse_vals.append(tail["lat_rmse"].mean())

    colors = [MODE_COLORS[m] for m in modes_ordered]
    labels = [MODE_LABELS.get(m, m) for m in modes_ordered]

    bars1 = ax3a.bar(x, rmse_vals, color=colors, alpha=0.85)
    ax3a.set_xticks(x)
    ax3a.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax3a.set_ylabel("稳态横向 RMSE (m)")
    ax3a.set_title("跟踪精度对比")
    ax3a.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars1, rmse_vals):
        ax3a.annotate(f"{val:.4f}",
                      xy=(bar.get_x() + bar.get_width() / 2, val),
                      xytext=(0, 4), textcoords="offset points",
                      ha="center", fontsize=9, fontweight="bold")

    # Heading RMSE
    head_vals = []
    for m in modes_ordered:
        tail = data[m].tail(min(n_steady, len(data[m])))
        if "head_rmse_deg" in tail.columns:
            head_vals.append(tail["head_rmse_deg"].mean())
        else:
            head_vals.append(0)

    bars2 = ax3b.bar(x, head_vals, color=colors, alpha=0.85)
    ax3b.set_xticks(x)
    ax3b.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax3b.set_ylabel("稳态航向 RMSE (deg)")
    ax3b.set_title("航向精度对比")
    ax3b.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars2, head_vals):
        if val > 0:
            ax3b.annotate(f"{val:.2f}",
                          xy=(bar.get_x() + bar.get_width() / 2, val),
                          xytext=(0, 4), textcoords="offset points",
                          ha="center", fontsize=9, fontweight="bold")

    fig3.suptitle("空间扰动补偿 — 稳态性能汇总", fontsize=14)
    fig3.tight_layout()

    # ═══════════════════════════════════════════════════════════════════════
    # 图4: 弯道 vs 直道 RMSE 分解
    # ═══════════════════════════════════════════════════════════════════════
    fig4 = None
    if all("lat_rmse_straight" in data[m].columns and "lat_rmse_corner" in data[m].columns
           for m in modes_ordered):
        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))

        for m in modes_ordered:
            df = data[m]
            ax4a.plot(df["lap"], df["lat_rmse_straight"],
                      marker="o", markersize=3, linewidth=1.5,
                      color=MODE_COLORS[m], label=MODE_LABELS.get(m, m))
            ax4b.plot(df["lap"], df["lat_rmse_corner"],
                      marker="s", markersize=3, linewidth=1.5,
                      color=MODE_COLORS[m], label=MODE_LABELS.get(m, m))

        ax4a.set_xlabel("圈数")
        ax4a.set_ylabel("横向 RMSE (m)")
        ax4a.set_title("直道误差")
        ax4a.legend(fontsize=8)
        ax4a.grid(True, alpha=0.3)

        ax4b.set_xlabel("圈数")
        ax4b.set_ylabel("横向 RMSE (m)")
        ax4b.set_title("弯道误差")
        ax4b.legend(fontsize=8)
        ax4b.grid(True, alpha=0.3)

        fig4.suptitle("弯道 vs 直道 — 空间扰动补偿效果", fontsize=14)
        fig4.tight_layout()

    # ── 保存 ──────────────────────────────────────────────────────────────
    for ext in ("pdf", "png"):
        fig1.savefig(os.path.join(out_dir, f"exp7_convergence.{ext}"), dpi=200, bbox_inches="tight")
        fig2.savefig(os.path.join(out_dir, f"exp7_bias_growth.{ext}"), dpi=200, bbox_inches="tight")
        fig3.savefig(os.path.join(out_dir, f"exp7_summary_bar.{ext}"), dpi=200, bbox_inches="tight")
        if fig4 is not None:
            fig4.savefig(os.path.join(out_dir, f"exp7_straight_vs_corner.{ext}"), dpi=200, bbox_inches="tight")

    print(f"  已保存至: {out_dir}/exp7_*.pdf/png")
    plt.close("all")
    print("\n完成。")


if __name__ == "__main__":
    main()
