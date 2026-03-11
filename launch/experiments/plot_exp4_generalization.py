#!/usr/bin/env python3
"""实验 4：多赛道泛化验证 — 画图脚本

用法:
    python3 plot_exp4_generalization.py  <DIR_circular>  <DIR_racetrack>  <DIR_figure8>  <DIR_square>

目录名中应包含 'gen_<scenario>' 以自动提取赛道名称。

输出:
    <公共父目录>/exp4_generalization.pdf
    <公共父目录>/exp4_generalization.png
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

# ── 通用辅助 ───────────────────────────────────────────────────────────────
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


def load_eval_csv(log_dir):
    files = find_csv(log_dir, "mppi_eval__*__mppi_ilc_prior_eval.csv")
    if not files:
        return None
    df = pd.read_csv(files[0])
    df.columns = df.columns.str.strip()
    return df


def extract_scenario(dirname):
    """从目录名提取赛道名，例如 'exp4_gen_circular_20250311' → 'circular'"""
    m = re.search(r"gen_(\w+?)_\d{8}", dirname)
    if m:
        return m.group(1)
    m = re.search(r"gen_(\w+)$", dirname)
    if m:
        return m.group(1)
    return os.path.basename(dirname)


SCENARIO_ZH = {
    "circular": "圆形",
    "racetrack": "跑道形",
    "figure8": "8 字形",
    "square": "方形",
}

COLORS = {
    "circular": "#2196F3",
    "racetrack": "#4CAF50",
    "figure8": "#FF9800",
    "square": "#9C27B0",
}

# ── 主逻辑 ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="实验 4: 多赛道泛化画图")
    parser.add_argument("dirs", nargs="+", help="各赛道对应的日志目录")
    args = parser.parse_args()

    entries = []
    for d in args.dirs:
        d = os.path.expanduser(d)
        scenario = extract_scenario(os.path.basename(d))
        lap_df = load_lap_csv(d)
        eval_df = load_eval_csv(d)
        if lap_df is not None:
            entries.append((scenario, lap_df, eval_df, d))

    if not entries:
        sys.exit("[错误] 未找到有效数据")

    out_dir = os.path.dirname(entries[0][3])
    n_scenarios = len(entries)

    # ── 图1: 轨迹 + RMSE 多面板 ─────────────────────────────────────────
    ncols = min(n_scenarios, 2)
    nrows = (n_scenarios + ncols - 1) // ncols
    fig1, axes_grid = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n_scenarios == 1:
        axes_flat = [axes_grid]
    else:
        axes_flat = np.array(axes_grid).flatten()

    for idx, (scenario, lap_df, eval_df, _) in enumerate(entries):
        ax = axes_flat[idx]
        color = COLORS.get(scenario, "#607D8B")
        zh = SCENARIO_ZH.get(scenario, scenario)

        if eval_df is not None:
            ax.plot(eval_df["global_x"], eval_df["global_y"], "-", color=color,
                    alpha=0.6, linewidth=0.8, label="实际轨迹")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.set_title(f"{zh} ({scenario})")
        ax.grid(True, alpha=0.3)
        if eval_df is not None:
            ax.legend(fontsize=8)

    # 隐藏多余子图
    for idx in range(n_scenarios, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig1.suptitle("实验 4: 多赛道轨迹对比", fontsize=14, y=1.02)
    fig1.tight_layout()

    # ── 图2: 收敛曲线对比 ────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax_rmse = axes2[0]
    ax_head = axes2[1]
    for scenario, lap_df, _, _ in entries:
        color = COLORS.get(scenario, "#607D8B")
        zh = SCENARIO_ZH.get(scenario, scenario)
        ax_rmse.plot(lap_df["lap"], lap_df["lat_rmse"], "o-", label=zh,
                     color=color, linewidth=1.5, markersize=5)
        ax_head.plot(lap_df["lap"], lap_df["head_rmse_deg"], "o-", label=zh,
                     color=color, linewidth=1.5, markersize=5)

    ax_rmse.set_ylabel("横向 RMSE (m)")
    ax_rmse.set_title("各赛道 ILC 收敛曲线")
    ax_rmse.legend()
    ax_rmse.grid(True, alpha=0.3)
    ax_rmse.set_ylim(bottom=0)

    ax_head.set_xlabel("圈数 (lap)")
    ax_head.set_ylabel("航向 RMSE (°)")
    ax_head.set_title("航向误差收敛")
    ax_head.legend()
    ax_head.grid(True, alpha=0.3)
    ax_head.set_ylim(bottom=0)
    set_lap_ticks(ax_head, *[lap_df["lap"] for _, lap_df, _, _ in entries])
    fig2.tight_layout()

    # ── 图3: 稳态柱状图 ─────────────────────────────────────────────────
    n_steady = 3
    scenario_names = []
    ss_vals = {"lat_rmse": [], "head_rmse_deg": []}
    for scenario, lap_df, _, _ in entries:
        zh = SCENARIO_ZH.get(scenario, scenario)
        scenario_names.append(zh)
        ss = lap_df.tail(min(n_steady, len(lap_df))).mean(numeric_only=True)
        ss_vals["lat_rmse"].append(ss["lat_rmse"])
        ss_vals["head_rmse_deg"].append(ss["head_rmse_deg"])

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(scenario_names))
    width = 0.35
    ax3.bar(x - width / 2, ss_vals["lat_rmse"], width, label="横向 RMSE (m)", color="#2196F3")
    ax3.bar(x + width / 2, ss_vals["head_rmse_deg"], width, label="航向 RMSE (°)", color="#FF9800")
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenario_names)
    ax3.set_ylabel("稳态值 (最后 3 圈均值)")
    ax3.set_title("各赛道稳态跟踪精度对比")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")
    fig3.tight_layout()

    # ── 保存 ──────────────────────────────────────────────────────────────
    for ext in ("pdf", "png"):
        fig1.savefig(os.path.join(out_dir, f"exp4_generalization_traj.{ext}"), dpi=200, bbox_inches="tight")
        fig2.savefig(os.path.join(out_dir, f"exp4_generalization_curves.{ext}"), dpi=200, bbox_inches="tight")
        fig3.savefig(os.path.join(out_dir, f"exp4_generalization_bar.{ext}"), dpi=200, bbox_inches="tight")

    print(f"  已保存至: {out_dir}/exp4_generalization_*.pdf/png")
    plt.close("all")
    print("\n完成。")


if __name__ == "__main__":
    main()
