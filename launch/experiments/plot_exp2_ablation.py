#!/usr/bin/env python3
"""实验 2：ILC 消融实验 — 画图脚本

用法:
    python3 plot_exp2_ablation.py  <DIR_ILC_ON>  <DIR_ILC_OFF>

输出:
    <DIR_ILC_ON>/../exp2_ablation.pdf
    <DIR_ILC_ON>/../exp2_ablation.png
"""
import argparse
import glob
import os
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
        sys.exit(f"[错误] 未在 {log_dir} 中找到 ilc_lap_metrics__*.csv")
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


# ── 主逻辑 ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="实验 2: ILC 消融对比画图")
    parser.add_argument("dir_on", help="ILC ON 日志目录")
    parser.add_argument("dir_off", help="ILC OFF 日志目录")
    args = parser.parse_args()

    dir_on = os.path.expanduser(args.dir_on)
    dir_off = os.path.expanduser(args.dir_off)
    out_dir = os.path.dirname(dir_on)  # 父目录

    df_on = load_lap_csv(dir_on)
    df_off = load_lap_csv(dir_off)

    # ── 图1: 逐圈收敛曲线对比 ─────────────────────────────────────────────
    fig1, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1 = axes[0]
    ax1.plot(df_on["lap"], df_on["lat_rmse"], "o-", label="ILC ON", color="#2196F3", linewidth=2)
    ax1.plot(df_off["lap"], df_off["lat_rmse"], "s--", label="ILC OFF", color="#F44336", linewidth=2)
    ax1.set_ylabel("横向 RMSE (m)")
    ax1.set_title("ILC 消融实验 — 横向跟踪误差对比")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2 = axes[1]
    ax2.plot(df_on["lap"], df_on["head_rmse_deg"], "o-", label="ILC ON", color="#2196F3", linewidth=2)
    ax2.plot(df_off["lap"], df_off["head_rmse_deg"], "s--", label="ILC OFF", color="#F44336", linewidth=2)
    ax2.set_xlabel("圈数 (lap)")
    ax2.set_ylabel("航向 RMSE (°)")
    ax2.set_title("航向跟踪误差对比")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    set_lap_ticks(ax2, df_on["lap"], df_off["lap"])

    fig1.tight_layout()

    # ── 图2: 稳态指标柱状图 ───────────────────────────────────────────────
    # 取最后 3 圈的均值作为稳态
    n_steady = min(3, len(df_on), len(df_off))
    ss_on = df_on.tail(n_steady).mean(numeric_only=True)
    ss_off = df_off.tail(n_steady).mean(numeric_only=True)

    metrics = ["lat_rmse", "lat_rmse_corner", "head_rmse_deg"]
    labels = ["总体 RMSE (m)", "弯道 RMSE (m)", "航向 RMSE (°)"]
    available = [(m, l) for m, l in zip(metrics, labels) if m in ss_on.index]
    if not available:
        available = [("lat_rmse", "总体 RMSE (m)")]

    met_keys = [a[0] for a in available]
    met_labels = [a[1] for a in available]

    x = np.arange(len(met_keys))
    width = 0.35

    fig2, ax = plt.subplots(figsize=(8, 5))
    bars_on = ax.bar(x - width / 2, [ss_on[m] for m in met_keys], width, label="ILC ON", color="#2196F3")
    bars_off = ax.bar(x + width / 2, [ss_off[m] for m in met_keys], width, label="ILC OFF", color="#F44336")

    ax.set_xticks(x)
    ax.set_xticklabels(met_labels)
    ax.set_ylabel("稳态值 (最后3圈均值)")
    ax.set_title("ILC ON vs OFF — 稳态跟踪精度")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 在柱子上方标注数值
    for bar in list(bars_on) + list(bars_off):
        h = bar.get_height()
        ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    fig2.tight_layout()

    # ── 图3: 轨迹对比（如有 per-cycle 数据）─────────────────────────────
    eval_on = load_eval_csv(dir_on)
    eval_off = load_eval_csv(dir_off)
    fig3 = None

    if eval_on is not None and eval_off is not None:
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        ax3.plot(eval_on["global_x"], eval_on["global_y"], "-", label="ILC ON",
                 color="#2196F3", alpha=0.7, linewidth=0.8)
        ax3.plot(eval_off["global_x"], eval_off["global_y"], "-", label="ILC OFF",
                 color="#F44336", alpha=0.7, linewidth=0.8)
        ax3.set_xlabel("X (m)")
        ax3.set_ylabel("Y (m)")
        ax3.set_title("实际轨迹对比")
        ax3.set_aspect("equal")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()

    # ── 保存 ──────────────────────────────────────────────────────────────
    for ext in ("pdf", "png"):
        fig1.savefig(os.path.join(out_dir, f"exp2_ablation_curves.{ext}"), dpi=200, bbox_inches="tight")
        fig2.savefig(os.path.join(out_dir, f"exp2_ablation_bar.{ext}"), dpi=200, bbox_inches="tight")
        if fig3 is not None:
            fig3.savefig(os.path.join(out_dir, f"exp2_ablation_traj.{ext}"), dpi=200, bbox_inches="tight")

    print(f"  已保存至: {out_dir}/exp2_ablation_*.pdf/png")
    plt.close("all")
    print("\n完成。")


if __name__ == "__main__":
    main()
