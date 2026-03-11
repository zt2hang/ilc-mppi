#!/usr/bin/env python3
"""实验 1：ILC 收敛性验证 — 画图脚本

用法:
    python3 plot_exp1_convergence.py  <LOG_DIR>

输出:
    <LOG_DIR>/exp1_convergence.pdf
    <LOG_DIR>/exp1_convergence.png
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
    # 去掉列名前后空格
    df.columns = df.columns.str.strip()
    return df


# ── 主逻辑 ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="实验 1: ILC 收敛性画图")
    parser.add_argument("log_dir", help="实验日志目录")
    args = parser.parse_args()

    log_dir = os.path.expanduser(args.log_dir)
    df = load_lap_csv(log_dir)

    if df.empty:
        sys.exit("[错误] CSV 文件为空")

    laps = df["lap"].values

    # ── 创建 3 个子图 ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    # 子图 1: 横向 RMSE
    ax1 = axes[0]
    ax1.plot(laps, df["lat_rmse"], "o-", label="总体 RMSE", color="#2196F3", linewidth=2)
    if "lat_rmse_straight" in df.columns:
        ax1.plot(laps, df["lat_rmse_straight"], "s--", label="直道 RMSE", color="#4CAF50")
    if "lat_rmse_corner" in df.columns:
        ax1.plot(laps, df["lat_rmse_corner"], "^--", label="弯道 RMSE", color="#FF9800")
    ax1.set_ylabel("横向 RMSE (m)")
    ax1.set_title("ILC 收敛性验证 — 横向跟踪误差")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # 子图 2: 偏置 RMS
    ax2 = axes[1]
    ax2.plot(laps, df["bias_rms_vy"], "o-", label=r"$b_{v_y}$ RMS", color="#9C27B0", linewidth=2)
    ax2.plot(laps, df["bias_rms_omega"], "s-", label=r"$b_{\omega}$ RMS", color="#E91E63", linewidth=2)
    ax2.set_ylabel("偏置 RMS")
    ax2.set_title("ILC 偏置项演化")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    # 子图 3: 航向 RMSE
    ax3 = axes[2]
    ax3.plot(laps, df["head_rmse_deg"], "o-", label="航向 RMSE", color="#F44336", linewidth=2)
    ax3.set_xlabel("圈数 (lap)")
    ax3.set_ylabel("航向 RMSE (°)")
    ax3.set_title("航向跟踪误差")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)

    set_lap_ticks(ax3, laps)

    fig.tight_layout()

    # ── 保存 ──────────────────────────────────────────────────────────────
    for ext in ("pdf", "png"):
        out = os.path.join(log_dir, f"exp1_convergence.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  已保存: {out}")

    plt.close(fig)
    print("\n完成。")


if __name__ == "__main__":
    main()
