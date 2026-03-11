#!/usr/bin/env python3
"""实验 3：遗忘因子 γ 扫描 — 画图脚本

用法:
    python3 plot_exp3_decay_sweep.py  <DIR_γ1>  <DIR_γ2>  ...

目录名中应包含 'decay_XXXX' 以自动提取 γ 值（如 exp3_decay_0980_...）。

输出:
    <公共父目录>/exp3_decay_sweep.pdf
    <公共父目录>/exp3_decay_sweep.png
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


def extract_gamma(dirname):
    """从目录名中提取 γ 值，例如 'exp3_decay_0980_20250311' → 0.980"""
    m = re.search(r"decay_(\d{4})", dirname)
    if m:
        code = m.group(1)
        if code == "1000":
            return 1.0
        return float("0." + code[1:])    # "0980" → 0.980
    return None


# ── 色彩方案 ──────────────────────────────────────────────────────────────
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336",
          "#00BCD4", "#795548", "#607D8B"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


# ── 主逻辑 ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="实验 3: γ 遗忘因子扫描画图")
    parser.add_argument("dirs", nargs="+", help="各 γ 值对应的日志目录")
    args = parser.parse_args()

    # 加载数据
    entries = []
    for d in args.dirs:
        d = os.path.expanduser(d)
        gamma = extract_gamma(os.path.basename(d))
        df = load_lap_csv(d)
        if df is not None and gamma is not None:
            entries.append((gamma, df, d))
    entries.sort(key=lambda x: x[0])

    if not entries:
        sys.exit("[错误] 未找到有效数据")

    out_dir = os.path.dirname(entries[0][2])

    # ── 图1: 多曲线 — 横向 RMSE vs 圈数 ──────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax1 = axes[0]
    for i, (gamma, df, _) in enumerate(entries):
        c = COLORS[i % len(COLORS)]
        m = MARKERS[i % len(MARKERS)]
        ax1.plot(df["lap"], df["lat_rmse"], f"{m}-", label=f"γ = {gamma:.3f}",
                 color=c, linewidth=1.5, markersize=5)
    ax1.set_ylabel("横向 RMSE (m)")
    ax1.set_title("遗忘因子 γ 对收敛性的影响")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2 = axes[1]
    for i, (gamma, df, _) in enumerate(entries):
        c = COLORS[i % len(COLORS)]
        m = MARKERS[i % len(MARKERS)]
        ax2.plot(df["lap"], df["bias_rms_vy"], f"{m}-", label=f"γ = {gamma:.3f}",
                 color=c, linewidth=1.5, markersize=5)
    ax2.set_xlabel("圈数 (lap)")
    ax2.set_ylabel(r"偏置 $b_{v_y}$ RMS")
    ax2.set_title("ILC 偏置演化 (各 γ 值)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    set_lap_ticks(ax2, *[df["lap"] for _, df, _ in entries])

    fig.tight_layout()

    # ── 图2: 稳态 RMSE 柱状图 ────────────────────────────────────────────
    n_steady = 3
    gammas = []
    ss_rmse = []
    ss_rmse_corner = []
    for gamma, df, _ in entries:
        ss = df.tail(min(n_steady, len(df))).mean(numeric_only=True)
        gammas.append(f"{gamma:.3f}")
        ss_rmse.append(ss["lat_rmse"])
        if "lat_rmse_corner" in ss.index:
            ss_rmse_corner.append(ss["lat_rmse_corner"])

    fig2, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(gammas))
    width = 0.35
    ax.bar(x - width / 2, ss_rmse, width, label="总体 RMSE", color="#2196F3")
    if ss_rmse_corner:
        ax.bar(x + width / 2, ss_rmse_corner, width, label="弯道 RMSE", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels([f"γ={g}" for g in gammas])
    ax.set_ylabel("稳态 RMSE (m) — 最后 3 圈均值")
    ax.set_title("不同 γ 值的稳态跟踪精度")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig2.tight_layout()

    # ── 保存 ──────────────────────────────────────────────────────────────
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"exp3_decay_sweep_curves.{ext}"), dpi=200, bbox_inches="tight")
        fig2.savefig(os.path.join(out_dir, f"exp3_decay_sweep_bar.{ext}"), dpi=200, bbox_inches="tight")

    print(f"  已保存至: {out_dir}/exp3_decay_sweep_*.pdf/png")
    plt.close("all")
    print("\n完成。")


if __name__ == "__main__":
    main()
