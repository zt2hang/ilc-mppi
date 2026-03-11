#!/usr/bin/env python3
"""实验 5：采样数预算分析 — 画图脚本

用法:
    python3 plot_exp5_sample_budget.py  <DIR_K256_on>  <DIR_K256_off>  <DIR_K512_on>  ...

目录名中应包含 'K<XXXX>_ilc_<on|off>' 以自动提取 K 值和 ILC 状态。

输出:
    <公共父目录>/exp5_sample_budget.pdf
    <公共父目录>/exp5_sample_budget.png
"""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict

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


def extract_k_ilc(dirname):
    """从目录名提取 K 和 ILC 状态
    例如 'exp5_K0256_ilc_on_20250311' → (256, 'on')
    """
    m = re.search(r"K(\d{4})_ilc_(on|off)", dirname)
    if m:
        return int(m.group(1)), m.group(2)
    return None, None


# ── 主逻辑 ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="实验 5: 采样数预算画图")
    parser.add_argument("dirs", nargs="+", help="各 K×ILC 组合的日志目录")
    args = parser.parse_args()

    # 加载数据, key = (K, ilc_state)
    data = {}
    eval_data = {}
    for d in args.dirs:
        d = os.path.expanduser(d)
        k, ilc = extract_k_ilc(os.path.basename(d))
        if k is None:
            print(f"[警告] 无法从 {d} 提取 K/ILC 信息，跳过")
            continue
        lap_df = load_lap_csv(d)
        if lap_df is not None:
            data[(k, ilc)] = lap_df
        eval_df = load_eval_csv(d)
        if eval_df is not None:
            eval_data[(k, ilc)] = eval_df

    if not data:
        sys.exit("[错误] 未找到有效数据")

    out_dir = os.path.dirname(os.path.expanduser(args.dirs[0]))

    # 获取所有 K 值（排序）
    all_k = sorted(set(k for k, _ in data.keys()))
    n_steady = 3

    # ── 计算稳态指标 ─────────────────────────────────────────────────────
    ss = {}  # (K, ilc) → dict of metrics
    for (k, ilc), df in data.items():
        tail = df.tail(min(n_steady, len(df))).mean(numeric_only=True)
        ss[(k, ilc)] = tail

    # ── 图1: 分组柱状图 — 横向 RMSE ────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(all_k))
    width = 0.35

    rmse_on = [ss[(k, "on")]["lat_rmse"] if (k, "on") in ss else 0 for k in all_k]
    rmse_off = [ss[(k, "off")]["lat_rmse"] if (k, "off") in ss else 0 for k in all_k]

    bars1 = ax1.bar(x - width / 2, rmse_on, width, label="ILC ON", color="#2196F3")
    bars2 = ax1.bar(x + width / 2, rmse_off, width, label="ILC OFF", color="#F44336")

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(k) for k in all_k])
    ax1.set_xlabel("采样数 K")
    ax1.set_ylabel("稳态横向 RMSE (m)")
    ax1.set_title("采样数预算 vs 跟踪精度 (ILC ON/OFF)")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0:
            ax1.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    fig1.tight_layout()

    # ── 图2: 计算时间对比 ────────────────────────────────────────────────
    fig2 = None
    calc_on = []
    calc_off = []
    valid_k_calc = []
    for k in all_k:
        ct_on = eval_data.get((k, "on"))
        ct_off = eval_data.get((k, "off"))
        if ct_on is not None or ct_off is not None:
            valid_k_calc.append(k)
            calc_on.append(ct_on["calc_time_ms"].mean() if ct_on is not None else 0)
            calc_off.append(ct_off["calc_time_ms"].mean() if ct_off is not None else 0)

    if valid_k_calc:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        x2 = np.arange(len(valid_k_calc))
        ax2.bar(x2 - width / 2, calc_on, width, label="ILC ON", color="#2196F3")
        ax2.bar(x2 + width / 2, calc_off, width, label="ILC OFF", color="#F44336")
        ax2.set_xticks(x2)
        ax2.set_xticklabels([str(k) for k in valid_k_calc])
        ax2.set_xlabel("采样数 K")
        ax2.set_ylabel("平均计算时间 (ms)")
        ax2.set_title("MPPI 求解计算时间 vs 采样数")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")
        fig2.tight_layout()

    # ── 图3: 收敛曲线对比（选取 K_min 和 K_max）──────────────────────
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    COLORS_K = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_k)))
    for i, k in enumerate(all_k):
        for ilc, ls in [("on", "-"), ("off", "--")]:
            if (k, ilc) not in data:
                continue
            df = data[(k, ilc)]
            label = f"K={k} ILC {'ON' if ilc == 'on' else 'OFF'}"
            ax3.plot(df["lap"], df["lat_rmse"], ls, color=COLORS_K[i],
                     label=label, linewidth=1.5, markersize=4, marker="o")

    ax3.set_xlabel("圈数 (lap)")
    ax3.set_ylabel("横向 RMSE (m)")
    ax3.set_title("不同采样数的 RMSE 收敛曲线")
    ax3.legend(loc="upper right", fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)
    set_lap_ticks(ax3, *[df["lap"] for df in data.values()])
    fig3.tight_layout()

    # ── 保存 ──────────────────────────────────────────────────────────────
    for ext in ("pdf", "png"):
        fig1.savefig(os.path.join(out_dir, f"exp5_sample_budget_bar.{ext}"), dpi=200, bbox_inches="tight")
        if fig2 is not None:
            fig2.savefig(os.path.join(out_dir, f"exp5_sample_budget_calc_time.{ext}"), dpi=200, bbox_inches="tight")
        fig3.savefig(os.path.join(out_dir, f"exp5_sample_budget_curves.{ext}"), dpi=200, bbox_inches="tight")

    print(f"  已保存至: {out_dir}/exp5_sample_budget_*.pdf/png")
    plt.close("all")
    print("\n完成。")


if __name__ == "__main__":
    main()
