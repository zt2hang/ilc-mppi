#!/usr/bin/env python3
"""实验 6：采样效率分析 — 画图脚本

用法:
    python3 plot_exp6_sample_efficiency.py  <DIR_K032_off> <DIR_K032_on> <DIR_K032_aggressive> ...

目录名中应包含 'K<XXXX>_ilc_<off|on|aggressive>' 以自动提取 K 值和 ILC 状态。

输出:
    <公共父目录>/exp6_sample_efficiency_*.pdf/png

核心图表:
  1. RMSE vs K 曲线（三条线：off/on/aggressive）
  2. RMSE vs Lap 收敛曲线（选取代表性 K）
  3. 「等效采样数」比值表
  4. Max tracking error 对比
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

ILC_COLORS = {"off": "#F44336", "on": "#2196F3", "aggressive": "#4CAF50"}
ILC_LABELS = {"off": "ILC OFF", "on": "ILC ON (标准)", "aggressive": "ILC ON (激进)"}
ILC_MARKERS = {"off": "s", "on": "o", "aggressive": "^"}


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


def extract_k_ilc(dirname):
    """从目录名提取 K 和 ILC 状态
    例如 'exp6_K0064_ilc_aggressive_20250311' → (64, 'aggressive')
    """
    m = re.search(r"K(\d{4})_ilc_(off|on|aggressive)", dirname)
    if m:
        return int(m.group(1)), m.group(2)
    return None, None


def main():
    parser = argparse.ArgumentParser(description="实验 6: 采样效率分析画图")
    parser.add_argument("dirs", nargs="+", help="各 K×ILC 组合的日志目录")
    args = parser.parse_args()

    data = {}
    for d in args.dirs:
        d = os.path.expanduser(d)
        k, ilc = extract_k_ilc(os.path.basename(d))
        if k is None:
            print(f"[警告] 无法从 {d} 提取 K/ILC 信息，跳过")
            continue
        lap_df = load_lap_csv(d)
        if lap_df is not None:
            data[(k, ilc)] = lap_df

    if not data:
        sys.exit("[错误] 未找到有效数据")

    out_dir = os.path.dirname(os.path.expanduser(args.dirs[0]))
    all_k = sorted(set(k for k, _ in data.keys()))
    all_ilc = sorted(set(ilc for _, ilc in data.keys()),
                     key=lambda x: ["off", "on", "aggressive"].index(x)
                     if x in ["off", "on", "aggressive"] else 99)
    n_steady = 3  # last N laps for steady-state

    # ── 计算稳态指标 ─────────────────────────────────────────────────────
    ss = {}
    for (k, ilc), df in data.items():
        tail = df.tail(min(n_steady, len(df)))
        ss[(k, ilc)] = {
            "rmse_mean": tail["lat_rmse"].mean(),
            "rmse_std": tail["lat_rmse"].std(),
            "max_lat": tail["lat_rmse"].max(),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # 图1: RMSE vs K 曲线 (核心图)
    # ═══════════════════════════════════════════════════════════════════════
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    for ilc in all_ilc:
        ks = [k for k in all_k if (k, ilc) in ss]
        rmses = [ss[(k, ilc)]["rmse_mean"] for k in ks]
        stds = [ss[(k, ilc)]["rmse_std"] for k in ks]

        ax1.errorbar(ks, rmses, yerr=stds,
                     marker=ILC_MARKERS.get(ilc, "o"),
                     color=ILC_COLORS.get(ilc, "gray"),
                     label=ILC_LABELS.get(ilc, ilc),
                     linewidth=2, markersize=8, capsize=4)

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(all_k)
    ax1.set_xticklabels([str(k) for k in all_k])
    ax1.set_xlabel("采样数 K (log₂ scale)")
    ax1.set_ylabel("稳态横向 RMSE (m)")
    ax1.set_title("采样效率：ILC 先验 vs 纯 MPPI")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # 标注等效采样数
    if "off" in all_ilc and "aggressive" in all_ilc:
        for ilc_tag in ["on", "aggressive"]:
            if ilc_tag not in all_ilc:
                continue
            # 对每个 K_ilc，找到 ILC off 在哪个 K 能达到同样 RMSE
            ks_off = sorted([k for k in all_k if (k, "off") in ss])
            rmses_off = [ss[(k, "off")]["rmse_mean"] for k in ks_off]
            for k_ilc in sorted([k for k in all_k if (k, ilc_tag) in ss]):
                r = ss[(k_ilc, ilc_tag)]["rmse_mean"]
                # 找 off 中最接近的 RMSE
                equiv_k = None
                for i in range(len(ks_off) - 1):
                    if rmses_off[i] >= r >= rmses_off[i + 1]:
                        # Linear interpolation
                        frac = (rmses_off[i] - r) / max(rmses_off[i] - rmses_off[i + 1], 1e-9)
                        equiv_k = ks_off[i] + frac * (ks_off[i + 1] - ks_off[i])
                        break
                if equiv_k is not None and equiv_k > k_ilc * 1.3:
                    ax1.annotate(
                        f"≈K={int(equiv_k)}",
                        xy=(k_ilc, r),
                        xytext=(k_ilc * 1.3, r - 0.005),
                        fontsize=8, color=ILC_COLORS.get(ilc_tag, "gray"),
                        arrowprops=dict(arrowstyle="->", color=ILC_COLORS.get(ilc_tag, "gray"), lw=0.8)
                    )

    fig1.tight_layout()

    # ═══════════════════════════════════════════════════════════════════════
    # 图2: 收敛曲线 (选取代表性 K = 最小、中间、最大)
    # ═══════════════════════════════════════════════════════════════════════
    representative_k = []
    if len(all_k) >= 3:
        representative_k = [all_k[0], all_k[len(all_k) // 2], all_k[-1]]
    else:
        representative_k = all_k

    fig2, axes2 = plt.subplots(1, len(representative_k),
                                figsize=(5 * len(representative_k), 5),
                                sharey=True)
    if len(representative_k) == 1:
        axes2 = [axes2]

    for ax, k in zip(axes2, representative_k):
        for ilc in all_ilc:
            if (k, ilc) not in data:
                continue
            df = data[(k, ilc)]
            ax.plot(df["lap"], df["lat_rmse"],
                    marker=ILC_MARKERS.get(ilc, "o"),
                    color=ILC_COLORS.get(ilc, "gray"),
                    label=ILC_LABELS.get(ilc, ilc),
                    linewidth=1.5, markersize=4)
        ax.set_xlabel("圈数")
        ax.set_title(f"K = {k}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        set_lap_ticks(ax, *[data[(k, ilc)]["lap"]
                            for ilc in all_ilc if (k, ilc) in data])

    axes2[0].set_ylabel("横向 RMSE (m)")
    fig2.suptitle("逐圈收敛曲线 — 不同采样数", fontsize=14)
    fig2.tight_layout()

    # ═══════════════════════════════════════════════════════════════════════
    # 图3: 分组柱状图 — RMSE 对比
    # ═══════════════════════════════════════════════════════════════════════
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    n_ilc = len(all_ilc)
    x = np.arange(len(all_k))
    total_width = 0.7
    bar_w = total_width / n_ilc

    for i, ilc in enumerate(all_ilc):
        rmses = [ss[(k, ilc)]["rmse_mean"] if (k, ilc) in ss else 0 for k in all_k]
        offset = (i - n_ilc / 2 + 0.5) * bar_w
        bars = ax3.bar(x + offset, rmses, bar_w,
                       label=ILC_LABELS.get(ilc, ilc),
                       color=ILC_COLORS.get(ilc, "gray"), alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax3.annotate(f"{h:.4f}",
                             xy=(bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points",
                             ha="center", fontsize=7)

    ax3.set_xticks(x)
    ax3.set_xticklabels([str(k) for k in all_k])
    ax3.set_xlabel("采样数 K")
    ax3.set_ylabel("稳态横向 RMSE (m)")
    ax3.set_title("采样数 vs 跟踪精度 — 三种模式对比")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")
    fig3.tight_layout()

    # ═══════════════════════════════════════════════════════════════════════
    # 图4: 等效采样数比值表
    # ═══════════════════════════════════════════════════════════════════════
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.axis("off")

    table_data = [["K(ILC)", "RMSE(ILC)", "RMSE(off)", "K_equiv(off)", "效率增益"]]
    for ilc_tag in ["on", "aggressive"]:
        if ilc_tag not in all_ilc:
            continue
        ks_off = sorted([k for k in all_k if (k, "off") in ss])
        rmses_off = [ss[(k, "off")]["rmse_mean"] for k in ks_off]
        for ki in sorted([k for k in all_k if (k, ilc_tag) in ss]):
            r = ss[(ki, ilc_tag)]["rmse_mean"]
            equiv = "—"
            gain = "—"
            for i in range(len(ks_off) - 1):
                if rmses_off[i] >= r >= rmses_off[i + 1]:
                    frac = (rmses_off[i] - r) / max(rmses_off[i] - rmses_off[i + 1], 1e-9)
                    ek = ks_off[i] + frac * (ks_off[i + 1] - ks_off[i])
                    equiv = f"{int(ek)}"
                    gain = f"{ek / ki:.1f}×"
                    break
            r_off = ss.get((ki, "off"), {}).get("rmse_mean", 0)
            label_k = f"{ki} ({ILC_LABELS.get(ilc_tag, ilc_tag)})"
            table_data.append([label_k, f"{r:.4f}", f"{r_off:.4f}", equiv, gain])

    tbl = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                    cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)
    # Header styling
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#E3F2FD")
            cell.set_text_props(weight="bold")
    ax4.set_title("等效采样数分析", fontsize=13, pad=20)
    fig4.tight_layout()

    # ── 保存 ──────────────────────────────────────────────────────────────
    for ext in ("pdf", "png"):
        fig1.savefig(os.path.join(out_dir, f"exp6_rmse_vs_K.{ext}"), dpi=200, bbox_inches="tight")
        fig2.savefig(os.path.join(out_dir, f"exp6_convergence_curves.{ext}"), dpi=200, bbox_inches="tight")
        fig3.savefig(os.path.join(out_dir, f"exp6_bar_chart.{ext}"), dpi=200, bbox_inches="tight")
        fig4.savefig(os.path.join(out_dir, f"exp6_equiv_table.{ext}"), dpi=200, bbox_inches="tight")

    print(f"  已保存至: {out_dir}/exp6_*.pdf/png")
    plt.close("all")
    print("\n完成。")


if __name__ == "__main__":
    main()
