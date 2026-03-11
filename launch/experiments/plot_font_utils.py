#!/usr/bin/env python3
"""Matplotlib 中文字体配置辅助模块。"""

import os
import sys

import numpy as np
from matplotlib import font_manager


def configure_chinese_font(plt):
    candidate_fonts = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]

    fallback_names = [
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "Noto Serif CJK JP",
        "Noto Serif CJK SC",
        "Droid Sans Fallback",
        "AR PL UKai CN",
        "AR PL UMing CN",
    ]
    selected_name = None

    for font_path in candidate_fonts:
        if not os.path.exists(font_path):
            continue
        font_manager.fontManager.addfont(font_path)
        selected_name = font_manager.FontProperties(fname=font_path).get_name()
        break

    if selected_name is None:
        for family_name in fallback_names:
            try:
                font_manager.findfont(family_name, fallback_to_default=False)
                selected_name = family_name
                break
            except ValueError:
                continue

    if selected_name is not None:
        plt.rcParams["font.family"] = [selected_name]
        plt.rcParams["font.sans-serif"] = [selected_name, "DejaVu Sans"]
    else:
        print("[警告] 未找到可用中文字体，图中的中文可能仍显示异常", file=sys.stderr)
        plt.rcParams["font.family"] = ["DejaVu Sans"]

    plt.rcParams["axes.unicode_minus"] = False


def set_lap_ticks(ax, *lap_sequences):
    lap_arrays = [np.asarray(seq, dtype=float).ravel() for seq in lap_sequences if seq is not None]
    if not lap_arrays:
        return

    lap_values = np.concatenate(lap_arrays)
    lap_values = lap_values[np.isfinite(lap_values)]
    if lap_values.size == 0:
        return

    max_lap = int(np.ceil(lap_values.max()))
    ticks = np.arange(0, max_lap + 1, 1)
    ax.set_xticks(ticks)
    ax.set_xlim(0, max_lap)