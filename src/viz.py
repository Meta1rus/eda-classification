"""Единый стиль графиков для всех тетрадок."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from .data import CLASS_COLORS

# Категориальная палитра (фиксированный порядок, проверена на различимость при дальтонизме)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE = CATEGORICAL[0], CATEGORICAL[1]
CLASS_PALETTE = [CLASS_COLORS["Нет покупки"], CLASS_COLORS["Покупка"]]

# Последовательная шкала (один оттенок, светлый -> тёмный) и расходящаяся (синий — серый — красный)
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f3a6a5", "#e34948", "#9e2223"])


def setup() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "figure.figsize": (9, 4.5),
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#c9c8c3",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#ebeae6",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
    })


def class_name(y) -> list[str]:
    return ["Покупка" if int(v) == 1 else "Нет покупки" for v in y]


def annotate_bars(ax, fmt="{:.0f}", fontsize=8, inside=False):
    for p in ax.patches:
        h = p.get_height()
        if h != h or h == 0:  # nan или 0
            continue
        ax.annotate(fmt.format(h), (p.get_x() + p.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=fontsize, color="#52514e",
                    xytext=(0, 2), textcoords="offset points")
