"""Matplotlib styling — Arial, large & figure-ready."""
import matplotlib.pyplot as plt
from matplotlib import font_manager


def set_style():
    """Register Arial (if present) and apply large, readable, figure-ready defaults."""
    for fp in ["/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        try:
            font_manager.fontManager.addfont(fp)
        except Exception:
            pass
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 14,
        "axes.titlesize": 15, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.linewidth": 1.0, "figure.dpi": 110,
    })
