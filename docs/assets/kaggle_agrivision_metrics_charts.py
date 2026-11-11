# AgriVision — polished presentation charts (synthetic demo data)
# Run as one Kaggle notebook cell. PNGs are saved to /kaggle/working.
# IMPORTANT: project files do not include measured crop-classification or
# calibrated weather-forecast accuracy. Values below are synthetic examples.

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

OUT = "/kaggle/working"
os.makedirs(OUT, exist_ok=True)

# Synthetic, deliberately non-rounded example values; replace with real results.
crop_metrics = {
    "Precision": 0.873,
    "Recall":    0.814,
    "F1-score":  0.842,
    "Accuracy":  0.861,
}

# Illustrative simulated weather-label accuracy by forecast lead.
# Not a measured AgriVision result or a calibrated forecast evaluation.
lead_days = np.array([1, 2, 3, 5, 7, 10, 14])
weather_accuracy = np.array([0.914, 0.889, 0.852, 0.811, 0.763, 0.714, 0.668])

# ---------- Visual style ----------
BG = "#F5F7F2"
INK = "#19392F"
MUTED = "#687A72"
GRID = "#DCE5DE"
GREEN = "#25845D"
TEAL = "#2A9D8F"
BLUE = "#477DB3"
GOLD = "#E3A93B"
CORAL = "#E6785D"
PURPLE = "#8267B5"

sns.set_theme(style="white", font="DejaVu Sans")
plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.edgecolor": BG,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.titlepad": 20,
    "font.size": 11,
})

def finish(ax, title, subtitle, ylabel=""):
    ax.set_title(title, loc="left", fontsize=19, pad=28)
    ax.text(0, 1.025, subtitle, transform=ax.transAxes,
            fontsize=10, color=MUTED, va="bottom")
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=10)
    ax.set_ylim(0, 1.06)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=8)

# ==========================================================
# 1. Crop metrics: Precision, Recall, F1-score AND Accuracy
# ==========================================================
fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
fig.subplots_adjust(top=0.80, bottom=0.18, left=0.10, right=0.96)
labels = list(crop_metrics.keys())
values = np.array(list(crop_metrics.values()))
colors = [BLUE, TEAL, PURPLE, GREEN]
x = np.arange(len(labels))

# Soft horizontal guide band gives the bars a presentation-card look.
ax.axhspan(0.8, 0.9, color="#E9F2EB", alpha=0.75, zorder=0)
bars = ax.bar(x, values, width=0.58, color=colors, zorder=3,
              edgecolor="white", linewidth=2)
for bar, val in zip(bars, values):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.035, f"{val:.1%}",
            ha="center", va="bottom", fontsize=13, fontweight="bold", color=INK)
ax.set_xticks(x, labels, fontsize=12, fontweight="semibold")
finish(ax, "Crop prediction · evaluation snapshot",
       "Synthetic example · all four classification metrics on one scale",
       "Score")
ax.text(0.99, -0.18, "ILLUSTRATIVE DATA", transform=ax.transAxes,
        ha="right", va="center", fontsize=9, color=CORAL, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", fc="#FCECE7", ec="none"))
fig.savefig(f"{OUT}/01_crop_metrics.png", dpi=240, bbox_inches="tight")
plt.show()

# ==========================================================
# 2. Weather prediction accuracy by forecast lead
# ==========================================================
fig, ax = plt.subplots(figsize=(11, 6.2), constrained_layout=True)
fig.subplots_adjust(top=0.80, bottom=0.18, left=0.10, right=0.96)
ax.axhspan(0.8, 1.0, color="#E9F2EB", alpha=0.72, zorder=0)
ax.axhspan(0.0, 0.7, color="#F8ECE8", alpha=0.55, zorder=0)
ax.plot(lead_days, weather_accuracy, color=GREEN, linewidth=3.2,
        marker="o", markersize=10, markerfacecolor=GOLD,
        markeredgecolor="white", markeredgewidth=2, zorder=4)
ax.fill_between(lead_days, weather_accuracy, 0.6, color=GREEN, alpha=0.10)
for d, v in zip(lead_days, weather_accuracy):
    ax.annotate(f"{v:.1%}", (d, v), xytext=(0, 13),
                textcoords="offset points", ha="center",
                fontsize=10, fontweight="bold", color=INK)
ax.set_xticks(lead_days, [f"{d} day" if d == 1 else f"{d} days" for d in lead_days])
ax.set_xlim(0, 15)
finish(ax, "Weather prediction · accuracy by lead time",
       "Synthetic example · illustrative accuracy declines at longer forecast horizons",
       "Illustrative accuracy")
ax.text(0.99, -0.18, "SIMULATED EXAMPLE · NOT MEASURED", transform=ax.transAxes,
        ha="right", va="center", fontsize=9, color=CORAL, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", fc="#FCECE7", ec="none"))
fig.savefig(f"{OUT}/02_weather_accuracy.png", dpi=240, bbox_inches="tight")
plt.show()

print("Saved files:")
print(f"{OUT}/01_crop_metrics.png")
print(f"{OUT}/02_weather_accuracy.png")
print("\nReminder: in the current project, crop suitability classification and")
print("calibrated weather-forecast accuracy are not recorded evaluation results.")

