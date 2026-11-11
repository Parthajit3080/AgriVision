# Kaggle cell: check report yield estimates against broad public benchmarks.
# This creates a yield-plausibility matrix, NOT a classification confusion matrix.
# Confusion matrices require actual crop labels paired with predictions.
#
# Benchmark sources (India, approximate broad averages):
# - Govt. of India, Agricultural Statistics at a Glance 2024-25:
#   Rice 2.835, Wheat 3.547, Maize 3.422, Jute 2.665,
#   Sugarcane 82.741 tonnes/ha (normal estimates / major crop table).
# - Potato: Government 2024-25 state data gives West Bengal about 26.585 t/ha;
#   All-India about 24.568 t/ha.
# Broad averages are context only; local, seasonal, variety, moisture,
# and harvested-product conventions can differ.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm

OUT = "/kaggle/working"
os.makedirs(OUT, exist_ok=True)

report_yields = {
    "Hooghly · Summer 2030": {
        "Jute": 12.44, "Potato": 24.46, "Turmeric": 3.39,
        "Dry chillies": 1.10, "Wheat": 1.18,
    },
    "North 24 Parganas · Kharif 2027": {
        "Garlic": 13.80, "Cardamom": 1.04, "Jute": 13.44,
        "Turmeric": 2.59, "Potato": 16.06,
    },
    "North East Delhi · Summer 2031": {
        "Potato": 18.90, "Wheat": 1.18, "Rice": 2.30,
        "Maize": 2.02, "Sugarcane": 35.36,
    },
}

# Reference values in tonnes/ha. For potato use state reference for WB scenarios
# and All-India reference for Delhi. Other crops use broad India benchmarks.
columns = ["Jute", "Potato", "Wheat", "Rice", "Maize", "Sugarcane"]
all_india_benchmarks = {
    "Jute": 2.665,
    "Potato": 24.568,
    "Wheat": 3.547,
    "Rice": 2.835,
    "Maize": 3.422,
    "Sugarcane": 82.741,
}
wb_potato_benchmark = 26.585

# Ratio bands are illustrative screening thresholds:
# Below = <75% of reference; Near = 75%-125%; Above = >125%.
# They do not decide whether a yield is agronomically possible.
def ref_for(scenario, crop):
    if crop == "Potato" and scenario.startswith(("Hooghly", "North 24")):
        return wb_potato_benchmark
    return all_india_benchmarks[crop]

ratios = pd.DataFrame(index=list(report_yields), columns=columns, dtype=float)
actual_values = pd.DataFrame(index=list(report_yields), columns=columns, dtype=float)
for scenario, crops in report_yields.items():
    for crop in columns:
        if crop in crops:
            actual_values.loc[scenario, crop] = crops[crop]
            ratios.loc[scenario, crop] = crops[crop] / ref_for(scenario, crop)

# Draw colored matrix: red = above broad reference; green = near; amber = below.
# Blank cells mean that crop was not listed in that report.
display = np.full(ratios.shape, np.nan)
labels = np.empty(ratios.shape, dtype=object)
labels[:] = ""
for i, scenario in enumerate(ratios.index):
    for j, crop in enumerate(columns):
        ratio = ratios.iloc[i, j]
        if pd.notna(ratio):
            display[i, j] = 0 if ratio < 0.75 else (1 if ratio <= 1.25 else 2)
            value = actual_values.iloc[i, j]
            tag = "LOW" if ratio < 0.75 else ("NEAR" if ratio <= 1.25 else "HIGH")
            labels[i, j] = f"{value:.2f} t/ha\n{ratio:.1f}× ref · {tag}"

sns.set_theme(style="white", font="DejaVu Sans")
fig, ax = plt.subplots(figsize=(13, 5.8), facecolor="#F5F7F2")
ax.set_facecolor("#EEF2ED")

cmap = ListedColormap(["#E7B44D", "#5A9B72", "#D9755D"])
cmap.set_bad("#E4E9E4")
sns.heatmap(
    display, mask=np.isnan(display), cmap=cmap, vmin=-0.5, vmax=2.5,
    annot=labels, fmt="", linewidths=3, linecolor="#F5F7F2",
    cbar=False, square=False, ax=ax,
    annot_kws={"fontsize": 10, "fontweight": "bold", "color": "#173A2E"}
)

ax.set_title("Reported yields · plausibility screening matrix",
             loc="left", fontsize=19, fontweight="bold",
             color="#19392F", pad=25)
ax.text(0, 1.04,
        "Cell shows report yield and ratio to broad public benchmark · blank = crop absent from report",
        transform=ax.transAxes, fontsize=10, color="#687A72")
ax.set_xlabel("Crop", fontsize=11, labelpad=12, color="#52665C")
ax.set_ylabel("Sample report scenario", fontsize=11, labelpad=12, color="#52665C")
ax.set_xticklabels(columns, rotation=0, fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
for s in ax.spines.values():
    s.set_visible(False)

# Compact color legend
from matplotlib.patches import Patch
legend = [
    Patch(facecolor="#E7B44D", label="Below 75% of reference"),
    Patch(facecolor="#5A9B72", label="Within 75%-125%"),
    Patch(facecolor="#D9755D", label="Above 125%"),
]
ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.20),
          ncol=3, frameon=False, fontsize=9)

fig.text(0.08, 0.015,
         "Not a confusion matrix or model-accuracy test. Broad benchmarks are not location/season-specific; verify crop, unit, and harvest basis.",
         fontsize=8.5, color="#B65342")
plt.tight_layout(rect=[0.02, 0.08, 1, 0.98])
path = f"{OUT}/report_yield_plausibility_matrix.png"
plt.savefig(path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()

print("Saved:", path)
print("\nInterpretation:")
print("- Jute values are about 4.7× and 5.0× the broad India reference; investigate units, data, or crop encoding.")
print("- Wheat estimates are about one-third of the broad reference in the reports.")
print("- Hooghly potato is close to the West Bengal reference; other potato values are lower.")
print("- Rice is near its broad reference; maize and sugarcane are below their broad references.")
print("- These gaps flag values for review; they do not prove a prediction is wrong.")
print("- No confusion matrix can be inferred without actual crop labels and predicted labels.")

