# AgriVision presentation graphs — Kaggle-ready
# Produces three charts:
# 1) Yield estimates shown in the supplied sample field reports
# 2) Confusion matrix + accuracy from a SIMULATED crop-classification exercise
# 3) Climate planning scores shown in the supplied sample field reports
#
# IMPORTANT:
# - Report values are sample recommendations, not model validation scores.
# - The classification data below is synthetic and trains a separate demo model.
# - The project reports do not contain labeled classification results or weather accuracy.

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

OUT = "/kaggle/working"
os.makedirs(OUT, exist_ok=True)

# -------------------- Presentation style --------------------
BG = "#F4F7F2"
INK = "#19392F"
MUTED = "#697B72"
GREEN = "#2A805A"
MINT = "#71A982"
GOLD = "#E6AD42"
BLUE = "#4D81B5"
CORAL = "#DF775C"
PURPLE = "#8C70B5"

sns.set_theme(style="white", font="DejaVu Sans")
plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 11,
    "axes.titleweight": "bold",
})

def style_axes(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#D9E2DA")
    ax.grid(axis="y", color="#DDE5DE", linewidth=0.9, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=8)

# ============================================================
# GRAPH 1 — Crop yields shown in the sample reports
# ============================================================
# Values copied from the example reports. Scenarios differ by
# location, season, year, and farm size; this is not a controlled comparison.
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
all_crops = sorted({crop for values in report_yields.values() for crop in values})
scenario_names = list(report_yields)
palette = [GREEN, BLUE, GOLD]

fig, ax = plt.subplots(figsize=(13, 7))
y = np.arange(len(all_crops))
height = 0.23
for i, (scenario, values) in enumerate(report_yields.items()):
    offset = (i - 1) * height
    for yi, crop in enumerate(all_crops):
        if crop not in values:
            continue
        val = values[crop]
        ax.barh(yi + offset, val, height=height*0.88, color=palette[i],
                edgecolor="white", linewidth=1.2,
                label=scenario if yi == 0 else None, zorder=3)
        ax.text(val + 0.45, yi + offset, f"{val:g}",
                va="center", fontsize=8.5, color=INK, fontweight="bold")

ax.set_yticks(y, all_crops)
ax.invert_yaxis()
ax.set_xlabel("Report yield estimate (tonnes per hectare)")
ax.set_title("Crop yield estimates in the sample AgriVision reports",
             loc="left", fontsize=19, pad=24)
ax.text(0, 1.025,
        "Different report scenarios · bars shown only for crops listed in each report",
        transform=ax.transAxes, fontsize=10, color=MUTED)
ax.legend(frameon=False, ncol=1, loc="lower right", fontsize=9)
style_axes(ax)
fig.text(0.10, 0.015,
         "Sample report outputs, not measured model accuracy; scenarios are not directly comparable.",
         fontsize=9, color=CORAL)
plt.tight_layout(rect=[0.02, 0.045, 1, 1])
fig.savefig(f"{OUT}/01_report_crop_yields.png", dpi=240, bbox_inches="tight")
plt.show()

# ============================================================
# GRAPH 2 — Synthetic classification confusion matrix + accuracy
# ============================================================
# Build a fictional feature/label dataset and train a demo Random Forest.
# This demonstrates the evaluation workflow; it is NOT the deployed
# AgriVision model and does not use the reports as training data.
crop_classes = ["Rice", "Maize", "Chickpea", "Cotton", "Banana", "Lentil"]
X, y_demo = make_classification(
    n_samples=1800,
    n_features=14,
    n_informative=8,
    n_redundant=3,
    n_repeated=0,
    n_classes=6,
    n_clusters_per_class=1,
    weights=[0.22, 0.20, 0.17, 0.16, 0.13, 0.12],
    class_sep=0.92,
    flip_y=0.065,
    random_state=27,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_demo, test_size=0.25, stratify=y_demo, random_state=19
)
demo_model = RandomForestClassifier(
    n_estimators=240,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=31,
    n_jobs=-1,
)
demo_model.fit(X_train, y_train)
pred = demo_model.predict(X_test)
acc = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred, labels=np.arange(len(crop_classes)))

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="YlGn",
    linewidths=1.5, linecolor=BG, square=True,
    xticklabels=crop_classes, yticklabels=crop_classes,
    cbar_kws={"label": "Test examples"},
    annot_kws={"size": 11, "weight": "bold"},
    ax=ax,
)
ax.set_title("Crop classification · synthetic demo evaluation",
             loc="left", fontsize=19, pad=26)
ax.text(0, 1.025,
        f"Random Forest trained on synthetic features · {len(y_test)} test examples · Accuracy {acc:.1%}",
        transform=ax.transAxes, fontsize=10, color=MUTED)
ax.set_xlabel("Predicted crop", labelpad=12)
ax.set_ylabel("Actual crop", labelpad=12)
ax.tick_params(axis="x", rotation=0, labelsize=9)
ax.tick_params(axis="y", rotation=0, labelsize=9)
style_axes(ax)
fig.text(0.13, 0.015,
         "Synthetic demo only — not measured AgriVision performance or evidence about its deployed model.",
         fontsize=9, color=CORAL)
plt.tight_layout(rect=[0.02, 0.045, 1, 1])
fig.savefig(f"{OUT}/02_synthetic_crop_confusion_matrix.png",
            dpi=240, bbox_inches="tight")
plt.show()

print(f"Synthetic demo crop classification accuracy: {acc:.1%}")
print(classification_report(y_test, pred, target_names=crop_classes, digits=3))

# ============================================================
# GRAPH 3 — Prototype climate planning scores from the reports
# ============================================================
# These are the climate exposure scores printed in the sample reports.
# They are planning indices, not weather prediction accuracy.
climate_scenarios = [
    "Hooghly\nSummer 2030",
    "North 24 Parganas\nKharif 2027",
    "North East Delhi\nSummer 2031",
]
risk_scores = [41.7, 47.2, 57.1]

fig, ax = plt.subplots(figsize=(9.5, 5.5))
bars = ax.bar(climate_scenarios, risk_scores,
              color=[GREEN, GOLD, CORAL], width=0.58,
              edgecolor="white", linewidth=2, zorder=3)
ax.set_ylim(0, 75)
ax.set_ylabel("Prototype climate exposure index (0–100)")
ax.set_title("Climate exposure scores in the sample reports",
             loc="left", fontsize=19, pad=24)
ax.text(0, 1.025,
        "Report planning scores for three different example locations and seasons",
        transform=ax.transAxes, fontsize=10, color=MUTED)
for bar, score in zip(bars, risk_scores):
    ax.text(bar.get_x() + bar.get_width()/2, score + 2,
            f"{score:.1f}", ha="center", fontsize=13,
            fontweight="bold", color=INK)
style_axes(ax)
fig.text(0.10, 0.015,
         "These are prototype risk indices, not forecast accuracy or observed probabilities of loss.",
         fontsize=9, color=CORAL)
plt.tight_layout(rect=[0.02, 0.045, 1, 1])
fig.savefig(f"{OUT}/03_report_climate_scores.png", dpi=240, bbox_inches="tight")
plt.show()

print("\nSaved charts to /kaggle/working:")
print("01_report_crop_yields.png")
print("02_synthetic_crop_confusion_matrix.png")
print("03_report_climate_scores.png")

