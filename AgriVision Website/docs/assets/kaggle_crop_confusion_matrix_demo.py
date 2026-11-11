# Kaggle-ready crop classification confusion matrix demo
# SIMULATED example only. This constructs illustrative predictions directly;
# it does not train/evaluate the deployed AgriVision model.

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Six illustrative crop classes, 50 synthetic test cases per class.
classes = ["Rice", "Maize", "Chickpea", "Cotton", "Banana", "Lentil"]

# Rows = true class; columns = predicted class.
# Counts are intentionally constructed to demonstrate a confusion matrix.
counts = np.array([
    [44, 2, 1, 1, 1, 1],  # Rice
    [ 3,41, 2, 1, 1, 2],  # Maize
    [ 1, 2,43, 1, 1, 2],  # Chickpea
    [ 1, 1, 1,45, 1, 1],  # Cotton
    [ 1, 1, 0, 1,46, 1],  # Banana
    [ 2, 2, 2, 1, 1,42],  # Lentil
], dtype=int)

# Expand the illustrative count table into y_true/y_pred arrays.
y_true, y_pred = [], []
for true_idx, row in enumerate(counts):
    for pred_idx, n in enumerate(row):
        y_true.extend([classes[true_idx]] * int(n))
        y_pred.extend([classes[pred_idx]] * int(n))

y_true = np.array(y_true)
y_pred = np.array(y_pred)
cm = confusion_matrix(y_true, y_pred, labels=classes)
acc = accuracy_score(y_true, y_pred)

# Plot
sns.set_theme(style="white", font="DejaVu Sans")
fig, ax = plt.subplots(figsize=(10, 8), facecolor="#F5F7F2")
ax.set_facecolor("#F5F7F2")

sns.heatmap(
    cm, annot=True, fmt="d", cmap="YlGn",
    linewidths=2, linecolor="#F5F7F2", square=True,
    xticklabels=classes, yticklabels=classes,
    cbar_kws={"label": "Number of test examples"},
    annot_kws={"size": 13, "weight": "bold"},
    ax=ax
)
ax.set_title("Crop Classifier · Confusion Matrix", loc="left",
             fontsize=20, fontweight="bold", color="#19392F", pad=24)
ax.text(0, 1.02,
        f"Synthetic test set · {len(y_true)} examples · Accuracy {acc:.1%}",
        transform=ax.transAxes, fontsize=11, color="#687A72", va="bottom")
ax.set_xlabel("Predicted crop", fontsize=12, color="#344A40", labelpad=12)
ax.set_ylabel("Actual crop", fontsize=12, color="#344A40", labelpad=12)
ax.tick_params(axis="x", rotation=0, labelsize=10)
ax.tick_params(axis="y", rotation=0, labelsize=10)
for spine in ax.spines.values():
    spine.set_visible(False)

fig.text(0.12, 0.025,
         "Illustrative simulated predictions only — not measured AgriVision performance.",
         fontsize=9, color="#B65342")
plt.tight_layout(rect=[0.02, 0.06, 1, 1])

out = "/kaggle/working/crop_confusion_matrix_demo.png"
plt.savefig(out, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()

print(f"Accuracy: {acc:.1%}")
print("\nPer-class precision / recall / F1:")
print(classification_report(y_true, y_pred, labels=classes, zero_division=0))
print("Saved:", out)

