"""Create AgriVision presentation charts from recorded metrics and seeded dummy data.

The synthetic benchmark and fertilizer example are demonstrations only. They are
not evaluations or recommendations produced by the deployed AgriVision models.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import xml.sax.saxutils as xml

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
META_PATH = ROOT / "model" / "agrivision_onnx_metadata.json"
OUT = ROOT / "docs" / "assets" / "ml_charts"
OUT.mkdir(parents=True, exist_ok=True)

BG = "#F7F6EF"
INK = "#163A2A"
GREEN = "#26734E"
BLUE = "#4E84C4"
GOLD = "#E0AF3D"
PURPLE = "#8A67C7"
MUTED = "#65766C"
LINE = "#DCE3D9"
RED = "#BE695A"


def esc(value: object) -> str:
    return xml.escape(str(value))


def svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" rx="24" fill="{BG}"/>',
        f'<text x="54" y="62" font-family="Arial,sans-serif" font-size="29" font-weight="700" fill="{INK}">{esc(title)}</text>',
    ]


def text(parts: list[str], x: float, y: float, value: object, size=17, color=INK, weight=400, anchor="start"):
    parts.append(
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}">{esc(value)}</text>'
    )


def save_svg(name: str, parts: list[str]):
    parts.append("</svg>")
    (OUT / name).write_text("\n".join(parts), encoding="utf-8")


def actual_metrics_chart(meta: dict):
    metrics = meta["yield_model_metrics"]
    parts = svg_start(1200, 620, "AgriVision ONNX model · recorded holdout evaluation")
    text(parts, 55, 111, "Random Forest yield regression · values from shipped model metadata", 17, MUTED)

    cards = [
        (55, "MAE", f'{metrics["holdout_mae_t_ha"]:.2f}', "tonnes per hectare", GREEN),
        (440, "R²", f'{metrics["holdout_r2"]:.3f}', "coefficient of determination", BLUE),
        (825, "Holdout", f'{metrics["holdout_rows"]:,}', "rows evaluated", GOLD),
    ]
    for x, label, value, caption, color in cards:
        parts.append(f'<rect x="{x}" y="165" width="320" height="210" rx="18" fill="#FFFFFF" stroke="{LINE}" stroke-width="2"/>')
        parts.append(f'<rect x="{x}" y="165" width="320" height="9" rx="4" fill="{color}"/>')
        text(parts, x+24, 215, label, 16, color, 700)
        text(parts, x+24, 292, value, 55, INK, 700)
        text(parts, x+24, 338, caption, 16, MUTED)

    parts.append(f'<rect x="55" y="418" width="1090" height="138" rx="18" fill="#EAF1E8"/>')
    text(parts, 82, 461, "Use for this regression model", 17, GREEN, 700)
    text(parts, 82, 503, "MAE · RMSE · R²", 25, INK, 700)
    text(parts, 650, 461, "Not applicable to continuous yield prediction", 17, RED, 700)
    text(parts, 650, 503, "Accuracy · F1-score", 25, INK, 700)
    text(parts, 55, 593, f'Training rows: {metrics["training_rows"]:,}   ·   RMSE is not recorded in the project metadata.', 15, MUTED)
    save_svg("actual_model_metrics.svg", parts)


def make_dummy_data(seed=2026, n=900):
    rng = np.random.default_rng(seed)
    # Synthetic agronomic-like features and nonlinear yield response; deliberately
    # generated only to demonstrate regression comparison charts.
    n_level = rng.uniform(15, 150, n)
    p_level = rng.uniform(5, 90, n)
    k_level = rng.uniform(5, 100, n)
    temperature = rng.uniform(15, 38, n)
    rainfall = rng.uniform(150, 1800, n)
    ph = rng.uniform(4.5, 8.5, n)
    X = np.column_stack([n_level, p_level, k_level, temperature, rainfall, ph])
    yield_t_ha = (
        1.2
        + 4.0 * (1 - np.exp(-n_level / 70))
        + 1.8 * (1 - np.exp(-p_level / 38))
        + 1.2 * (1 - np.exp(-k_level / 48))
        - 0.020 * (temperature - 25) ** 2
        - 0.0000025 * (rainfall - 820) ** 2
        - 0.55 * (ph - 6.4) ** 2
        + 0.7 * np.sin(n_level / 28) * np.cos(rainfall / 420)
        + rng.normal(0, 0.35, n)
    )
    return X, np.maximum(0.05, yield_t_ha)


def fit_linear(X, y):
    design = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(design, y, rcond=None)[0]


def predict_linear(X, weights):
    return np.column_stack([np.ones(len(X)), X]) @ weights


def build_tree(X, y, rng, depth=0, max_depth=7, min_leaf=10, feature_count=None):
    value = float(np.mean(y))
    if depth >= max_depth or len(y) < 2 * min_leaf or np.var(y) < 1e-10:
        return ("leaf", value)
    feature_total = X.shape[1]
    if feature_count is None or feature_count >= feature_total:
        features = np.arange(feature_total)
    else:
        features = rng.choice(feature_total, feature_count, replace=False)
    best = None
    best_sse = float("inf")
    for feature in features:
        order = np.argsort(X[:, feature], kind="mergesort")
        xs, ys = X[order, feature], y[order]
        csum = np.cumsum(ys)
        csum2 = np.cumsum(ys * ys)
        lo, hi = min_leaf, len(y) - min_leaf
        if hi < lo:
            continue
        positions = np.arange(lo, hi + 1)
        distinct = xs[positions - 1] < xs[positions]
        positions = positions[distinct]
        if len(positions) == 0:
            continue
        left_n = positions
        right_n = len(y) - positions
        left_sum = csum[positions - 1]
        right_sum = csum[-1] - left_sum
        left_sq = csum2[positions - 1]
        right_sq = csum2[-1] - left_sq
        sse = left_sq - left_sum**2/left_n + right_sq - right_sum**2/right_n
        idx = int(np.argmin(sse))
        if float(sse[idx]) < best_sse:
            pos = int(positions[idx])
            threshold = float((xs[pos - 1] + xs[pos]) / 2)
            best = (int(feature), threshold, order[:pos], order[pos:])
            best_sse = float(sse[idx])
    if best is None:
        return ("leaf", value)
    feature, threshold, left_idx, right_idx = best
    left = build_tree(X[left_idx], y[left_idx], rng, depth+1, max_depth, min_leaf, feature_count)
    right = build_tree(X[right_idx], y[right_idx], rng, depth+1, max_depth, min_leaf, feature_count)
    return ("node", feature, threshold, left, right)


def predict_tree(tree, X):
    out = np.empty(len(X), dtype=float)
    def visit(node, indices):
        if not len(indices):
            return
        if node[0] == "leaf":
            out[indices] = node[1]
            return
        _, feature, threshold, left, right = node
        mask = X[indices, feature] <= threshold
        visit(left, indices[mask])
        visit(right, indices[~mask])
    visit(tree, np.arange(len(X)))
    return out


def fit_forest(X, y, seed=7, trees=35):
    rng = np.random.default_rng(seed)
    models = []
    feature_count = max(2, int(math.sqrt(X.shape[1])))
    for _ in range(trees):
        sample = rng.integers(0, len(X), len(X))
        models.append(build_tree(X[sample], y[sample], rng, max_depth=8, min_leaf=9, feature_count=feature_count))
    return models


def predict_forest(models, X):
    return np.mean([predict_tree(tree, X) for tree in models], axis=0)


def score(y, prediction):
    residual = y - prediction
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    r2 = float(1 - np.sum(residual**2) / np.sum((y - np.mean(y))**2))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def synthetic_comparison_chart(results, holdout_count):
    names = list(results)
    parts = svg_start(1380, 800, "Synthetic regression benchmark · demonstration only")
    text(parts, 58, 110, f"Seeded dummy dataset · {holdout_count} synthetic holdout rows · lower MAE/RMSE is better", 16, MUTED)
    colors = [MUTED, BLUE, PURPLE, GREEN]
    left, maxbar, top, row_h = 270, 485, 205, 112
    max_error = max(results[n]["RMSE"] for n in names) * 1.12
    text(parts, left, 165, "Error (t/ha)", 15, MUTED, 700)
    text(parts, 895, 165, "R²", 15, MUTED, 700)
    text(parts, 1095, 165, "MAE", 15, MUTED, 700)
    text(parts, 1220, 165, "RMSE", 15, MUTED, 700)
    for i, name in enumerate(names):
        y = top + i*row_h
        color = colors[i]
        text(parts, 60, y+35, name, 18, INK, 700)
        mae = results[name]["MAE"]
        rmse = results[name]["RMSE"]
        r2 = results[name]["R2"]
        # Two bars share the same t/ha scale.
        parts.append(f'<rect x="{left}" y="{y}" width="{maxbar}" height="22" rx="9" fill="#E5E9E1"/>')
        parts.append(f'<rect x="{left}" y="{y}" width="{maxbar*mae/max_error:.1f}" height="22" rx="9" fill="{color}"/>')
        parts.append(f'<rect x="{left}" y="{y+30}" width="{maxbar}" height="22" rx="9" fill="#E5E9E1"/>')
        parts.append(f'<rect x="{left}" y="{y+30}" width="{maxbar*rmse/max_error:.1f}" height="22" rx="9" fill="{color}" opacity=".52"/>')
        text(parts, left+maxbar+18, y+18, "MAE", 12, MUTED, 700)
        text(parts, left+maxbar+18, y+48, "RMSE", 12, MUTED, 700)
        text(parts, 895, y+34, f"{r2:.3f}", 18, color, 700)
        text(parts, 1095, y+34, f"{mae:.2f}", 17, INK, 700)
        text(parts, 1220, y+34, f"{rmse:.2f}", 17, INK, 700)
    parts.append('<line x1="58" y1="660" x2="1320" y2="660" stroke="#DCE3D9" stroke-width="2"/>')
    text(parts, 60, 704, "These scores compare algorithms trained and tested on generated dummy data only.", 16, RED, 700)
    text(parts, 60, 740, "They are not results from the shipped AgriVision ONNX model and must not be presented as its accuracy.", 15, MUTED)
    save_svg("synthetic_model_comparison.svg", parts)


def fertilizer_example_chart(meta):
    # Illustrative target-balancing example derived from model metadata nutrient
    # targets, product nutrient fractions and embedded product costs.
    start = {"N": 40.0, "P": 20.0, "K": 15.0}
    target = meta["fertilizer_safe_targets_npk"]
    products = {p["name"]: p for p in meta["fertilizer_products"]}
    deficits = {k: max(0.0, float(target[k]) - start[k]) for k in ("N", "P", "K")}
    # For this standalone illustration, nutrient fractions are interpreted
    # conventionally: 46% N means 0.46 kg N per 1 kg product.
    dap = deficits["P"] / products["DAP"]["p"]
    urea = max(0.0, deficits["N"] - dap * products["DAP"]["n"]) / products["Urea"]["n"]
    mop = deficits["K"] / products["MOP"]["k"]
    rates = {"Urea": urea, "DAP": dap, "MOP": mop}
    costs = {name: rates[name] * products[name]["cost"] for name in rates}
    total_cost = sum(costs.values())

    parts = svg_start(1240, 700, "Illustrative fertilizer target-balancing example")
    text(parts, 56, 108, "Dummy soil inputs · manually calculated to illustrate nutrient gaps and product-rate math", 16, MUTED)
    labels = [("Nitrogen (N)", "N", BLUE), ("Phosphorus (P)", "P", PURPLE), ("Potassium (K)", "K", GOLD)]
    x0, y0, scale = 240, 180, 5.2
    text(parts, x0, 150, "Soil level / target (metadata units)", 15, MUTED, 700)
    for i, (label, key, color) in enumerate(labels):
        y = y0+i*92
        text(parts, 58, y+27, label, 17, INK, 700)
        parts.append(f'<rect x="{x0}" y="{y}" width="{scale*100:.1f}" height="24" rx="8" fill="#E4E8E1"/>')
        parts.append(f'<rect x="{x0}" y="{y}" width="{scale*float(target[key]):.1f}" height="24" rx="8" fill="{color}" opacity=".28"/>')
        parts.append(f'<rect x="{x0}" y="{y}" width="{scale*start[key]:.1f}" height="24" rx="8" fill="{color}"/>')
        text(parts, x0+scale*start[key]+12, y+19, f'Start {start[key]:.0f}', 13, INK, 700)
        text(parts, x0+scale*float(target[key])-8, y+51, f'Target {float(target[key]):.0f}', 13, MUTED, 700, "end")

    parts.append(f'<rect x="820" y="150" width="360" height="300" rx="18" fill="#FFFFFF" stroke="{LINE}" stroke-width="2"/>')
    text(parts, 850, 190, "Illustrative product rates", 17, GREEN, 700)
    for i, name in enumerate(rates):
        y = 235+i*58
        color = [BLUE, PURPLE, GOLD][i]
        text(parts, 850, y, name, 16, INK, 700)
        text(parts, 1150, y, f"{rates[name]:.1f} kg/ha", 15, color, 700, "end")
        text(parts, 850, y+22, f"₹{costs[name]:,.0f}/ha estimated input cost", 13, MUTED)

    parts.append('<rect x="58" y="500" width="1122" height="118" rx="18" fill="#FFF4D8"/>')
    text(parts, 82, 542, f"Illustrative product cost ≈ ₹{total_cost:,.0f}/ha", 21, INK, 700)
    text(parts, 82, 580, "Not generated by the website optimizer and not an agronomic recommendation.", 16, RED, 700)
    text(parts, 58, 660, "Rates use embedded nutrient fractions and product cost assumptions; verify with a current soil test and local guidance.", 14, MUTED)
    save_svg("illustrative_fertilizer_example.svg", parts)
    return {"starting_soil": start, "targets": {k: float(v) for k,v in target.items()}, "rates_kg_ha": rates, "costs_inr_ha": costs, "total_cost_inr_ha": total_cost}


def main():
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    actual_metrics_chart(meta)
    X, y = make_dummy_data()
    rng = np.random.default_rng(123)
    indices = rng.permutation(len(y))
    split = int(0.8 * len(y))
    train_idx, test_idx = indices[:split], indices[split:]
    x_train, y_train = X[train_idx], y[train_idx]
    x_test, y_test = X[test_idx], y[test_idx]

    baseline = np.full(len(y_test), float(np.mean(y_train)))
    linear = predict_linear(x_test, fit_linear(x_train, y_train))
    tree = predict_tree(build_tree(x_train, y_train, np.random.default_rng(9), max_depth=7, min_leaf=10), x_test)
    forest = predict_forest(fit_forest(x_train, y_train), x_test)
    comparison = {
        "Mean baseline": score(y_test, baseline),
        "Linear regression": score(y_test, linear),
        "Decision tree": score(y_test, tree),
        "Random forest": score(y_test, forest),
    }
    synthetic_comparison_chart(comparison, len(test_idx))
    fertilizer = fertilizer_example_chart(meta)
    summary = {
        "actual_model_metrics": meta["yield_model_metrics"],
        "synthetic_dummy_data": {"seed": 2026, "rows": len(y), "holdout_rows": len(test_idx), "model_scores": comparison},
        "fertilizer_dummy_example": fertilizer,
        "notice": "Synthetic comparison and fertilizer illustration are examples only; they are not performance or recommendations from the shipped model."
    }
    (OUT / "chart_data.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Charts written to {OUT}")
    print(json.dumps({"synthetic_scores": comparison, "fertilizer_example": fertilizer}, indent=2))

if __name__ == "__main__":
    main()
