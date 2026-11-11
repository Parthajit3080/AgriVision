"""Create two data-led AgriVision checkpoint visuals using only Python's standard library.

Reads the exact split and holdout metrics from model/agrivision_onnx_metadata.json
and writes reusable SVG visuals for the PowerPoint build.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "checkpoint_visuals"
OUT.mkdir(parents=True, exist_ok=True)
meta = json.loads((ROOT / "model" / "agrivision_onnx_metadata.json").read_text(encoding="utf-8"))
metrics = meta["yield_model_metrics"]
train = int(metrics["training_rows"])
holdout = int(metrics["holdout_rows"])
total = train + holdout
train_pct = train / total * 100
holdout_pct = holdout / total * 100

# Chart: real train / holdout composition from shipped model metadata.
w, h = 1200, 420
bar_x, bar_y, bar_w, bar_h = 72, 185, 1056, 62
train_w = bar_w * train / total
split_svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<rect width="1200" height="420" fill="#F5F4EB"/>
<text x="72" y="68" fill="#193B2C" font-family="Arial,sans-serif" font-size="30" font-weight="700">Evaluation data split</text>
<text x="72" y="112" fill="#62756A" font-family="Arial,sans-serif" font-size="19">Random Forest yield model · {total:,} records in total</text>
<rect x="{bar_x}" y="{bar_y}" width="{train_w:.2f}" height="{bar_h}" rx="12" fill="#216B4B"/>
<path d="M {bar_x+train_w:.2f} {bar_y} h {bar_w-train_w-12:.2f} a 12 12 0 0 1 12 12 v 38 a 12 12 0 0 1 -12 12 h -{bar_w-train_w-12:.2f} z" fill="#E2B44C"/>
<text x="{bar_x+train_w/2:.1f}" y="{bar_y+40}" text-anchor="middle" fill="#FFFFFF" font-family="Arial,sans-serif" font-size="22" font-weight="700">Train {train_pct:.0f}%</text>
<text x="{bar_x+train_w+(bar_w-train_w)/2:.1f}" y="{bar_y+40}" text-anchor="middle" fill="#193B2C" font-family="Arial,sans-serif" font-size="20" font-weight="700">{holdout_pct:.0f}%</text>
<circle cx="82" cy="318" r="8" fill="#216B4B"/><text x="104" y="325" fill="#193B2C" font-family="Arial,sans-serif" font-size="22" font-weight="700">{train:,} training rows</text>
<circle cx="575" cy="318" r="8" fill="#E2B44C"/><text x="597" y="325" fill="#193B2C" font-family="Arial,sans-serif" font-size="22" font-weight="700">{holdout:,} holdout rows</text>
<text x="72" y="384" fill="#62756A" font-family="Arial,sans-serif" font-size="17">Metrics shown separately: MAE  {metrics['holdout_mae_t_ha']:.2f} t/ha  ·  R²  {metrics['holdout_r2']:.3f}</text>
</svg>'''
(OUT / "holdout_split.svg").write_text(split_svg, encoding="utf-8")

# Cover illustration: original vector field rows, sun and growing crop. Decorative, not a real field image.
cover = '''<svg xmlns="http://www.w3.org/2000/svg" width="680" height="680" viewBox="0 0 680 680">
<defs><linearGradient id="sky" x2="0" y2="1"><stop stop-color="#244F3C"/><stop offset="1" stop-color="#10291F"/></linearGradient><linearGradient id="land" x2="0" y2="1"><stop stop-color="#9AC56B"/><stop offset="1" stop-color="#315D3B"/></linearGradient></defs>
<circle cx="470" cy="150" r="76" fill="#E1B953" opacity=".95"/>
<path d="M0 374 Q170 290 350 373 T680 355 V680 H0Z" fill="#2E6847"/>
<path d="M0 425 Q185 330 360 430 T680 410 V680 H0Z" fill="url(#land)"/>
<g fill="none" stroke="#D9E8B2" stroke-width="3" opacity=".62">
<path d="M-20 690 Q140 500 340 405"/><path d="M70 700 Q220 530 390 415"/><path d="M190 700 Q300 555 440 420"/><path d="M330 700 Q395 570 500 422"/><path d="M470 700 Q490 580 560 424"/><path d="M610 700 Q590 580 620 420"/>
</g>
<g stroke="#D9E8B2" stroke-width="5" fill="none" stroke-linecap="round">
<path d="M347 433 Q341 363 357 302 M357 357 Q323 342 309 315 M355 380 Q387 356 402 330"/><path d="M220 482 Q210 428 229 382 M227 444 Q199 429 190 410 M226 459 Q252 442 264 422"/><path d="M475 470 Q470 421 484 375 M480 439 Q454 424 447 407 M480 450 Q502 434 515 415"/>
</g>
<g fill="#D9E8B2"><ellipse cx="314" cy="311" rx="23" ry="10" transform="rotate(32 314 311)"/><ellipse cx="402" cy="328" rx="23" ry="10" transform="rotate(-30 402 328)"/><ellipse cx="192" cy="408" rx="18" ry="8" transform="rotate(30 192 408)"/><ellipse cx="262" cy="421" rx="18" ry="8" transform="rotate(-25 262 421)"/></g>
</svg>'''
(OUT / "cover_field.svg").write_text(cover, encoding="utf-8")
print(f"Created split visual: {train:,} train / {holdout:,} holdout; MAE {metrics['holdout_mae_t_ha']:.2f}; R2 {metrics['holdout_r2']:.3f}")

