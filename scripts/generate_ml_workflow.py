"""Build an editable two-page draw.io workflow from the saved ONNX training/runtime code."""
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parents[1] / "docs" / "AgriVision_ML_Workflow.drawio"
PALETTE = {
    "data": ("#DCEBFF", "#5486C4", "#183C68"),
    "crop": ("#E9E0FF", "#8560CB", "#412A71"),
    "fert": ("#FFF0C9", "#D4A128", "#654C0A"),
    "climate": ("#DDF5E7", "#3D9B68", "#164C31"),
    "artifact": ("#FFE1E7", "#CC6475", "#662734"),
    "neutral": ("#E9EFF0", "#66818A", "#2F4D55"),
}

class Diagram:
    def __init__(self, mx, name, page_id, width, height, prefix):
        d = ET.SubElement(mx, "diagram", {"id": page_id, "name": name})
        model = ET.SubElement(d, "mxGraphModel", {
            "dx": str(width), "dy": str(height), "grid": "1", "gridSize": "10",
            "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
            "fold": "1", "page": "1", "pageScale": "1", "pageWidth": str(width),
            "pageHeight": str(height), "math": "0", "shadow": "0"
        })
        self.root = ET.SubElement(model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})
        self.prefix, self.i, self.ids = prefix, 2, {}

    def id(self):
        result = f"{self.prefix}{self.i}"
        self.i += 1
        return result

    def lane(self, title, x, y, w, h, fill, stroke):
        style = f"swimlane;html=1;startSize=38;horizontal=1;rounded=1;fillColor={fill};swimlaneFillColor=#FFFFFF;strokeColor={stroke};strokeWidth=1.5;fontStyle=1;fontSize=16;fontColor=#20364D;"
        c = ET.SubElement(self.root, "mxCell", {"id": self.id(), "value": title, "style": style, "vertex": "1", "parent": "1"})
        ET.SubElement(c, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})

    def box(self, key, title, subtitle, x, y, w, h, theme, shape="rounded=1"):
        fill, stroke, fg = PALETTE[theme]
        style = f"{shape};html=1;whiteSpace=wrap;overflow=hidden;spacing=8;fillColor={fill};strokeColor={stroke};strokeWidth=2;fontColor={fg};align=center;verticalAlign=middle;fontSize=16;fontStyle=1;"
        value = f"<b>{title}</b>" + (f'<br><font style="font-size:12px">{subtitle}</font>' if subtitle else "")
        c = ET.SubElement(self.root, "mxCell", {"id": self.id(), "value": value, "style": style, "vertex": "1", "parent": "1"})
        ET.SubElement(c, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})
        self.ids[key] = c.get("id")

    def note(self, value, x, y, w, h, color="#52675A", size=13):
        style = f"text;html=1;strokeColor=none;fillColor=none;fontSize={size};fontColor={color};align=left;verticalAlign=middle;whiteSpace=wrap;"
        c = ET.SubElement(self.root, "mxCell", {"id": self.id(), "value": value, "style": style, "vertex": "1", "parent": "1"})
        ET.SubElement(c, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})

    def edge(self, source, target, label="", color="#526E5D", dashed=False):
        style = f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor={color};strokeWidth=2.2;fontSize=12;fontStyle=1;fontColor=#40594A;labelBackgroundColor=#FFFFFF;"
        if dashed:
            style += "dashed=1;dashPattern=6 4;"
        c = ET.SubElement(self.root, "mxCell", {
            "id": self.id(), "value": label, "style": style, "edge": "1", "parent": "1",
            "source": self.ids[source], "target": self.ids[target]
        })
        ET.SubElement(c, "mxGeometry", {"relative": "1", "as": "geometry"})

def training_page(d):
    d.note("AGRIVISION  ·  TRAINING → SAVED MODEL", 45, 24, 1000, 42, "#173A2B", 25)
    d.note("Verified from export_agrivision_onnx.py and the saved website ONNX bundle", 48, 68, 1300, 28, "#52675A", 15)
    d.lane("TRAINING DATA + FEATURES", 35, 120, 420, 575, "#F5F9FF", "#9BBDE9")
    d.lane("THREE TRAINED OUTPUT HEADS", 470, 120, 875, 575, "#FBFAF5", "#D4C48D")
    d.lane("EXPORT + SAVED ARTIFACT", 1360, 120, 530, 575, "#FFF6F7", "#E1A0AA")

    d.box("yield_data", "Crop yield data", "Crop identity + soil/weather features", 65, 195, 355, 90, "data")
    d.box("fert_data", "Fertilizer/yield records", "Associational labels; not treatment-response trials", 65, 335, 355, 90, "data")
    d.box("climate_data", "Historical state weather", "State + year features; temperature, rain, humidity", 65, 475, 355, 90, "data")
    d.box("split", "Holdout evaluation", "Separate rows before final training", 65, 605, 355, 58, "neutral")

    d.box("rf", "Crop-yield model", "RandomForestRegressor · crop one-hot + numeric features", 505, 195, 375, 95, "crop")
    d.box("mlp", "Fertilizer response model", "MLPRegressor (48, 24) · scikit-learn pipeline", 925, 335, 375, 95, "fert")
    d.box("lin", "Climate trend model", "LinearRegression · state + state × year features", 505, 475, 375, 95, "climate")
    d.box("validation", "Reported yield metrics", "Holdout MAE 1.194 t/ha · R² 0.936 · 3,904 rows", 925, 195, 375, 95, "neutral")
    d.box("no_classifier", "No suitability classifier", "Source data has no labeled crop-suitability outcome", 925, 475, 375, 95, "neutral")
    d.box("merge", "Merge ONNX heads", "crop_variable · fert_variable · climate_variable", 1390, 220, 235, 105, "artifact")
    d.box("saved", "Saved website model", "agrivision_website_model.onnx", 1640, 220, 220, 105, "artifact", "shape=hexagon")
    d.box("metadata", "Metadata JSON", "Labels · defaults · climate summaries · metrics", 1390, 410, 470, 85, "data")
    d.box("runtime", "Deploy bundle", "ONNX + metadata · inference only at request time", 1390, 555, 470, 80, "neutral")

    d.edge("yield_data", "rf", "fit crop head", "#8560CB")
    d.edge("fert_data", "mlp", "fit response head", "#D4A128")
    d.edge("climate_data", "lin", "fit trend head", "#3D9B68")
    d.edge("yield_data", "validation", "holdout metrics", "#66818A", True)
    d.edge("split", "rf", "fit after evaluation split", "#8560CB", True)
    d.edge("rf", "merge", "convert to ONNX", "#8560CB")
    d.edge("mlp", "merge", "convert to ONNX", "#D4A128")
    d.edge("lin", "merge", "convert to ONNX", "#3D9B68")
    d.edge("merge", "saved", "export", "#CC6475")
    d.edge("saved", "runtime", "saved artifact", "#CC6475")
    d.edge("metadata", "runtime", "pack together", "#5486C4")
    d.note("Model-name check: crop output is a RandomForestRegressor, not a suitability classifier; no LightGBM or climate Random Forest appears in the saved ONNX export.", 55, 730, 1780, 30, "#A24F47", 13)
    d.note("A separate full-Python implementation uses per-crop HistGradientBoostingRegressor, ARIMA climate forecasts, and MLP + genetic search. This diagram follows the website ONNX export/runtime.", 55, 765, 1780, 30, "#52675A", 12)

def runtime_page(d):
    d.note("AGRIVISION  ·  USER INPUT → PREDICTION", 45, 24, 1000, 42, "#173A2B", 25)
    d.note("The saved ONNX bundle is loaded by the API; runtime branches below show what the website actually does", 48, 68, 1600, 28, "#52675A", 15)
    d.lane("REQUEST PATH", 35, 120, 1845, 185, "#F5F8FC", "#9BBDE9")
    d.box("user", "User inputs", "Location · crop/season · soil · budgets · preferences", 70, 180, 285, 82, "data", "ellipse")
    d.box("web", "Frontend", "Farm planner form → JSON request", 450, 180, 300, 82, "crop")
    d.box("api", "FastAPI validation", "Validate fields; read session if present", 850, 180, 330, 82, "climate")
    d.box("onnx", "ONNX Runtime", "Model session loads once and is reused", 1280, 180, 320, 82, "artifact", "shape=hexagon")
    d.edge("user", "web", "submit inputs", "#5486C4")
    d.edge("web", "api", "POST /api/predict", "#8560CB")
    d.edge("api", "onnx", "validated features", "#3D9B68")

    d.lane("CROP-YIELD + CROP RANKING", 35, 335, 1845, 225, "#F8F5FF", "#B6A2E4")
    d.box("crop_features", "Crop feature vector", "Crop one-hot + N/P/K + pH + weather", 70, 410, 300, 90, "crop")
    d.box("crop_head", "RandomForestRegressor", "ONNX crop_variable output", 455, 410, 300, 90, "crop")
    d.box("yield", "Yield estimate", "Repeat for eligible crop candidates · t/ha", 840, 410, 310, 90, "climate")
    d.box("rank", "Rank crop options", "Predicted yield × embedded crop price = gross revenue", 1240, 410, 400, 90, "artifact")
    d.edge("onnx", "crop_features", "crop inputs", "#8560CB", True)
    d.edge("crop_features", "crop_head", "feature tensor", "#8560CB")
    d.edge("crop_head", "yield", "predicted yield", "#3D9B68")
    d.edge("yield", "rank", "post-processing", "#CC6475")

    d.lane("FERTILIZER PLAN", 35, 585, 1845, 225, "#FFFBF0", "#E0C477")
    d.box("fert_inputs", "Soil + plan constraints", "N/P/K · pH · crop · budget · allowed products", 70, 660, 300, 90, "data")
    d.box("fert_head", "MLPRegressor", "ONNX fert_variable estimates yield response", 455, 660, 300, 90, "fert")
    d.box("optimizer", "Coordinate-search optimizer", "3 passes · cost + nutrient guardrails + budget", 840, 660, 310, 90, "fert")
    d.box("fert_output", "Fertilizer plan", "Product rates + estimated cost", 1240, 660, 400, 90, "climate")
    d.edge("onnx", "fert_inputs", "soil + optimizer inputs", "#D4A128", True)
    d.edge("fert_inputs", "fert_head", "candidate features", "#D4A128")
    d.edge("fert_head", "optimizer", "candidate yield scores", "#D4A128")
    d.edge("optimizer", "fert_output", "selected rates", "#3D9B68")
    d.edge("rank", "fert_inputs", "selected crop", "#8560CB", True)

    d.lane("CLIMATE + RISK CONTEXT", 35, 835, 1845, 230, "#F1FAF3", "#9AD2AE")
    d.box("clim_inputs", "State + year + season", "Manual weather may override scenario values", 70, 910, 300, 90, "data")
    d.box("clim_metadata", "Historical state summaries", "Means + standard deviations in metadata JSON", 455, 910, 300, 90, "climate")
    d.box("simulator", "Scenario simulator", "Deterministic annual variability + seasonal heuristics", 840, 910, 310, 90, "climate")
    d.box("clim_output", "Climate + risk output", "Illustrative outlook + heuristic 0–100 risk index", 1240, 910, 400, 90, "artifact")
    d.edge("api", "clim_inputs", "state/year/season", "#3D9B68", True)
    d.edge("clim_inputs", "clim_metadata", "state lookup", "#3D9B68")
    d.edge("clim_metadata", "simulator", "historical context", "#3D9B68")
    d.edge("simulator", "clim_output", "scenario + exposure", "#3D9B68")
    d.edge("clim_output", "crop_features", "weather features", "#3D9B68", True)
    d.edge("clim_output", "fert_inputs", "scenario context", "#3D9B68", True)

    d.note("Climate distinction: a LinearRegression climate_variable head is present in the saved ONNX file, but the current website adapter does not call it; it uses metadata summaries + simulation. Climate values are not observed forecasts.", 55, 1090, 1780, 48, "#52675A", 13)
    d.note("The uploaded image’s MLP + GA matches the fuller Python optimizer; the deployed ONNX website uses an MLP response head plus coordinate search. No LightGBM or climate Random Forest was found in the saved website export.", 55, 1142, 1780, 42, "#A24F47", 13)

def main():
    mx = ET.Element("mxfile", {
        "host": "app.diagrams.net",
        "modified": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "agent": "Codex", "version": "24.7.17", "type": "device"
    })
    training_page(Diagram(mx, "1 - Training and saved model", "ml-training", 1940, 820, "t"))
    runtime_page(Diagram(mx, "2 - User input and predictions", "ml-runtime", 1940, 1200, "r"))
    ET.indent(mx, space="  ")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(mx).write(OUT, encoding="utf-8", xml_declaration=True)
    ET.parse(OUT)
    print(f"Updated {OUT}")

if __name__ == "__main__":
    main()
