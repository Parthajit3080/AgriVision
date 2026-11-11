# AgriVision website inference bundle

## Files

- `agrivision_website_model.onnx`: crop-yield, crop-conditioned associational yield, and state-specific climate-trend heads.
- `agrivision_onnx_metadata.json`: compact crop labels, local feature defaults, climate summaries, regional mandi prices, and district yield profiles. It is not the training dataset.
- `agrivision_onnx_inference.py`: website/backend adapter that returns the stable AgriVision JSON structure.
- `export_agrivision_onnx.py`: one-time training and export script; it reads the CSVs only while rebuilding the model.

At website startup, create one `AgriVisionONNX` instance and reuse it for requests. `predict()` accepts the same farmer-input fields as `USER_INPUTS` and returns the report dictionary. For Kaggle, the command-line adapter writes `/kaggle/working/agrivision_outputs.json` by default. Pass `--input user_inputs.json` if the inputs are stored in a file.

```python
from agrivision_onnx_inference import AgriVisionONNX

model = AgriVisionONNX()  # load the ONNX file once when the backend starts
report = model.predict({
    "location": {"state": "Maharashtra", "district": "Pune"},
    "season": "Kharif",
    "land_size": 2,
    "land_unit": "acres",
    "soil_test": {"N": 70, "P": 45, "K": 45, "ph": 6.5},
})
```

Install runtime dependencies with `pip install -r requirements_onnx.txt`. The inference adapter does not read or require the CSV files. Keep the ONNX file and metadata JSON together. Refresh the metadata/model when you need newer mandi prices or updated training data.

ONNX emits numeric predictions; the adapter converts them to the website JSON format. The chronological 2016–2020 holdout MAE is 1.51 t/ha for the global model and 1.42 t/ha for the per-crop selected estimator, compared with 2.30 t/ha for the crop-median baseline. A crop uses the ONNX estimate only when its own holdout MAE beats its median baseline by at least 5%; otherwise the output labels that crop as a median fallback and includes its holdout MAE. `input_diagnostics` flags farmer-entered values outside the central training range.

Production costs are absent from the training data, so send `crop_cost_per_ha` to calculate profit; without costs, crops are ordered by estimated yield and are not described as the most profitable. Suitability probabilities are null because the source contains no labeled suitability outcome. The available weather data are annual and state-level, so district and season do not change climate projections. Year-to-year climate variation is a reproducible bootstrap of observed state anomalies added to each state's trend; it is a scenario projection, not a year-specific weather forecast. District yield history and mandi prices are used when exact regional records are available. Fertilizer contributions are recalculated from each displayed product rate and its N/P/K fractions, so the per-product contribution rows sum to the reported totals. Rates use an empirical nutrient-target cost proxy because validated crop-specific fertilizer dosage and treatment-response labels are missing; they are not prescriptions.
