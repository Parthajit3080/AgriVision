import json
import unittest

from backend.agrivision_onnx_inference import _simulate_annual_climate


class ClimateProjectionTests(unittest.TestCase):
    def setUp(self):
        self.history = {
            "last_year": 2020,
            "means": {"temperature": 26.7, "rainfall": 1200.0, "humidity": 67.0},
            "stds": {"temperature": 0.5, "rainfall": 300.0, "humidity": 4.0},
        }
        self.fallback_stds = {"temperature": 0.45, "rainfall": 341.0, "humidity": 3.3}

    def test_series_is_reproducible_and_horizon_stable(self):
        short = _simulate_annual_climate("Maharashtra", list(range(2021, 2028)),
                                         self.history, self.fallback_stds)
        long = _simulate_annual_climate("Maharashtra", list(range(2021, 2031)),
                                        self.history, self.fallback_stds)
        self.assertEqual(short, {year: long[year] for year in short})
        again = _simulate_annual_climate("Maharashtra", list(range(2021, 2028)),
                                         self.history, self.fallback_stds)
        self.assertEqual(short, again)

    def test_rainfall_and_humidity_vary_across_years(self):
        series = _simulate_annual_climate("Maharashtra", list(range(2021, 2031)),
                                          self.history, self.fallback_stds)
        for variable in ("rainfall", "humidity"):
            values = [round(row[variable], 2) for row in series.values()]
            self.assertGreater(len(set(values)), 5, variable)

    def test_values_stay_in_physical_ranges_and_use_state_baseline(self):
        series = _simulate_annual_climate("Maharashtra", list(range(2021, 2041)),
                                          self.history, self.fallback_stds)
        for row in series.values():
            self.assertGreaterEqual(row["temperature"], 0)
            self.assertLessEqual(row["temperature"], 50)
            self.assertGreaterEqual(row["rainfall"], 0)
            self.assertGreaterEqual(row["humidity"], 0)
            self.assertLessEqual(row["humidity"], 100)
        average_humidity = sum(row["humidity"] for row in series.values()) / len(series)
        self.assertLess(abs(average_humidity - self.history["means"]["humidity"]),
                        2 * self.history["stds"]["humidity"])

    def test_prediction_request_and_output_contract(self):
        from backend.app import PredictRequest, model

        request = PredictRequest.model_validate({
            "location": {"state": "Maharashtra", "district": "Pune"},
            "season": "Kharif",
            "land_size": 1,
            "land_unit": "acres",
            "soil_test": {"N": 70, "P": 45, "K": 45, "ph": 6.5},
            "risk_tolerance": "medium",
            "target_year": 2028,
        })
        report = model.predict(request.model_dump())
        self.assertEqual(
            set(report),
            {"crop_profit_ranking", "weather_climate_risk", "fertilizer_nutrient_plan", "inputs", "location_match"},
        )
        climate = report["weather_climate_risk"]
        self.assertIn("simulated", climate["forecast_method"])
        self.assertEqual(len(climate["seasonal_outlook"]), 24)
        rainfall = [row["forecast"] for row in climate["forecast"] if row["variable"] == "rainfall"]
        humidity = [row["forecast"] for row in climate["forecast"] if row["variable"] == "humidity"]
        self.assertGreater(len(set(round(value, 2) for value in rainfall)), 1)
        self.assertGreater(len(set(round(value, 2) for value in humidity)), 1)
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
