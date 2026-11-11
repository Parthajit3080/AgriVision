"""Dataset-free AgriVision website inference using the exported ONNX bundle.

The adapter returns the same stable top-level JSON keys as the Kaggle pipeline.
Only NumPy, ONNX Runtime, the ONNX model, and compact metadata are needed at
request time; no training CSV is opened here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
from typing import Any, Optional

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("AGRIVISION_ONNX_PATH", HERE / "agrivision_website_model.onnx"))
METADATA_PATH = Path(os.environ.get("AGRIVISION_ONNX_METADATA", HERE / "agrivision_onnx_metadata.json"))
OUTPUT_PATH = Path(os.environ.get("AGRIVISION_JSON_PATH", "/kaggle/working/agrivision_outputs.json"))
ACRE_TO_HA = 0.4046856422
SOIL_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
FERT_FEATURES = ["N", "P", "K", "temperature", "rainfall", "ph"]


def _crop_key(value: Any) -> str:
    key = "".join(ch for ch in str(value).casefold() if ch.isalnum())
    return {"cashewnut": "cashewnuts", "cardamom": "cardamoms", "horsegram": "kulthihorsegram"}.get(key, key)


def _location_key(value: Any) -> str:
    key = "".join(ch for ch in str(value).casefold() if ch.isalnum())
    return {"chattisgarh": "chhattisgarh", "jammukashmir": "jammuandkashmir",
            "orissa": "odisha", "pondicherry": "puducherry", "uttaranchal": "uttarakhand"}.get(key, key)


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _normalize_inputs(raw: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Validate website inputs so invalid values cannot silently become model features."""
    values = dict(raw or {})
    values.setdefault("location", {"state": None, "district": None})
    if not isinstance(values["location"], dict):
        raise ValueError("location must be an object with optional state and district fields.")
    values["location"] = {k: (str(values["location"].get(k)).strip() or None) if values["location"].get(k) is not None else None
                          for k in ("state", "district")}
    values.setdefault("land_size", 1.0)
    values.setdefault("land_unit", "acres")
    values.setdefault("crop_budget_per_acre", None)
    values.setdefault("fertilizer_budget_per_acre", None)
    values.setdefault("risk_tolerance", "medium")
    values.setdefault("optimization_goal", "balance_roi")
    values.setdefault("climate_scenario", "normal")
    values.setdefault("confidence_level", .90)
    values.setdefault("target_year", 2027)
    values.setdefault("horizon_years", 1)
    values.setdefault("horizon_months", 12)
    values.setdefault("soil_test", {})
    values.setdefault("manual_weather", {})
    values.setdefault("crop_cost_per_ha", {})
    values.setdefault("fertilizer_prices_per_kg", {"Urea": 6.2, "DAP": 27.0, "MOP": 17.0, "NPK 15-15-15": 24.0})
    for key in ("soil_test", "manual_weather", "crop_cost_per_ha", "fertilizer_prices_per_kg"):
        if not isinstance(values.get(key), dict):
            raise ValueError(f"{key} must be an object.")
    for key in ("include_crops", "exclude_crops", "preferred_fertilizers", "excluded_fertilizers"):
        value = values.get(key, [])
        values[key] = [x.strip() for x in value.split(",") if x.strip()] if isinstance(value, str) else list(value or [])
    values["land_size"] = float(values["land_size"])
    if not math.isfinite(values["land_size"]) or values["land_size"] <= 0:
        raise ValueError("land_size must be a finite number greater than zero.")
    for key in ("crop_budget_per_acre", "fertilizer_budget_per_acre"):
        if values[key] is not None:
            values[key] = float(values[key])
            if not math.isfinite(values[key]) or values[key] < 0:
                raise ValueError(f"{key} must be finite and nonnegative.")
    for key, lower, upper in (("N", 0, 1000), ("P", 0, 1000), ("K", 0, 1000), ("ph", 3, 10)):
        if values["soil_test"].get(key) is not None:
            x = float(values["soil_test"][key])
            if not math.isfinite(x) or not lower <= x <= upper:
                unit = " kg/ha" if key in ("N", "P", "K") else ""
                raise ValueError(f"soil_test.{key} must be between {lower} and {upper}{unit}.")
            values["soil_test"][key] = x
    for key, lower, upper in (("temperature", -10, 60), ("rainfall", 0, 10000), ("humidity", 0, 100)):
        if values["manual_weather"].get(key) is not None:
            x = float(values["manual_weather"][key])
            if not math.isfinite(x) or not lower <= x <= upper:
                raise ValueError(f"manual_weather.{key} must be between {lower} and {upper}.")
            values["manual_weather"][key] = x
    for key, value in list(values["crop_cost_per_ha"].items()):
        x = float(value)
        if not math.isfinite(x) or x < 0:
            raise ValueError("crop_cost_per_ha values must be finite and nonnegative INR/ha.")
        values["crop_cost_per_ha"][key] = x
    for key, value in list(values["fertilizer_prices_per_kg"].items()):
        x = float(value)
        if not math.isfinite(x) or x <= 0:
            raise ValueError("fertilizer_prices_per_kg values must be finite and greater than zero.")
        values["fertilizer_prices_per_kg"][key] = x
    if values.get("price_override") is not None:
        prices = values["price_override"].values() if isinstance(values["price_override"], dict) else [values["price_override"]]
        for price in prices:
            if price is not None and (not math.isfinite(float(price)) or float(price) <= 0):
                raise ValueError("price_override values must be finite and greater than zero INR/tonne.")
    values["land_unit"] = "hectares" if str(values["land_unit"]).casefold().startswith("hect") else "acres"
    if values.get("season") is not None:
        values["season"] = str(values["season"]).strip() or None
    values["horizon_years"] = max(1, int(values["horizon_years"] or 1))
    values["horizon_months"] = max(1, int(values["horizon_months"] or 12))
    values["confidence_level"] = float(values["confidence_level"])
    if values["confidence_level"] > 1:
        values["confidence_level"] /= 100.0
    if not 0 < values["confidence_level"] < 1:
        raise ValueError("confidence_level must be a fraction between 0 and 1, such as 0.90.")
    values["risk_tolerance"] = str(values["risk_tolerance"]).strip().casefold()
    if values["risk_tolerance"] not in {"low", "medium", "high"}:
        raise ValueError("risk_tolerance must be low, medium, or high.")
    values["optimization_goal"] = str(values["optimization_goal"]).strip().casefold()
    if values["optimization_goal"] not in {"minimize_cost", "maximize_yield", "balance_roi"}:
        raise ValueError("optimization_goal must be minimize_cost, maximize_yield, or balance_roi.")
    values["climate_scenario"] = str(values["climate_scenario"]).strip().casefold()
    if values["climate_scenario"] not in {"normal", "drought-prone", "flood-prone"}:
        raise ValueError("climate_scenario must be normal, drought-prone, or flood-prone.")
    return values


class AgriVisionONNX:
    def __init__(self, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
        self.meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.inputs = {x.name for x in self.session.get_inputs()}
        self.outputs = {x.name for x in self.session.get_outputs()}
        self.crop_output = self.meta["crop_output"]
        self.fert_output = self.meta["fertilizer_output"]
        self.climate_output = self.meta["climate_output"]
        self.crops = self.meta["crops"]
        self.crop_index = {c: i for i, c in enumerate(self.crops)}
        self.states = self.meta["states"]
        self.state_index = {_location_key(s): i for i, s in enumerate(self.states)}
        self.crop_width = len(self.crops) + len(SOIL_FEATURES)
        self.fert_width = len(self.crops) + len(FERT_FEATURES)
        self.climate_width = len(self.states) * 2
        self.climate_year_center = float(self.meta.get("climate_year_center", 2008.5))

    def _dummy_inputs(self) -> dict[str, np.ndarray]:
        return {
            next(n for n in self.inputs if "crop_features" in n): np.zeros((1, self.crop_width), dtype=np.float32),
            next(n for n in self.inputs if "fertilizer_features" in n): np.zeros((1, self.fert_width), dtype=np.float32),
            next(n for n in self.inputs if "climate_features" in n): np.zeros((1, self.climate_width), dtype=np.float32),
        }

    def _predict_crop_yields(self, crops: list[str], features: dict[str, float]) -> np.ndarray:
        x = np.zeros((len(crops), self.crop_width), dtype=np.float32)
        for row, crop in enumerate(crops):
            x[row, self.crop_index[crop]] = 1.0
            x[row, len(self.crops):] = [features[c] for c in SOIL_FEATURES]
        feed = self._dummy_inputs()
        feed[next(n for n in feed if "crop_features" in n)] = x
        return np.maximum(0, self.session.run([self.crop_output], feed)[0].reshape(-1))

    def _predict_fertilizer_yield(self, rows: np.ndarray, crop: str) -> np.ndarray:
        crop_hot = np.zeros((len(rows), len(self.crops)), dtype=np.float32)
        crop_hot[:, self.crop_index[crop]] = 1.0
        model_rows = np.concatenate([crop_hot, np.asarray(rows, dtype=np.float32)], axis=1)
        feed = self._dummy_inputs()
        feed[next(n for n in feed if "fertilizer_features" in n)] = model_rows
        return np.maximum(0, self.session.run([self.fert_output], feed)[0].reshape(-1))

    def _predict_climate(self, state: Optional[str], years: list[int]) -> np.ndarray:
        x = np.zeros((len(years), self.climate_width), dtype=np.float32)
        if state is None:
            x[:, :len(self.states)] = 1.0 / max(1, len(self.states))
        else:
            index = self.state_index.get(_location_key(state))
            if index is None:
                raise ValueError(f"Unknown state '{state}'. Choose a state in metadata states.")
            x[:, index] = 1.0
        centered_years = np.asarray(years, dtype=np.float32) - self.climate_year_center
        state_hot = x[:, :len(self.states)].copy()
        x[:, len(self.states):] = state_hot * centered_years[:, None]
        feed = self._dummy_inputs()
        feed[next(n for n in feed if "climate_features" in n)] = x
        return self.session.run([self.climate_output], feed)[0]

    def _historical_climate_anomaly(self, state: Optional[str], year: int) -> np.ndarray:
        """Select a reproducible, same-state historical annual residual vector."""
        by_state = self.meta.get("climate_residuals_by_state", {})
        selected_state = next((name for name in by_state if state and _location_key(name) == _location_key(state)), None)
        residual_rows = by_state.get(selected_state, []) if selected_state else self.meta.get("pooled_climate_residuals", [])
        if not residual_rows:
            return np.zeros(3, dtype=np.float64)
        seed_text = f"agrivision-climate-v1|{_location_key(selected_state or 'regional')}|{int(year)}"
        seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
        index = seed % len(residual_rows)
        return np.asarray(residual_rows[index], dtype=np.float64)

    def _clip_climate_to_observed_range(self, state: Optional[str], values: np.ndarray) -> np.ndarray:
        """Keep trend-plus-residual values inside the observed state envelope."""
        by_state = self.meta.get("climate_projection_bounds_by_state", {})
        selected_state = next((name for name in by_state if state and _location_key(name) == _location_key(state)), None)
        if selected_state:
            bounds = by_state[selected_state]
            for index, name in enumerate(("temperature", "rainfall", "humidity")):
                values[index] = np.clip(values[index], bounds[name]["min"], bounds[name]["max"])
        return values

    def _fertilizer_plan(self, crop: str, soil: dict[str, float], climate: dict[str, float],
                         area_ha: float, inputs: dict[str, Any], price: Optional[float]) -> dict[str, Any]:
        products = [dict(p) for p in self.meta["fertilizer_products"]]
        local_prices = inputs.get("fertilizer_prices_per_kg") or {}
        for product in products:
            if product["name"] in local_prices and local_prices[product["name"]] is not None:
                product["cost"] = float(local_prices[product["name"]])
        excluded = {str(x).casefold() for x in inputs.get("excluded_fertilizers", [])}
        preferred = {str(x).casefold() for x in inputs.get("preferred_fertilizers", [])}
        products = [p for p in products if p["name"].casefold() not in excluded]
        if not products:
            raise ValueError("At least one fertilizer must remain after exclusions.")
        caps = np.array([p["max_kg_ha"] for p in products], dtype=np.float32)
        matrix = np.array([[p["n"], p["p"], p["k"]] for p in products], dtype=np.float64)
        costs = np.array([p["cost"] for p in products], dtype=np.float64)
        base = np.array([soil[x] for x in ["N", "P", "K"]], dtype=np.float64)
        crop_targets = self.meta.get("fertilizer_safe_targets_by_crop", {}).get(crop, self.meta["fertilizer_safe_targets_npk"])
        safe = np.array([crop_targets[x] for x in ["N", "P", "K"]], dtype=np.float32)
        base_row = np.array([soil["N"], soil["P"], soil["K"], climate["temperature"], climate["rainfall"], soil["ph"]], dtype=np.float32)
        baseline = float(self._predict_fertilizer_yield(base_row[None, :], crop)[0])
        budget_value = inputs.get("fertilizer_budget_per_acre")
        budget_ha = float(budget_value) / ACRE_TO_HA if budget_value is not None else None
        goal = str(inputs.get("optimization_goal", "balance_roi"))
        def evaluate_many(candidate: np.ndarray) -> np.ndarray:
            # Product compositions are nutrient fractions (0.46 means 46%), so
            # kg product/ha times the fraction already yields kg nutrient/ha.
            nutrients = base[None, :] + (candidate @ matrix)
            costs_total = candidate @ costs
            # No fertilizer-treatment response labels exist in this dataset. Use
            # nutrient-target distance and product cost to select rates; reserve
            # the MLP estimate for reporting only, not as a causal yield reward.
            target_distance = np.sum((nutrients - safe[None, :]) ** 2, axis=1)
            penalties = (10.0 * target_distance
                         + np.maximum(0, candidate.sum(axis=1) - 420) * 100)
            if goal in ("minimize_cost", "minimize cost", "cost"):
                scores = -costs_total - penalties
            elif goal in ("maximize_yield", "maximize yield", "yield"):
                scores = -target_distance - .0002 * costs_total - penalties
            else:
                # Balance ROI cannot be estimated from observational yields, so
                # use the lower-cost nutrient-target proxy and label that limit.
                scores = -costs_total - penalties
            for i, p in enumerate(products):
                if p["name"].casefold() in preferred:
                    scores += .02 * candidate[:, i]
            if budget_ha is not None:
                scores -= np.maximum(0, costs_total - budget_ha) * 100
            return scores

        genes = np.zeros(len(products), dtype=np.float32)
        # The former 12-point grid changed urea in 20 kg/ha jumps and DAP in
        # roughly 16 kg/ha jumps. Small crop-target or soil changes then
        # collapsed to the same plan. Search one kg/ha increments and iterate
        # to a stable coordinate-wise solution instead.
        levels = [np.unique(np.append(np.arange(0, int(math.floor(cap)) + 1, dtype=np.float32), cap))
                  for cap in caps]
        for _ in range(12):
            previous = genes.copy()
            for i in range(len(products)):
                candidates = np.tile(genes, (len(levels[i]), 1))
                candidates[:, i] = levels[i]
                scores = evaluate_many(candidates)
                genes[i] = candidates[int(np.argmax(scores)), i]
            if np.array_equal(previous, genes):
                break
        reported_genes = np.array([round(float(q), 2) if q > .1 else 0.0 for q in genes], dtype=np.float64)
        product_contributions = np.round(reported_genes[:, None] * matrix, 2)
        added_nutrients = product_contributions.sum(axis=0)
        final_nutrients = base + added_nutrients
        final_features = np.array([[final_nutrients[0], final_nutrients[1], final_nutrients[2],
                                    climate["temperature"], climate["rainfall"], soil["ph"]]], dtype=np.float32)
        pred_yield = float(self._predict_fertilizer_yield(final_features, crop)[0])
        plan = {p["name"]: float(q) for p, q in zip(products, reported_genes) if q > 0}
        contribution_breakdown = {
            p["name"]: {k: float(v) for k, v in zip(["N", "P", "K"], row)}
            for p, q, row in zip(products, reported_genes, product_contributions) if q > 0
        }
        cost = float(reported_genes @ costs)
        return {
            "crop": crop,
            "fertilizer_kg_ha": plan,
            "nutrient_contributions_by_product_kg_ha": contribution_breakdown,
            "total_product_kg_ha": round(float(reported_genes.sum()), 2),
            "estimated_cost_per_ha": round(cost, 2),
            "predicted_yield_t_ha": round(pred_yield, 3),
            "baseline_predicted_yield_t_ha": round(baseline, 3),
            "estimated_yield_gain_t_ha": round(pred_yield - baseline, 3),
            "final_soil_NPK": {k: round(float(v), 2) for k, v in zip(["N", "P", "K"], final_nutrients)},
            "starting_soil_NPK": {k: round(float(v), 2) for k, v in zip(["N", "P", "K"], base)},
            "added_soil_NPK": {k: round(float(v), 2) for k, v in zip(["N", "P", "K"], added_nutrients)},
            "guardrail_targets_NPK": {k: round(float(v), 2) for k, v in zip(["N", "P", "K"], safe)},
            "optimization_goal": goal,
            "requested_optimization_goal": goal,
            "budget_per_ha": budget_ha,
            "within_budget": budget_ha is None or cost <= budget_ha,
            "whole_farm_cost": cost * area_ha,
            "whole_farm_fertilizer_kg": {k: float(v) * area_ha for k, v in plan.items()},
            "guardrail_target_basis_note": "Targets use observed soil-data quantiles; the source does not contain validated crop-specific fertilizer-rate labels.",
            "yield_response_note": "Yield change is an associational model estimate; the source has no fertilizer-treatment labels, so it is not used to choose application rates.",
            "optimization_note": (
                "ROI cannot be estimated without production costs and fertilizer-treatment response labels; rates use a nutrient-target cost proxy."
                if goal == "balance_roi" else
                "Yield maximization cannot be inferred causally from these observational rows; rates use nutrient-target matching, not a yield-response objective."
                if goal == "maximize_yield" else
                "Rates minimize the supplied product costs while matching empirical nutrient targets; targets are not validated dosage prescriptions."
            ),
            "warning": "Estimated rates are not verified agronomic prescriptions. Confirm soil-test units, local crop guidance, and product labels before application.",
        }

    def predict(self, user_inputs: Optional[dict[str, Any]] = None,
                output_path: Optional[Path] = None) -> dict[str, Any]:
        inputs = _normalize_inputs(user_inputs)
        location = inputs.get("location") or {}
        state = location.get("state")
        state_meta = next((v for k, v in self.meta["state_climate_summary"].items()
                           if state and _location_key(k) == _location_key(state)), {})
        state_defaults = next((v for k, v in self.meta["state_feature_defaults"].items()
                           if state and _location_key(k) == _location_key(state)), self.meta["default_features"])
        features = dict(self.meta["default_features"])
        features.update(state_defaults)
        features.update({k: float(v) for k, v in (inputs.get("soil_test") or {}).items() if k in ("N", "P", "K", "ph") and v is not None})
        manual = inputs.get("manual_weather") or {}
        
        last_year = int(self.meta["last_training_weather_year"])
        target = inputs.get("target_year", 2027)
        horizon = max(1, int(inputs.get("horizon_years", 1) or 1))
        if target is None:
            years = list(range(last_year + 1, last_year + horizon + 1))
        else:
            target = int(target)
            years = list(range(last_year + 1, max(last_year + 1, target) + 1))
            if len(years) > 30:
                years = years[-30:]
        climate_raw = self._predict_climate(state, years)
        climate_by_year = {}
        scenario = str(inputs.get("climate_scenario", "normal")).casefold()
        for year, row in zip(years, climate_raw):
            projected = np.asarray(row, dtype=np.float64) + self._historical_climate_anomaly(state, year)
            projected = self._clip_climate_to_observed_range(state, projected)
            temp, rain, humidity = map(float, projected)
            if scenario == "drought-prone": temp += 1.0; rain *= .70
            elif scenario == "flood-prone": rain *= 1.30
            climate_by_year[year] = {"temperature": temp, "rainfall": max(0.0, rain), "humidity": float(np.clip(humidity, 0, 100))}
        chosen_climate = climate_by_year[years[-1]]
        features.update(chosen_climate)
        for k in ("temperature", "rainfall", "humidity"):
            if manual.get(k) is not None:
                features[k] = float(manual[k])

        candidate_crops = self.crops
        if state:
            match = next((v for k, v in self.meta["crops_by_state"].items() if _location_key(k) == _location_key(state)), None)
            if match is None:
                raise ValueError(f"No crop-history state match for '{state}'. Use a state listed in the source data.")
            candidate_crops = [c for c in candidate_crops if c in match]
        season = inputs.get("season")
        if season:
            match = next((v for k, v in self.meta["crops_by_season"].items() if k.casefold() == str(season).casefold()), None)
            if match is None:
                raise ValueError(f"No crop-history season match for '{season}'. Use a season listed in the source data.")
            candidate_crops = [c for c in candidate_crops if c in match]
        include = {_crop_key(x) for x in inputs.get("include_crops", [])}
        exclude = {_crop_key(x) for x in inputs.get("exclude_crops", [])}
        if include: candidate_crops = [c for c in candidate_crops if _crop_key(c) in include]
        if exclude: candidate_crops = [c for c in candidate_crops if _crop_key(c) not in exclude]
        if not candidate_crops:
            raise ValueError("No crop candidates match the selected state, season, include list, and exclude list.")
        yields = self._predict_crop_yields(candidate_crops, features)
        yield_baselines = self.meta.get("crop_median_yield_t_ha", {})
        yield_selection = self.meta.get("yield_model_selection_by_crop", {})
        yield_metrics_by_crop = self.meta.get("yield_model_metrics", {}).get("per_crop", {})
        for index, crop_name in enumerate(candidate_crops):
            if yield_selection.get(crop_name, "crop_median") == "crop_median":
                yields[index] = float(yield_baselines.get(crop_name, yields[index]))
        district = location.get("district")
        district_profile = {}
        state_profile = {}
        district_profiles = self.meta.get("district_yield_profiles", {})
        state_profiles = self.meta.get("state_yield_profiles", {})
        if district and state:
            matched_state = next((v for k, v in district_profiles.items() if _location_key(k) == _location_key(state)), {})
            district_profile = next((v for k, v in matched_state.items() if _location_key(k) == _location_key(district)), {})
            state_profile = next((v for k, v in state_profiles.items() if _location_key(k) == _location_key(state)), {})
        if district_profile:
            for index, crop_name in enumerate(candidate_crops):
                district_yield = district_profile.get(crop_name)
                state_yield = state_profile.get(crop_name)
                if district_yield is not None and state_yield is not None and state_yield > 0:
                    yields[index] *= float(district_yield) / float(state_yield)

        market = self.meta["market_prices"]
        selected_market = {}
        regional_market = self.meta.get("regional_market_prices", {})
        if state:
            state_prices = next((v for k, v in regional_market.items() if _location_key(k) == _location_key(state)), {})
            if district:
                selected_market = next((v for k, v in state_prices.items() if k and _location_key(k) == _location_key(district)), {})
            if not selected_market:
                selected_market = next((v for k, v in state_prices.items() if not k), {})
        override = inputs.get("price_override")
        costs = {_crop_key(k): float(v) for k, v in (inputs.get("crop_cost_per_ha") or {}).items() if v is not None}
        area_size = float(inputs.get("land_size", 1.0) or 1.0)
        area_ha = area_size * (ACRE_TO_HA if str(inputs.get("land_unit", "acres")).casefold().startswith("acre") else 1.0)
        crop_budget = inputs.get("crop_budget_per_acre")
        crop_budget_ha = float(crop_budget) / ACRE_TO_HA if crop_budget is not None else None
        rows = []
        for crop, yld in zip(candidate_crops, yields):
            key = _crop_key(crop)
            price_meta = selected_market.get(key, market.get(key, {}))
            price = price_meta.get("price_per_ton")
            source = price_meta.get("price_source", "unavailable")
            observation_date = price_meta.get("price_observation_date")
            if isinstance(override, dict):
                for name, value in override.items():
                    if _crop_key(name) == key and value is not None: price = float(value); source = "user_override"
            elif override is not None:
                price = float(override); source = "user_override"
            cost = costs.get(key)
            revenue = float(yld) * float(price) if price is not None else None
            profit_sim = None
            if revenue is not None and cost is not None:
                rng = np.random.default_rng(42)
                price_draw = float(price) * rng.lognormal(-.5*.15**2, .15, 10000)
                yield_draw = float(yld) * rng.lognormal(-.5*.12**2, .12, 10000)
                profit_draw = price_draw * yield_draw - cost
                profit_sim = {"mean_profit": float(np.mean(profit_draw)),
                              "median_profit": float(np.median(profit_draw)),
                              "p10_profit": float(np.percentile(profit_draw, 10)),
                              "p90_profit": float(np.percentile(profit_draw, 90)),
                              "probability_loss": float(np.mean(profit_draw < 0))}
            profit = profit_sim["mean_profit"] if profit_sim else None
            risk_weight = {"low": .75, "medium": .45, "high": .20}.get(str(inputs.get("risk_tolerance", "medium")).casefold(), .45)
            risk_adjusted = (profit + risk_weight * (profit_sim["p10_profit"] - profit)) if profit_sim else None
            within = None if cost is None else (True if crop_budget_ha is None else cost <= crop_budget_ha)
            crop_validation = yield_metrics_by_crop.get(crop, {})
            rows.append({"crop": crop, "suitability_probability": None,
                         "estimated_yield_t_ha": round(float(yld), 4),
                         "yield_estimator": yield_selection.get(crop, "crop_median"),
                         "yield_holdout_mae_t_ha": crop_validation.get("selected_mae_t_ha"),
                         "recommendation_basis": "ONNX crop-yield estimate; production costs are required for profit ranking",
                         "estimated_cost_per_ha": cost, "cost_source": "user_input" if cost is not None else "unavailable",
                         "price_per_ton": price, "price_source": source, "price_observation_date": observation_date,
                         "expected_gross_revenue_per_ha": revenue, "expected_profit_per_ha": profit,
                         "profit_p10": profit_sim["p10_profit"] if profit_sim else None,
                         "profit_p90": profit_sim["p90_profit"] if profit_sim else None,
                         "probability_of_loss": profit_sim["probability_loss"] if profit_sim else None,
                         "risk_adjusted_profit_per_ha": risk_adjusted, "within_budget": within,
                         "_profit_simulation": profit_sim})
        has_profit_data = any(r["risk_adjusted_profit_per_ha"] is not None for r in rows)
        ranking_basis = "risk_adjusted_profit_with_supplied_costs" if has_profit_data else "estimated_yield_proxy_cost_data_missing"
        for row in rows:
            row["ranking_basis"] = ranking_basis
            row["recommendation_basis"] = ranking_basis
        rows.sort(key=lambda r: (r["risk_adjusted_profit_per_ha"] is not None,
                                 r["risk_adjusted_profit_per_ha"] if r["risk_adjusted_profit_per_ha"] is not None else r["estimated_yield_t_ha"]), reverse=True)
        top_rows = rows[:5]
        best = top_rows[0]
        best_simulation = best.pop("_profit_simulation", None)
        for item in top_rows[1:]:
            item.pop("_profit_simulation", None)
        farm_rev = best["expected_gross_revenue_per_ha"] * area_ha if best["expected_gross_revenue_per_ha"] is not None else None
        farm_profit = best["expected_profit_per_ha"] * area_ha if best["expected_profit_per_ha"] is not None else None

        # Forecast uncertainty is represented by confidence bands around annual
        # climate-model means; no artificial year-to-year noise is added.
        conf = float(inputs.get("confidence_level", .90))
        z = NormalDist().inv_cdf(.5 + min(.999, max(.5, conf)) / 2)
        residual_map = self.meta.get("climate_residual_sd_by_state", {})
        residual_state = next((name for name in residual_map if state and _location_key(name) == _location_key(state)), None)
        residuals = residual_map.get(residual_state, self.meta["climate_residual_sd"])
        forecast = []
        risk_periods = []
        history = state_meta or {"means": {"temperature": 24, "rainfall": 1200, "humidity": 60},
                                 "stds": {"temperature": 1, "rainfall": 300, "humidity": 10}}
        tmean, rmean = history["means"]["temperature"], history["means"]["rainfall"]
        tscale = max(history["stds"]["temperature"], 1.0)
        rscale = max(history["stds"]["rainfall"], history["means"]["rainfall"] * .1, 1.0)
        for year in years:
            climate = climate_by_year[year]
            lead_years = max(1, year - int(self.meta["last_training_weather_year"]))
            lead_scale = math.sqrt(lead_years)
            vals = {}
            for variable in ["temperature", "rainfall", "humidity"]:
                value = climate[variable]
                sd = float(residuals[variable]) * lead_scale
                lo, hi = value - z * sd, value + z * sd
                if variable == "humidity": lo, hi = max(0, lo), min(100, hi)
                if variable == "rainfall": lo = max(0, lo)
                vals[variable] = value
                forecast.append({"date": f"{year}-01-01T00:00:00", "variable": variable, "forecast": value,
                                 "lower_95": lo, "upper_95": hi, "confidence_level": conf, "scenario": scenario})
            th = climate["temperature"] + z * float(residuals["temperature"])
            rl = climate["rainfall"] - z * float(residuals["rainfall"])
            rh = climate["rainfall"] + z * float(residuals["rainfall"])
            heat = np.clip((th - 30) / 12, 0, 1)
            drought = np.clip((rmean - rl) / (2 * rscale), 0, 1)
            flood = np.clip((rh - rmean) / (2 * rscale), 0, 1)
            anomaly = np.clip((abs((climate["temperature"]-tmean)/tscale) + abs((climate["rainfall"]-rmean)/rscale))/5, 0, 1)
            h_extreme = np.clip(max(vals["humidity"]-80, 20-vals["humidity"])/20, 0, 1)
            score = float(100 * (.25*heat + .20*drought + .20*flood + .25*anomaly + .10*h_extreme))
            risk_periods.append({"date": f"{year}-01-01T00:00:00", **vals, "risk_score": score,
                                 "risk_class": "High" if score >= 60 else "Moderate" if score >= 30 else "Low"})
        average_risk = float(np.mean([x["risk_score"] for x in risk_periods]))
        risk_level = "High" if average_risk >= 60 else "Moderate" if average_risk >= 30 else "Low"
        current_manual_risk = None
        if manual.get("temperature") is not None and manual.get("rainfall") is not None:
            ct = float(manual["temperature"])
            cr = float(manual["rainfall"])
            ch = float(manual.get("humidity", chosen_climate["humidity"]))
            heat = np.clip((ct - 30) / 12, 0, 1)
            drought = np.clip((history["means"]["rainfall"] - cr) / (2 * rscale), 0, 1)
            flood = np.clip((cr - history["means"]["rainfall"]) / (2 * rscale), 0, 1)
            anomaly = np.clip((abs((ct - tmean) / tscale) + abs((cr-rmean)/rscale))/5, 0, 1)
            humidity_extreme = np.clip(max(ch-80, 20-ch)/20, 0, 1)
            current_manual_risk = float(100*(.25*heat+.20*drought+.20*flood+.25*anomaly+.10*humidity_extreme))
        risk_advice = ["High climate risk: consider staged sowing, contingency seed, and a locally approved crop insurance review." if risk_level == "High" else
                       "Moderate climate risk: monitor forecast updates and keep irrigation/drainage options ready." if risk_level == "Moderate" else
                       "Low modeled climate risk; continue routine weather monitoring."]

        plan_crops = [r["crop"] for r in top_rows[:3]]
        fertilizer_climate = {"temperature": features["temperature"], "rainfall": features["rainfall"]}
        plans = [self._fertilizer_plan(crop, features, fertilizer_climate, area_ha, inputs,
                    next((r["price_per_ton"] for r in top_rows if r["crop"].casefold() == crop.casefold()), None))
                 for crop in plan_crops[:3]]
        inputs_out = {
            "location": {"state": state, "district": location.get("district")},
            "season": inputs.get("season"), "land_size": area_size,
            "land_unit": inputs.get("land_unit", "acres"), "crop_budget_per_acre": crop_budget,
            "crop_cost_per_ha": inputs.get("crop_cost_per_ha", {}), "risk_tolerance": inputs.get("risk_tolerance", "medium"),
            "include_crops": inputs.get("include_crops", []), "exclude_crops": inputs.get("exclude_crops", []),
            "price_override": override, "target_year": target, "horizon_months": inputs.get("horizon_months", 12),
            "horizon_years": horizon, "climate_scenario": inputs.get("climate_scenario", "normal"),
            "confidence_level": conf, "manual_weather": {k: manual.get(k) for k in ("temperature", "rainfall", "humidity")},
            "fertilizer_crop": plans[0]["crop"] if plans else None,
            "soil_test": {k: (inputs.get("soil_test") or {}).get(k) for k in ("N", "P", "K", "ph")},
            "fertilizer_budget_per_acre": inputs.get("fertilizer_budget_per_acre"),
            "optimization_goal": inputs.get("optimization_goal", "balance_roi"),
            "preferred_fertilizers": inputs.get("preferred_fertilizers", []),
            "excluded_fertilizers": inputs.get("excluded_fertilizers", []),
            "fertilizer_prices_per_kg": inputs.get("fertilizer_prices_per_kg", {"Urea": 6.2, "DAP": 27.0, "MOP": 17.0, "NPK 15-15-15": 24.0}),
        }
        report = {
            "schema_version": self.meta.get("json_schema_version", "agrivision_outputs_v1"),
            "model_version": self.meta.get("model_version"),
            "crop_profit_ranking": {
                "recommendations": top_rows,
                "profit_simulation": best_simulation,
                "farm_profit_expected_inr": farm_profit,
                "farm_profit_p10_inr": (best["profit_p10"] * area_ha if best["profit_p10"] is not None else None),
                "farm_profit_p90_inr": (best["profit_p90"] * area_ha if best["profit_p90"] is not None else None),
                "farm_revenue_expected_inr": farm_rev, "land_area_ha": area_ha,
                "training_rows": self.meta["training_rows"],
                "classifier_training_rows": 0,
                "classifier_training_source": "not_used_crop_identity_is_not_a_suitability_label",
                "crop_suitability_model_status": "not_estimated_no_suitability_labels_in_source_data",
                "yield_rows_excluded_as_invalid_units": None,
                "yield_model": "Crop-conditioned ONNX regressor with per-crop median fallback",
                "yield_model_selection": "Use the ONNX estimate only when chronological per-crop holdout MAE beats that crop's median baseline by at least 5%; otherwise use and label the crop median.",
                "model_metrics": self.meta["yield_model_metrics"],
                "per_crop_validation_metrics": self.meta["yield_model_metrics"].get("per_crop", {}),
                "features_used": SOIL_FEATURES,
                "ranking_basis": ranking_basis,
                "location_data_note": ("District/year crop-yield medians from ICRISAT adjusted the state-trained crop estimates."
                                       if district_profile else
                                       "District crop-yield history is unavailable for this selection; state crop history is used where present."),
                "financial_data_note": "Production costs are not in the source dataset. Provide crop_cost_per_ha to calculate profit.",
                "market_data_note": "Prices prefer district and state mandi records before national records. Quotes outside 0.2x-5x of each commodity's historical median are excluded; refresh metadata before production use.",
            },
            "weather_climate_risk": {
                "forecast": forecast, "risk_by_period": risk_periods,
                "risk_score": average_risk, "risk_level": risk_level, "risk_advice": risk_advice,
                "current_manual_weather_risk_score": current_manual_risk, "source_resolution": "annual",
                "geographic_resolution": "state" if state else "multi_state_average",
                "district_level_weather_available": False,
                "forecast_method": "State trend plus deterministic bootstrap of historical annual residuals; a scenario projection, not a year-specific weather forecast.",
                "resolution_note": "Annual state-level weather is the finest available climate source. District and season do not change the climate projection. Year-to-year variation reuses historical state anomalies with a stable hash so repeated runs are reproducible.",
                "risk_score_note": "The 0-100 value is a modeled climate-exposure index, not a probability of crop loss.",
            },
            "fertilizer_nutrient_plan": {
                "crop": plans[0]["crop"] if plans else None,
                "plan": plans[0] if plans else None,
                "top_three_plans": plans,
                "whole_farm_cost_inr": plans[0]["whole_farm_cost"] if plans else None,
            },
            "inputs": inputs_out,
            "location_match": {"state": state is None or _location_key(state) in self.state_index,
                               "district": bool(district_profile), "district_yield_history": bool(district_profile),
                               "weather_state": state is None or _location_key(state) in self.state_index,
                               "weather_district": False,
                               "season": season is None or any(k.casefold() == str(season).casefold() for k in self.meta["crops_by_season"])},
            "input_diagnostics": self._input_support_diagnostics(inputs),
        }
        report = _safe(report)
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        return report

    def _input_support_diagnostics(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Flag farmer-entered values outside the model's central training range."""
        support = self.meta.get("training_feature_support", {})
        supplied = {}
        for key, value in (inputs.get("soil_test") or {}).items():
            if value is not None and key in {"N", "P", "K", "ph"}:
                supplied[key] = float(value)
        for key, value in (inputs.get("manual_weather") or {}).items():
            if value is not None and key in {"temperature", "rainfall", "humidity"}:
                supplied[key] = float(value)
        warnings = []
        for key, value in supplied.items():
            bounds = support.get(key)
            if not bounds:
                continue
            lower, upper = float(bounds["q01"]), float(bounds["q99"])
            if value < lower or value > upper:
                warnings.append({"field": key, "value": value,
                                 "training_q01": lower, "training_q99": upper,
                                 "message": "Input is outside the central 98% training range; the model is extrapolating."})
        return {"within_training_support": not warnings,
                "warnings": warnings,
                "note": "Training-range checks are diagnostic only; they do not replace a lab soil test or local agronomic review."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="JSON file containing farmer inputs; defaults to an empty input object")
    parser.add_argument("--output", help="JSON output file (default /kaggle/working/agrivision_outputs.json)")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--metadata", default=str(METADATA_PATH))
    args = parser.parse_args()
    user_inputs = json.loads(Path(args.input).read_text(encoding="utf-8")) if args.input else {}
    destination = Path(args.output) if args.output else OUTPUT_PATH
    report = AgriVisionONNX(Path(args.model), Path(args.metadata)).predict(user_inputs, destination)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
