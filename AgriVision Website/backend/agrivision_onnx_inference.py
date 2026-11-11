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


def _seasonalize_climate(annual: dict[str, float], state: Optional[str], season: Optional[str]) -> dict[str, float]:
    """Turn annual state climatology into transparent seasonal estimates.

    These profiles are heuristic planning adjustments, not measured seasonal
    observations or an official weather forecast.
    """
    season_key = str(season or "").strip().casefold()
    if season_key not in {"kharif", "rabi", "summer"}:
        return dict(annual)

    state_key = str(state or "").casefold()
    coastal_wet = {"andhra pradesh", "goa", "kerala", "maharashtra", "karnataka"}
    northeast = {"arunachal pradesh", "assam", "manipur", "meghalaya", "mizoram", "nagaland", "sikkim", "tripura"}
    southeast_monsoon = {"tamil nadu", "puducherry"}
    northwest_dry = {"rajasthan", "gujarat", "haryana", "punjab", "delhi"}
    if state_key in coastal_wet:
        zone = "coastal"
    elif state_key in northeast:
        zone = "northeast"
    elif state_key in southeast_monsoon:
        zone = "southeast"
    elif state_key in northwest_dry:
        zone = "northwest"
    else:
        zone = "inland"

    # Approximate share of annual state rainfall in each broad Indian crop
    # season; profiles vary by state group, especially the northeast monsoon.
    rain_share = {
        "coastal": {"kharif": .68, "rabi": .12, "summer": .06},
        "northeast": {"kharif": .70, "rabi": .10, "summer": .06},
        "southeast": {"kharif": .48, "rabi": .29, "summer": .08},
        "northwest": {"kharif": .66, "rabi": .18, "summer": .07},
        "inland": {"kharif": .62, "rabi": .17, "summer": .08},
    }[zone][season_key]

    annual_temp = float(annual["temperature"])
    temp_offsets = {
        "coastal": {"kharif": -.3, "rabi": -2.0, "summer": 2.0},
        "northeast": {"kharif": -.7, "rabi": -2.3, "summer": 2.3},
        "southeast": {"kharif": .2, "rabi": -1.5, "summer": 2.3},
        "northwest": {"kharif": -1.0, "rabi": -4.0, "summer": 4.5},
        "inland": {"kharif": -.4, "rabi": -3.0, "summer": 3.4},
    }[zone][season_key]
    humidity_offsets = {
        "coastal": {"kharif": 8, "rabi": -7, "summer": -13},
        "northeast": {"kharif": 10, "rabi": -9, "summer": -15},
        "southeast": {"kharif": 5, "rabi": 2, "summer": -12},
        "northwest": {"kharif": 8, "rabi": -12, "summer": -20},
        "inland": {"kharif": 7, "rabi": -9, "summer": -16},
    }[zone][season_key]
    # Rainfall dispersion is scaled with the seasonal share while retaining a
    # floor for local variability in the risk calculation.
    return {
        "temperature": max(0.0, annual_temp + temp_offsets),
        "rainfall": max(0.0, float(annual["rainfall"]) * rain_share),
        "humidity": float(np.clip(float(annual["humidity"]) + humidity_offsets, 0, 100)),
    }



def _simulate_annual_climate(state: Optional[str], years: list[int],
                             history: dict[str, Any],
                             fallback_stds: dict[str, float]) -> dict[int, dict[str, float]]:
    """Create a reproducible illustrative series around state historical normals.

    The bundled metadata contains state means and standard deviations, not raw
    annual observations. The series therefore represents simulated variability,
    not measured values or a calibrated weather forecast. A stable seed makes
    the same state and year return the same values on every request.
    """
    if not years:
        return {}

    variables = ("temperature", "rainfall", "humidity")
    means = {name: float(history.get("means", {}).get(name, 0.0)) for name in variables}
    standard_deviations = {
        name: max(0.0, float(history.get("stds", {}).get(name, fallback_stds.get(name, 0.0))))
        for name in variables
    }
    identity = str(state or "india").strip().casefold()
    digest = hashlib.sha256(("agrivision-climate-v1:" + identity).encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    correlation = np.array([
        [1.0, -0.20, -0.15],
        [-0.20, 1.0, 0.35],
        [-0.15, 0.35, 1.0],
    ], dtype=np.float64)
    historical_last_year = int(history.get("last_year", min(years) - 1))
    requested_years = set(years)
    previous_shock = np.zeros(3, dtype=np.float64)
    annual: dict[int, dict[str, float]] = {}
    persistence = 0.25
    innovation_scale = math.sqrt(1.0 - persistence ** 2)

    # Always generate from the first year after the historical record so a
    # shorter horizon remains an exact prefix of a longer horizon.
    for year in range(historical_last_year + 1, max(years) + 1):
        innovation = rng.multivariate_normal(np.zeros(3), correlation)
        previous_shock = persistence * previous_shock + innovation_scale * innovation
        temperature = float(np.clip(
            means["temperature"] + standard_deviations["temperature"] * previous_shock[0],
            0.0, 50.0,
        ))
        rainfall = float(max(0.0, means["rainfall"] + standard_deviations["rainfall"] * previous_shock[1]))
        humidity = float(np.clip(
            means["humidity"] + standard_deviations["humidity"] * previous_shock[2],
            0.0, 100.0,
        ))
        if year in requested_years:
            annual[year] = {
                "temperature": temperature,
                "rainfall": rainfall,
                "humidity": humidity,
            }
    return annual


def _seasonal_exposure_points(climate: dict[str, float], season: Optional[str]) -> float:
    """Small season-aware planning-index offset from broad crop-season stresses."""
    season_key = str(season or "").casefold()
    temp, rain, humidity = climate["temperature"], climate["rainfall"], climate["humidity"]
    if season_key == "kharif":
        points = 2 + 7 * np.clip((humidity - 55) / 30, 0, 1) + 3 * np.clip(rain / 1200, 0, 1)
    elif season_key == "rabi":
        points = 2 + 4 * np.clip((45 - humidity) / 30, 0, 1) + 3 * np.clip((350 - rain) / 350, 0, 1)
    elif season_key == "summer":
        points = 3 + 8 * np.clip((temp - 27) / 8, 0, 1) + 5 * np.clip((60 - humidity) / 35, 0, 1) + 3 * np.clip((400 - rain) / 400, 0, 1)
    else:
        points = 0
    return float(np.clip(points, 0, 15))


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
        self.state_index = {s.casefold(): i for i, s in enumerate(self.states)}
        self.crop_width = len(self.crops) + len(SOIL_FEATURES)
        self.climate_width = len(self.states) + 1

    def _dummy_inputs(self) -> dict[str, np.ndarray]:
        return {
            next(n for n in self.inputs if "crop_features" in n): np.zeros((1, self.crop_width), dtype=np.float32),
            next(n for n in self.inputs if "fertilizer_features" in n): np.zeros((1, len(FERT_FEATURES)), dtype=np.float32),
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

    def _predict_fertilizer_yield(self, rows: np.ndarray) -> np.ndarray:
        feed = self._dummy_inputs()
        feed[next(n for n in feed if "fertilizer_features" in n)] = np.asarray(rows, dtype=np.float32)
        return np.maximum(0, self.session.run([self.fert_output], feed)[0].reshape(-1))

    def _predict_climate(self, state: Optional[str], years: list[int]) -> np.ndarray:
        x = np.zeros((len(years), self.climate_width), dtype=np.float32)
        if state is None:
            x[:, :len(self.states)] = 1.0 / max(1, len(self.states))
        else:
            index = self.state_index.get(state.casefold())
            if index is None:
                raise ValueError(f"Unknown state '{state}'. Choose a state in metadata states.")
            x[:, index] = 1.0
        x[:, -1] = np.asarray(years, dtype=np.float32) - 2000.0
        feed = self._dummy_inputs()
        feed[next(n for n in feed if "climate_features" in n)] = x
        return self.session.run([self.climate_output], feed)[0]

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
        matrix = np.array([[p["n"], p["p"], p["k"]] for p in products], dtype=np.float32)
        costs = np.array([p["cost"] for p in products], dtype=np.float32)
        base = np.array([soil[x] for x in ["N", "P", "K"]], dtype=np.float32)
        safe = np.array([self.meta["fertilizer_safe_targets_npk"][x] for x in ["N", "P", "K"]], dtype=np.float32)
        base_row = np.array([soil["N"], soil["P"], soil["K"], climate["temperature"], climate["rainfall"], soil["ph"]], dtype=np.float32)
        baseline = float(self._predict_fertilizer_yield(base_row[None, :])[0])
        budget_value = inputs.get("fertilizer_budget_per_acre")
        budget_ha = float(budget_value) / ACRE_TO_HA if budget_value is not None else None
        goal = str(inputs.get("optimization_goal", "balance_roi"))
        active_price = price or 0.0

        def evaluate_many(candidate: np.ndarray) -> np.ndarray:
            nutrients = base[None, :] + (candidate @ matrix) * 100.0
            fert_features = np.column_stack([nutrients, np.full(len(candidate), climate["temperature"]),
                                              np.full(len(candidate), climate["rainfall"]),
                                              np.full(len(candidate), soil["ph"])]).astype(np.float32)
            yields = self._predict_fertilizer_yield(fert_features)
            costs_total = candidate @ costs
            excess = np.maximum(nutrients - safe[None, :], 0)
            shortfall = np.maximum(safe[None, :] - nutrients, 0)
            # The fitted yield response model is not a reliable source of crop-
            # specific nutrient response labels. Include a transparent nutrient
            # target deficit term so the optimizer does not recommend an empty
            # product list just because the yield regressor learned crop averages.
            penalties = (2500.0 * np.sum(shortfall, axis=1)
                         + 1000.0 * np.sum(excess, axis=1)
                         + np.maximum(0, candidate.sum(axis=1) - 420) * 100)
            if goal in ("minimize_cost", "minimize cost", "cost"):
                scores = -costs_total - np.maximum(0, baseline - yields) * 150 - penalties
            elif goal in ("maximize_yield", "maximize yield", "yield"):
                scores = yields - .0002 * costs_total - penalties
            else:
                scores = yields * active_price - costs_total - penalties
            for i, p in enumerate(products):
                if p["name"].casefold() in preferred:
                    scores += .02 * candidate[:, i]
            if budget_ha is not None:
                scores -= np.maximum(0, costs_total - budget_ha) * 100
            return scores

        genes = np.zeros(len(products), dtype=np.float32)
        levels = [np.linspace(0, cap, 12, dtype=np.float32) for cap in caps]
        for _ in range(3):
            for i in range(len(products)):
                candidates = np.tile(genes, (len(levels[i]), 1))
                candidates[:, i] = levels[i]
                scores = evaluate_many(candidates)
                genes[i] = candidates[int(np.argmax(scores)), i]
        final_nutrients = base + (genes @ matrix) * 100.0
        final_features = np.array([[final_nutrients[0], final_nutrients[1], final_nutrients[2],
                                    climate["temperature"], climate["rainfall"], soil["ph"]]], dtype=np.float32)
        pred_yield = float(self._predict_fertilizer_yield(final_features)[0])
        plan = {p["name"]: round(float(q), 2) for p, q in zip(products, genes) if q > .1}
        cost = float(genes @ costs)
        return {
            "crop": crop,
            "fertilizer_kg_ha": plan,
            "total_product_kg_ha": round(float(genes.sum()), 2),
            "estimated_cost_per_ha": round(cost, 2),
            "predicted_yield_t_ha": round(pred_yield, 3),
            "baseline_predicted_yield_t_ha": round(baseline, 3),
            "estimated_yield_gain_t_ha": round(pred_yield - baseline, 3),
            "final_soil_NPK": {k: round(float(v), 2) for k, v in zip(["N", "P", "K"], final_nutrients)},
            "starting_soil_NPK": {k: round(float(v), 2) for k, v in zip(["N", "P", "K"], base)},
            "added_soil_NPK": {k: round(float(v), 2) for k, v in zip(["N", "P", "K"], (genes @ matrix) * 100)},
            "guardrail_targets_NPK": {k: float(v) for k, v in zip(["N", "P", "K"], safe)},
            "optimization_goal": goal,
            "requested_optimization_goal": goal,
            "budget_per_ha": budget_ha,
            "within_budget": budget_ha is None or cost <= budget_ha,
            "whole_farm_cost": cost * area_ha,
            "whole_farm_fertilizer_kg": {k: float(v) * area_ha for k, v in plan.items()},
            "warning": "ONNX model estimate; confirm soil test, local nutrient guidance, and product labels before application.",
        }

    def predict(self, user_inputs: Optional[dict[str, Any]] = None,
                output_path: Optional[Path] = None) -> dict[str, Any]:
        inputs = dict(user_inputs or {})
        for key in ("include_crops", "exclude_crops", "preferred_fertilizers", "excluded_fertilizers"):
            value = inputs.get(key, [])
            inputs[key] = [x.strip() for x in value.split(",") if x.strip()] if isinstance(value, str) else list(value or [])
        location = inputs.get("location") or {}
        state = location.get("state")
        state_meta = next((v for k, v in self.meta["state_climate_summary"].items()
                           if state and k.casefold() == str(state).casefold()), None)
        state_defaults = next((v for k, v in self.meta["state_feature_defaults"].items()
                               if state and k.casefold() == str(state).casefold()), self.meta["default_features"])
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
        if state and not any(name.casefold() == str(state).casefold() for name in self.states):
            raise ValueError(f"Unknown state '{state}'. Choose a state in metadata states.")
        if state_meta is None:
            summaries = list(self.meta["state_climate_summary"].values())
            history = {
                "last_year": last_year,
                "means": {
                    name: float(np.mean([item["means"][name] for item in summaries]))
                    for name in ("temperature", "rainfall", "humidity")
                },
                "stds": {
                    name: float(np.sqrt(np.mean([item["stds"][name] ** 2 for item in summaries])))
                    for name in ("temperature", "rainfall", "humidity")
                },
            }
        else:
            history = state_meta

        annual_climate_by_year = _simulate_annual_climate(
            state, years, history, self.meta["climate_residual_sd"]
        )
        climate_by_year = {}
        scenario = str(inputs.get("climate_scenario", "normal")).casefold()
        season = inputs.get("season")
        for year in years:
            temp = annual_climate_by_year[year]["temperature"]
            rain = annual_climate_by_year[year]["rainfall"]
            humidity = annual_climate_by_year[year]["humidity"]
            if scenario == "drought-prone":
                temp += 1.0
                rain *= .70
            elif scenario == "flood-prone":
                rain *= 1.30
            annual_climate = {
                "temperature": temp,
                "rainfall": max(0.0, rain),
                "humidity": float(np.clip(humidity, 0, 100)),
            }
            annual_climate_by_year[year] = annual_climate
            climate_by_year[year] = _seasonalize_climate(annual_climate, state, season)
        chosen_climate = climate_by_year[years[-1]]
        features.update(chosen_climate)
        for k in ("temperature", "rainfall", "humidity"):
            if manual.get(k) is not None:
                features[k] = float(manual[k])

        candidate_crops = self.crops
        if state:
            match = next((v for k, v in self.meta["crops_by_state"].items() if k.casefold() == state.casefold()), None)
            if match: candidate_crops = [c for c in candidate_crops if c in match]
        if season:
            match = next((v for k, v in self.meta["crops_by_season"].items() if k.casefold() == str(season).casefold()), None)
            if match: candidate_crops = [c for c in candidate_crops if c in match]
        include = {_crop_key(x) for x in inputs.get("include_crops", [])}
        exclude = {_crop_key(x) for x in inputs.get("exclude_crops", [])}
        if include: candidate_crops = [c for c in candidate_crops if _crop_key(c) in include]
        if exclude: candidate_crops = [c for c in candidate_crops if _crop_key(c) not in exclude]
        if not candidate_crops:
            candidate_crops = self.crops
        yields = self._predict_crop_yields(candidate_crops, features)

        market = self.meta["market_prices"]
        override = inputs.get("price_override")
        costs = {_crop_key(k): float(v) for k, v in (inputs.get("crop_cost_per_ha") or {}).items() if v is not None}
        area_size = float(inputs.get("land_size", 1.0) or 1.0)
        area_ha = area_size * (ACRE_TO_HA if str(inputs.get("land_unit", "acres")).casefold().startswith("acre") else 1.0)
        crop_budget = inputs.get("crop_budget_per_acre")
        crop_budget_ha = float(crop_budget) / ACRE_TO_HA if crop_budget is not None else None
        rows = []
        for crop, yld in zip(candidate_crops, yields):
            key = _crop_key(crop)
            price_meta = market.get(key, {})
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
            rows.append({"crop": crop, "suitability_probability": None,
                         "estimated_yield_t_ha": round(float(yld), 4),
                         "recommendation_basis": "ONNX crop-yield estimate ranked by crop-matched latest price/revenue; suitability classifier omitted",
                         "estimated_cost_per_ha": cost, "cost_source": "user_input" if cost is not None else "unavailable",
                         "price_per_ton": price, "price_source": source, "price_observation_date": observation_date,
                         "expected_gross_revenue_per_ha": revenue, "expected_profit_per_ha": profit,
                         "profit_p10": profit_sim["p10_profit"] if profit_sim else None,
                         "profit_p90": profit_sim["p90_profit"] if profit_sim else None,
                         "probability_of_loss": profit_sim["probability_loss"] if profit_sim else None,
                         "risk_adjusted_profit_per_ha": risk_adjusted, "within_budget": within,
                         "_profit_simulation": profit_sim})
        rows.sort(key=lambda r: (r["risk_adjusted_profit_per_ha"] is not None,
                                r["expected_gross_revenue_per_ha"] is not None,
                                r["risk_adjusted_profit_per_ha"] or r["expected_gross_revenue_per_ha"] or r["estimated_yield_t_ha"]), reverse=True)
        top_rows = rows[:5]
        best = top_rows[0]
        best_simulation = best.pop("_profit_simulation", None)
        for item in top_rows[1:]:
            item.pop("_profit_simulation", None)
        farm_rev = best["expected_gross_revenue_per_ha"] * area_ha if best["expected_gross_revenue_per_ha"] is not None else None
        farm_profit = best["expected_profit_per_ha"] * area_ha if best["expected_profit_per_ha"] is not None else None

        # Bands use state-level historical variability; annual point values are
        # deterministic simulations because raw year-level records are absent.
        conf = float(inputs.get("confidence_level", .90))
        z = NormalDist().inv_cdf(.5 + min(.999, max(.5, conf)) / 2)
        climate_sd = {name: max(0.0, float(history.get("stds", {}).get(name, self.meta["climate_residual_sd"][name])))
                      for name in ("temperature", "rainfall", "humidity")}
        forecast = []
        risk_periods = []
        seasonal_outlook = []
        driver_totals = {"Heat stress": 0.0, "Dry-spell exposure": 0.0,
                         "Excess-rain exposure": 0.0, "Seasonal exposure": 0.0,
                         "Humidity extremes": 0.0}
        seasonal_history = _seasonalize_climate(history["means"], state, season)
        rain_share = seasonal_history["rainfall"] / max(float(history["means"]["rainfall"]), 1.0)
        tmean, rmean = seasonal_history["temperature"], seasonal_history["rainfall"]
        tscale = max(history["stds"]["temperature"], 1.0)
        rscale = max(history["stds"]["rainfall"] * rain_share, seasonal_history["rainfall"] * .15, 1.0)
        for season_name in ("Kharif", "Rabi", "Summer"):
            season_history = _seasonalize_climate(history["means"], state, season_name)
            season_rain_share = season_history["rainfall"] / max(float(history["means"]["rainfall"]), 1.0)
            season_rscale = max(history["stds"]["rainfall"] * season_rain_share,
                                season_history["rainfall"] * .15, 1.0)
            for outlook_year in years:
                season_climate = _seasonalize_climate(annual_climate_by_year[outlook_year], state, season_name)
                season_temp_high = season_climate["temperature"] + z * float(climate_sd["temperature"])
                season_rain_low = season_climate["rainfall"] - z * float(climate_sd["rainfall"]) * season_rain_share
                season_rain_high = season_climate["rainfall"] + z * float(climate_sd["rainfall"]) * season_rain_share
                season_heat = np.clip((season_temp_high - 30) / 12, 0, 1)
                season_drought = np.clip((season_history["rainfall"] - season_rain_low) / (2 * season_rscale), 0, 1)
                season_flood = np.clip((season_rain_high - season_history["rainfall"]) / (2 * season_rscale), 0, 1)
                season_anomaly = np.clip((abs((season_temp_high-season_history["temperature"])/tscale)
                                          + abs((season_rain_high-season_history["rainfall"])/season_rscale))/5, 0, 1)
                season_humidity_extreme = np.clip(max(season_climate["humidity"]-80, 20-season_climate["humidity"])/20, 0, 1)
                season_exposure = min(25.0, 25*float(season_anomaly) + _seasonal_exposure_points(season_climate, season_name))
                season_score = (25*float(season_heat) + 20*float(season_drought) + 20*float(season_flood)
                                + season_exposure + 10*float(season_humidity_extreme))
                seasonal_outlook.append({"date": f"{outlook_year}-01-01T00:00:00", "season": season_name,
                                         **season_climate, "risk_score": season_score,
                                         "risk_class": "High" if season_score >= 60 else "Moderate" if season_score >= 30 else "Low"})
        for year in years:
            climate = climate_by_year[year]
            vals = {}
            for variable in ["temperature", "rainfall", "humidity"]:
                value = climate[variable]
                sd = float(climate_sd[variable]) * (rain_share if variable == "rainfall" else 1.0)
                lo, hi = value - z * sd, value + z * sd
                if variable == "temperature": lo, hi = max(0, lo), max(0, hi)
                elif variable == "rainfall": lo = max(0, lo)
                elif variable == "humidity": lo, hi = max(0, lo), min(100, hi)
                vals[variable] = value
                forecast.append({"date": f"{year}-01-01T00:00:00", "variable": variable, "forecast": value,
                                 "lower_95": lo, "upper_95": hi, "confidence_level": conf, "scenario": scenario})
            th = climate["temperature"] + z * float(climate_sd["temperature"])
            seasonal_rain_sd = float(climate_sd["rainfall"]) * rain_share
            rl = climate["rainfall"] - z * seasonal_rain_sd
            rh = climate["rainfall"] + z * seasonal_rain_sd
            heat = np.clip((th - 30) / 12, 0, 1)
            drought = np.clip((rmean - rl) / (2 * rscale), 0, 1)
            flood = np.clip((rh - rmean) / (2 * rscale), 0, 1)
            anomaly = np.clip((abs((th-tmean)/tscale) + abs((rh-rmean)/rscale))/5, 0, 1)
            h_extreme = np.clip(max(vals["humidity"]-80, 20-vals["humidity"])/20, 0, 1)
            seasonal_exposure = min(25.0, 25 * float(anomaly) + _seasonal_exposure_points(climate, season))
            components = {"Heat stress": 25 * float(heat),
                          "Dry-spell exposure": 20 * float(drought),
                          "Excess-rain exposure": 20 * float(flood),
                          "Seasonal exposure": seasonal_exposure,
                          "Humidity extremes": 10 * float(h_extreme)}
            score = float(sum(components.values()))
            for driver, contribution in components.items():
                driver_totals[driver] += contribution
            risk_periods.append({"date": f"{year}-01-01T00:00:00", **vals, "season": season,
                                 "risk_score": score,
                                 "risk_class": "High" if score >= 60 else "Moderate" if score >= 30 else "Low"})
        average_risk = float(np.mean([x["risk_score"] for x in risk_periods]))
        display_climate_score = max(1.0, average_risk)
        risk_level = "High" if average_risk >= 60 else "Moderate" if average_risk >= 30 else "Low"
        driver_guidance = {
            "Heat stress": ("Heat stress", "Use the locally recommended sowing window and keep water available at sensitive growth stages.", "Avoid sowing outside the local calendar or applying foliar products during peak heat."),
            "Dry-spell exposure": ("Dry spells", "Check seed-zone moisture and confirm an irrigation or moisture-conservation plan before sowing.", "Avoid committing the full area to sowing when establishment moisture is uncertain."),
            "Excess-rain exposure": ("Heavy rain / waterlogging", "Check drainage and protect low-lying plots; split nutrient applications where locally advised.", "Avoid applying mobile nutrients immediately before forecast heavy rain."),
            "Seasonal exposure": ("Seasonal exposure", "Keep a backup sowing window and plan around the selected season's heat, rain and humidity pattern.", "Avoid treating this broad seasonal profile as a date-specific forecast."),
            "Humidity extremes": ("Humidity / disease pressure", "Scout crops regularly and use locally recommended resistant varieties where available.", "Avoid pesticide applications without a diagnosed problem and label guidance."),
        }
        driver_scores = {name: value / max(1, len(risk_periods)) for name, value in driver_totals.items()}
        top_drivers = sorted(driver_scores.items(), key=lambda item: item[1], reverse=True)[:3]
        risk_guidance_drivers = [
            {"driver": driver_guidance[name][0], "score": round(score, 1),
             "action": driver_guidance[name][1], "avoid": driver_guidance[name][2]}
            for name, score in top_drivers if score >= 1
        ]
        if not risk_guidance_drivers:
            risk_guidance_drivers = [{"driver": "Residual forecast uncertainty", "score": 1.0,
                "action": "Keep checking local weather and field conditions before sowing and input applications.",
                "avoid": "Treat the model outlook as a guarantee of safe conditions."}]
        # Crop score blends the state climate exposure with the crop's relative
        # uncertainty using the prototype holdout MAE. If costs are supplied,
        # simulated loss probability receives additional weight. This is a
        # planning index, not a calibrated probability of loss.
        yield_mae = float(self.meta["yield_model_metrics"].get("holdout_mae_t_ha", 1.19))
        for row in top_rows:
            relative_error_ratio = yield_mae / max(float(row["estimated_yield_t_ha"]), 0.01)
            yield_uncertainty_index = 100 * relative_error_ratio / (1 + relative_error_ratio)
            crop_climate_yield_proxy = .55 * display_climate_score + .45 * yield_uncertainty_index
            if row["probability_of_loss"] is None:
                risk_score = crop_climate_yield_proxy
                risk_method = "blend of state climate score and relative yield-model MAE"
            else:
                risk_score = .70 * row["probability_of_loss"] * 100 + .30 * crop_climate_yield_proxy
                risk_method = "blend of simulated financial loss, state climate score, and relative yield-model MAE"
            row["downside_risk_score"] = round(min(100.0, max(1.0, risk_score)), 1)
            row["downside_risk_method"] = risk_method
        if best["risk_adjusted_profit_per_ha"] is not None:
            top_pick_basis = "risk_adjusted_profit"
            top_pick_reason = (
                f"{best['crop']} ranks first on risk-adjusted expected profit "
                f"(₹{best['risk_adjusted_profit_per_ha']:,.0f}/ha), using the "
                f"{inputs.get('risk_tolerance', 'medium')} risk setting and user-provided production costs."
            )
        elif best["expected_gross_revenue_per_ha"] is not None:
            top_pick_basis = "modeled_gross_revenue"
            top_pick_reason = (
                f"{best['crop']} has the highest modeled gross revenue "
                f"(₹{best['expected_gross_revenue_per_ha']:,.0f}/ha) among the top five, "
                "calculated from its modeled yield and the embedded crop price. "
                "Profit is not ranked because production costs were not supplied."
            )
        else:
            top_pick_basis = "modeled_yield"
            top_pick_reason = (
                f"{best['crop']} has the highest modeled yield among the top five; "
                "matched price and production cost data are unavailable."
            )
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
        risk_advice = ["High climate exposure: use a staged sowing plan and prepare locally approved contingencies." if risk_level == "High" else
                       "Moderate climate exposure: check forecast updates and keep irrigation and drainage options ready." if risk_level == "Moderate" else
                       "Low modeled climate exposure; keep routine weather monitoring in the plan."]

        plan_crops = [r["crop"] for r in top_rows[:3]]
        plans = [self._fertilizer_plan(crop, features, chosen_climate, area_ha, inputs,
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
            "crop_profit_ranking": {
                "recommendations": top_rows,
                "top_pick_basis": top_pick_basis,
                "top_pick_reason": top_pick_reason,
                "profit_simulation": best_simulation,
                "farm_profit_expected_inr": farm_profit,
                "farm_profit_p10_inr": (best["profit_p10"] * area_ha if best["profit_p10"] is not None else None),
                "farm_profit_p90_inr": (best["profit_p90"] * area_ha if best["profit_p90"] is not None else None),
                "farm_revenue_expected_inr": farm_rev, "land_area_ha": area_ha,
                "training_rows": self.meta["training_rows"],
                "classifier_training_rows": self.meta["training_rows"],
                "classifier_training_source": "onnx_yield_regressor",
                "crop_suitability_model_status": "yield_only_no_calibrated_suitability_classifier",
                "yield_rows_excluded_as_invalid_units": None,
                "yield_model": "RandomForestRegressor exported to ONNX",
                "yield_model_selection": "Single crop-conditioned yield model; trained once from supplied crop and soil/weather data.",
                "model_metrics": self.meta["yield_model_metrics"], "per_crop_validation_metrics": {},
                "features_used": SOIL_FEATURES,
                "financial_data_note": "Production costs are not in the source dataset. Provide crop_cost_per_ha to calculate profit.",
                "market_data_note": "Mandi prices are embedded training-time snapshots; refresh metadata before production use.",
            },
            "weather_climate_risk": {
                "forecast": forecast, "risk_by_period": risk_periods,
                "seasonal_outlook": seasonal_outlook,
                "forecast_method": "Historical state climatology with deterministic simulated interannual variability; no unsupported linear trend is extrapolated.",
                "forecast_note": "The bundle contains state summaries but no raw year-level climate series. Annual point values are reproducible illustrative scenarios, not observations or a calibrated forecast.",
                "seasonal_method": "State-level historical climatology with heuristic seasonal adjustment and deterministic simulated year-to-year variability; not observed seasonal weather or a district-level forecast.",
                "risk_score": display_climate_score, "risk_level": risk_level, "risk_advice": risk_advice,
                "risk_guidance": {"score": display_climate_score, "scale": "0-100 planning index",
                                  "drivers": risk_guidance_drivers,
                                  "disclaimer": "Climate score is a prototype planning index, not an observed or crop-specific probability of loss."},
                "current_manual_weather_risk_score": current_manual_risk, "source_resolution": "state-level climatology and season adjusted",
            },
            "fertilizer_nutrient_plan": {
                "crop": plans[0]["crop"] if plans else None,
                "plan": plans[0] if plans else None,
                "top_three_plans": plans,
                "whole_farm_cost_inr": plans[0]["whole_farm_cost"] if plans else None,
            },
            "inputs": inputs_out,
            "location_match": {"state": state is None or state.casefold() in self.state_index,
                               "district": False, "season": season is None or any(k.casefold() == str(season).casefold() for k in self.meta["crops_by_season"])},
        }
        report = _safe(report)
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        return report


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
