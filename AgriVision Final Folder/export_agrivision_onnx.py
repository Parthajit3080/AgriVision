"""Train and export the AgriVision dataset-free website inference bundle.

Run once during model preparation. Runtime inference uses only the ONNX file and
agrivision_onnx_metadata.json; it does not open the source CSV files.
"""
from __future__ import annotations

import json
import os
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEPS = ROOT / ".onnx_build_deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import onnx
from onnx import compose
# skl2onnx eagerly imports optional clustering/mixture converters. They are not
# used by these regressors, and the desktop sandbox blocks loading their native
# extensions, so provide inert symbols only for converter registration.
import sklearn
cluster_stub = types.ModuleType("sklearn.cluster")
cluster_stub.KMeans = type("KMeans", (), {})
cluster_stub.MiniBatchKMeans = type("MiniBatchKMeans", (), {})
mixture_stub = types.ModuleType("sklearn.mixture")
mixture_stub.__path__ = []
mixture_stub.GaussianMixture = type("GaussianMixture", (), {})
mixture_stub.BayesianGaussianMixture = type("BayesianGaussianMixture", (), {})
mixture_submodule = types.ModuleType("sklearn.mixture.gaussian_mixture")
mixture_submodule._compute_log_det_cholesky = lambda *args, **kwargs: None
sys.modules.setdefault("sklearn.cluster", cluster_stub)
sys.modules.setdefault("sklearn.mixture", mixture_stub)
sys.modules.setdefault("sklearn.mixture.gaussian_mixture", mixture_submodule)
sys.modules.setdefault("sklearn.mixture._gaussian_mixture", mixture_submodule)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SOURCE_FILE = ROOT / "agrivision_agriintelligence.py"
DATA_DIR = Path(os.environ.get("AGRIVISION_DATA_DIR", ROOT / "data"))
MODEL_PATH = ROOT / "agrivision_website_model.onnx"
METADATA_PATH = ROOT / "agrivision_onnx_metadata.json"
RANDOM_STATE = 42
SOIL_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
FERT_FEATURES = ["N", "P", "K", "temperature", "rainfall", "ph"]


def main() -> None:
    os.environ["AGRIVISION_DATA_DIR"] = str(DATA_DIR)
    core = runpy.run_path(str(SOURCE_FILE))
    read = core["_canonicalize_columns"]
    raw = read(pd.read_csv(core["_find_csv"]("crop_yield.csv")))
    weather_file = core["_find_csv"]("state_weather_data_1997_2020.csv")
    weather_raw = read(pd.read_csv(weather_file)) if weather_file else None
    weather = core["_prepare_weather"](weather_raw) if weather_raw is not None else None
    data = core["_prepare_training_data"](raw, weather)
    data["crop"] = data.crop.astype(str).str.strip()
    data["yield_t_ha"] = pd.to_numeric(data.yield_t_ha, errors="coerce")
    data = data[data.crop.ne("") & data.yield_t_ha.between(0, 250)].copy()
    crops = sorted(data.crop.unique().tolist(), key=str.casefold)
    crop_to_idx = {name: i for i, name in enumerate(crops)}

    feature_medians = {c: float(data[c].median()) for c in SOIL_FEATURES}
    feature_array = data[SOIL_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(feature_medians).to_numpy(dtype=np.float32)
    crop_onehot = np.zeros((len(data), len(crops)), dtype=np.float32)
    crop_ids = data.crop.map(crop_to_idx).to_numpy()
    crop_onehot[np.arange(len(data)), crop_ids] = 1.0
    X_crop = np.concatenate([crop_onehot, feature_array], axis=1)
    y = data.yield_t_ha.to_numpy(dtype=np.float32)
    if "year" in data and pd.to_numeric(data.year, errors="coerce").nunique() >= 5:
        year_values = pd.to_numeric(data.year, errors="coerce")
        years = np.sort(year_values.dropna().unique())
        cutoff = years[max(1, int(len(years) * .8) - 1)]
        train_ix = np.flatnonzero((year_values <= cutoff).to_numpy())
        test_ix = np.flatnonzero((year_values > cutoff).to_numpy())
        validation_split = "chronological_holdout_by_year"
    else:
        train_ix, test_ix = train_test_split(np.arange(len(data)), test_size=.20, random_state=RANDOM_STATE)
        validation_split = "deterministic_random_fallback"
    if len(train_ix) < 10 or len(test_ix) < 5:
        train_ix, test_ix = train_test_split(np.arange(len(data)), test_size=.20, random_state=RANDOM_STATE)
        validation_split = "deterministic_random_fallback_small_time_holdout"
    crop_model = RandomForestRegressor(n_estimators=240, max_depth=12, min_samples_leaf=6,
                                       max_features=.8, n_jobs=-1, random_state=RANDOM_STATE)
    evaluation_model = RandomForestRegressor(n_estimators=240, max_depth=12, min_samples_leaf=6,
                                              max_features=.8, n_jobs=-1, random_state=RANDOM_STATE)
    evaluation_model.fit(X_crop[train_ix], y[train_ix])
    crop_pred = np.maximum(0, evaluation_model.predict(X_crop[test_ix]))
    train_crop_medians = pd.DataFrame({"crop_id": crop_ids[train_ix], "yield": y[train_ix]}).groupby("crop_id")["yield"].median()
    baseline_pred = np.array([float(train_crop_medians.get(crop_ids[i], np.median(y[train_ix]))) for i in test_ix])
    model_mae = float(mean_absolute_error(y[test_ix], crop_pred))
    baseline_mae = float(mean_absolute_error(y[test_ix], baseline_pred))
    holdout_frame = pd.DataFrame({
        "crop": data.iloc[test_ix].crop.astype(str).to_numpy(),
        "actual": y[test_ix], "model": crop_pred, "crop_median_baseline": baseline_pred,
    })
    per_crop_metrics = {
        str(crop): {
            "holdout_rows": int(len(group)),
            "mae_t_ha": float(mean_absolute_error(group.actual, group.model)),
            "crop_median_baseline_mae_t_ha": float(mean_absolute_error(group.actual, group.crop_median_baseline)),
        }
        for crop, group in holdout_frame.groupby("crop")
    }
    yield_model_selection = {
        crop: ("onnx_model" if crop in per_crop_metrics and
               per_crop_metrics[crop]["mae_t_ha"] < .95 * per_crop_metrics[crop]["crop_median_baseline_mae_t_ha"]
               else "crop_median")
        for crop in crops
    }
    selected_holdout = holdout_frame.copy()
    selected_holdout["selected"] = [
        row.model if yield_model_selection[row.crop] == "onnx_model" else row.crop_median_baseline
        for row in selected_holdout.itertuples()
    ]
    selected_estimator_mae = float(mean_absolute_error(selected_holdout.actual, selected_holdout.selected))
    for crop, group in selected_holdout.groupby("crop"):
        per_crop_metrics[crop]["selected_mae_t_ha"] = float(mean_absolute_error(group.actual, group.selected))
        per_crop_metrics[crop]["selected_estimator"] = yield_model_selection[crop]
    yield_metrics = {"holdout_mae_t_ha": model_mae,
                     "crop_median_baseline_mae_t_ha": baseline_mae,
                     "selected_estimator_holdout_mae_t_ha": selected_estimator_mae,
                     "model_beats_baseline": model_mae < baseline_mae,
                     "holdout_r2": float(r2_score(y[test_ix], crop_pred)),
                     "holdout_rows": int(len(test_ix)), "training_rows": int(len(train_ix)),
                     "validation_split": validation_split,
                     "holdout_years": sorted(int(v) for v in pd.to_numeric(data.iloc[test_ix].year, errors="coerce").dropna().unique()) if "year" in data else [],
                     "per_crop": per_crop_metrics}
    # Export a production model fitted on all observed years only after reporting
    # the untouched chronological holdout metrics above.
    crop_model.fit(X_crop, y)
    crop_onnx = convert_sklearn(crop_model, initial_types=[("crop_features", FloatTensorType([None, X_crop.shape[1]]))],
                                target_opset=17, options={id(crop_model): {"decision_leaf": False}})

    # This head estimates observed yield conditioned on crop and soil/weather
    # inputs. It is not a fertilizer-treatment response model; the source has
    # no randomized treatment or application-rate labels.
    fert_frame = data[FERT_FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    X_fert = np.concatenate([crop_onehot, fert_frame], axis=1)
    fert_model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(48, 24), activation="relu", alpha=.02,
                     max_iter=1200, early_stopping=True, random_state=RANDOM_STATE))
    fert_model.fit(X_fert, y)
    fert_onnx = convert_sklearn(fert_model, initial_types=[("fertilizer_features", FloatTensorType([None, len(crops) + len(FERT_FEATURES)]))],
                                target_opset=17)

    if weather_raw is None:
        raise FileNotFoundError("state_weather_data_1997_2020.csv is required to export the climate head.")
    climate = read(pd.read_csv(weather_file))
    if "year" not in climate or "state" not in climate:
        raise ValueError("Climate source must contain state and year columns.")
    climate = climate.dropna(subset=["state", "year", "temperature", "rainfall"])
    climate["state"] = climate.state.astype(str).str.strip()
    states = sorted(climate.state.unique().tolist(), key=str.casefold)
    state_index = {state.casefold(): i for i, state in enumerate(states)}
    climate = climate.groupby(["state", "year"], as_index=False)[["temperature", "rainfall", "humidity"]].mean()
    state_hot = np.zeros((len(climate), len(states)), dtype=np.float32)
    ids = climate.state.map(lambda v: state_index[v.casefold()]).to_numpy()
    state_hot[np.arange(len(climate)), ids] = 1.0
    years = pd.to_numeric(climate.year, errors="coerce").fillna(2000).to_numpy(dtype=np.float32)
    climate_year_center = float(np.median(years))
    # Give each state its own long-term slope. The former shared year slope made
    # every state's forecast drift in the same way and obscured regional trends.
    X_climate = np.column_stack([state_hot, state_hot * (years[:, None] - climate_year_center)]).astype(np.float32)
    climate_targets = climate[["temperature", "rainfall", "humidity"]].to_numpy(dtype=np.float32)
    climate_model = LinearRegression().fit(X_climate, climate_targets)
    climate_pred = climate_model.predict(X_climate)
    climate_residual = np.std(climate_targets - climate_pred, axis=0, ddof=1)
    climate_residual_by_state = {}
    climate_resid_frame = climate[["state"]].copy()
    climate_resid_frame[["temperature", "rainfall", "humidity"]] = climate_targets - climate_pred
    climate_residuals_by_state = {}
    climate_projection_bounds_by_state = {}
    for state, group in climate_resid_frame.groupby("state"):
        ordered = group.reset_index(drop=True)
        climate_residuals_by_state[state] = ordered[["temperature", "rainfall", "humidity"]].astype(float).values.tolist()
        observed = climate.loc[climate.state.eq(state), ["temperature", "rainfall", "humidity"]]
        climate_projection_bounds_by_state[state] = {
            variable: {"min": float(observed[variable].min()), "max": float(observed[variable].max())}
            for variable in ["temperature", "rainfall", "humidity"]
        }
        climate_residual_by_state[state] = {
            variable: float(group[variable].std(ddof=1)) if len(group) > 1 else float(climate_residual[i])
            for i, variable in enumerate(["temperature", "rainfall", "humidity"])
        }
    pooled_climate_residuals = climate_resid_frame[["temperature", "rainfall", "humidity"]].astype(float).values.tolist()
    climate_onnx = convert_sklearn(climate_model,
        initial_types=[("climate_features", FloatTensorType([None, len(states) * 2]))], target_opset=17)

    # skl2onnx can select opset 16 for its MLP converter even when requested
    # opset 17; both graphs use operators whose semantics are identical here.
    # Align graph imports before combining their independent inference heads.
    for model in (crop_onnx, fert_onnx, climate_onnx):
        for opset in model.opset_import:
            if opset.domain == "":
                opset.version = 17
    merged = compose.merge_models(crop_onnx, fert_onnx, io_map=[], prefix1="crop_", prefix2="fert_")
    merged = compose.merge_models(merged, climate_onnx, io_map=[], prefix1="", prefix2="climate_")
    onnx.checker.check_model(merged)
    onnx.save_model(merged, str(MODEL_PATH))

    # Compact, non-training metadata required to populate the stable JSON response.
    medians_by_state = {}
    if "state" in data:
        for state, group in data.groupby(data.state.astype(str).str.strip()):
            medians_by_state[core["_location_key"](state)] = {c: float(group[c].median()) for c in SOIL_FEATURES}
    if "state" in data:
        state_defaults = {s: medians_by_state.get(core["_location_key"](s), feature_medians) for s in states}
    else:
        state_defaults = {s: feature_medians for s in states}
    crops_by_state = {}
    if "state" in data:
        for state, group in data.groupby(data.state.astype(str).str.strip()):
            crops_by_state[state] = sorted(group.crop.unique().tolist(), key=str.casefold)
    crops_by_season = {}
    if "season" in data:
        for season, group in data.groupby(data.season.astype(str).str.strip()):
            crops_by_season[season] = sorted(group.crop.unique().tolist(), key=str.casefold)

    prices = {}
    regional_market_prices = {}
    price_file = core["_find_csv"]("Price_Agriculture_commodities_Week.csv")
    if price_file:
        market = read(pd.read_csv(price_file))
        if {"crop", "price_per_ton"}.issubset(market.columns):
            market["crop_key"] = market.crop.map(core["_crop_key"])
            market["price_per_ton"] = market.price_per_ton * 10.0
            historical_price_medians = market.groupby("crop_key").price_per_ton.transform("median")
            plausible_price = (market.price_per_ton > 0) & market.price_per_ton.between(historical_price_medians / 5, historical_price_medians * 5)
            market = market.loc[plausible_price].copy()
            if "date" in market:
                market["_date"] = pd.to_datetime(market.date, dayfirst=True, errors="coerce")
            else:
                market["_date"] = pd.NaT
            for key, rows in market.dropna(subset=["price_per_ton"]).groupby("crop_key"):
                dated = rows.dropna(subset=["_date"])
                if len(dated):
                    latest = dated._date.max()
                    latest_rows = dated[dated._date.eq(latest)]
                    prices[key] = {"price_per_ton": float(latest_rows.price_per_ton.median()),
                                   "price_observation_date": latest.isoformat(), "price_source": "national_latest_market"}
            if {"state", "district"}.issubset(market.columns):
                regional_rows = market.dropna(subset=["state", "price_per_ton"])
                group_cols = ["state", "district", "crop_key"]
                for (state_name, district_name, key), rows in regional_rows.groupby(group_cols, dropna=False):
                    dated = rows.dropna(subset=["_date"])
                    selected = dated[dated._date.eq(dated._date.max())] if len(dated) else rows
                    state_key = str(state_name).strip()
                    district_key = "" if pd.isna(district_name) else str(district_name).strip()
                    regional_market_prices.setdefault(state_key, {}).setdefault(district_key, {})[str(key)] = {
                        "price_per_ton": float(selected.price_per_ton.median()),
                        "price_observation_date": (selected._date.max().isoformat() if selected._date.notna().any() else None),
                        "price_source": "selected_region_latest_market",
                    }
                for (state_name, key), rows in regional_rows.groupby(["state", "crop_key"], dropna=False):
                    dated = rows.dropna(subset=["_date"])
                    selected = dated[dated._date.eq(dated._date.max())] if len(dated) else rows
                    state_key = str(state_name).strip()
                    regional_market_prices.setdefault(state_key, {}).setdefault("", {})[str(key)] = {
                        "price_per_ton": float(selected.price_per_ton.median()),
                        "price_observation_date": (selected._date.max().isoformat() if selected._date.notna().any() else None),
                        "price_source": "selected_state_latest_market",
                    }

    state_climate = {}
    for state, group in climate.groupby("state"):
        vals = group[["temperature", "rainfall", "humidity"]]
        state_climate[state] = {
            "last_year": int(group.year.max()),
            "means": {c: float(vals[c].mean()) for c in vals},
            "stds": {c: float(vals[c].std(ddof=1) if vals[c].std(ddof=1) > 0 else 1.0) for c in vals},
        }
    district_yield_profiles = {}
    state_yield_profiles = {}
    district_file = core["_find_csv"]("ICRISAT-District_Level_Data.csv")
    if district_file:
        district_raw = pd.read_csv(district_file)
        if {"State Name", "Dist Name", "Year"}.issubset(district_raw.columns):
            def comparable_crop_key(value: str) -> str:
                key = core["_crop_key"](value)
                return {"arhartur": "pigeonpea", "soyabean": "soybean", "bajra": "pearlmillet",
                        "sesamum": "sesame", "jowar": "sorghum"}.get(key, key)
            actual_names = {comparable_crop_key(name): name for name in crops}
            yield_name_map = {
                "RICE": "Rice", "WHEAT": "Wheat", "KHARIF SORGHUM": "Sorghum",
                "RABI SORGHUM": "Sorghum", "SORGHUM": "Sorghum", "PEARL MILLET": "Pearl millet",
                "MAIZE": "Maize", "FINGER MILLET": "Finger millet", "BARLEY": "Barley",
                "CHICKPEA": "Chickpea", "PIGEONPEA": "Pigeonpea", "MINOR PULSES": "Pulses",
                "GROUNDNUT": "Groundnut", "SESAMUM": "Sesame", "RAPESEED AND MUSTARD": "Rapeseed and mustard",
                "SAFFLOWER": "Safflower", "CASTOR": "Castor", "LINSEED": "Linseed",
                "SUNFLOWER": "Sunflower", "SOYABEAN": "Soybean", "OILSEEDS": "Oilseeds",
                "SUGARCANE": "Sugarcane", "COTTON": "Cotton",
            }
            profile_rows = []
            years_num = pd.to_numeric(district_raw["Year"], errors="coerce")
            for column in [c for c in district_raw if str(c).strip().upper().endswith(" YIELD (KG PER HA)")]:
                label = str(column).strip().upper().rsplit(" YIELD (KG PER HA)", 1)[0]
                generic_name = yield_name_map.get(label)
                if generic_name is None:
                    continue
                crop_key = comparable_crop_key(generic_name)
                if crop_key not in actual_names:
                    continue
                kg_ha = pd.to_numeric(district_raw[column], errors="coerce")
                valid = years_num.between(1997, 2020) & kg_ha.gt(0) & kg_ha.le(250000)
                if valid.any():
                    profile_rows.append(pd.DataFrame({
                        "state": district_raw.loc[valid, "State Name"].astype(str).str.strip().to_numpy(),
                        "district": district_raw.loc[valid, "Dist Name"].astype(str).str.strip().to_numpy(),
                        "crop": actual_names[crop_key], "yield_t_ha": kg_ha.loc[valid].to_numpy(dtype=float) / 1000.0,
                    }))
            if profile_rows:
                district_long = pd.concat(profile_rows, ignore_index=True)
                district_medians = district_long.groupby(["state", "district", "crop"]).yield_t_ha.median()
                for (state, district, crop), value in district_medians.items():
                    district_yield_profiles.setdefault(state, {}).setdefault(district, {})[crop] = float(value)
                state_medians = district_long.groupby(["state", "crop"]).yield_t_ha.median()
                for (state, crop), value in state_medians.items():
                    state_yield_profiles.setdefault(state, {})[crop] = float(value)
    input_names = [x.name for x in merged.graph.input]
    output_names = [x.name for x in merged.graph.output]
    metadata = {
        "model_file": MODEL_PATH.name,
        "model_version": "1.2.0",
        "json_schema_version": "agrivision_outputs_v1",
        "model_inputs": input_names,
        "model_outputs": output_names,
        "crop_output": next(x.name for x in merged.graph.output if "crop_" in x.name),
        "fertilizer_output": next(x.name for x in merged.graph.output if "fert_" in x.name),
        "climate_output": next(x.name for x in merged.graph.output if "climate_" in x.name),
        "crops": crops,
        "crop_feature_names": [f"crop:{c}" for c in crops] + SOIL_FEATURES,
        "soil_feature_names": SOIL_FEATURES,
        "fertilizer_feature_names": FERT_FEATURES,
        "states": states,
        "crops_by_state": crops_by_state,
        "crops_by_season": crops_by_season,
        "default_features": feature_medians,
        "training_feature_support": {
            feature: {"min": float(data[feature].min()), "q01": float(data[feature].quantile(.01)),
                      "q99": float(data[feature].quantile(.99)), "max": float(data[feature].max())}
            for feature in SOIL_FEATURES
        },
        "crop_median_yield_t_ha": {str(crop): float(group.yield_t_ha.median())
                                   for crop, group in data.groupby("crop")},
        "yield_model_selection_by_crop": yield_model_selection,
        "state_feature_defaults": state_defaults,
        "market_prices": prices,
        "regional_market_prices": regional_market_prices,
        "state_climate_summary": state_climate,
        "climate_residual_sd": {k: float(v) for k, v in zip(["temperature", "rainfall", "humidity"], climate_residual)},
        "climate_residual_sd_by_state": climate_residual_by_state,
        "climate_residuals_by_state": climate_residuals_by_state,
        "pooled_climate_residuals": pooled_climate_residuals,
        "climate_projection_bounds_by_state": climate_projection_bounds_by_state,
        "climate_year_center": climate_year_center,
        "climate_geographic_resolution": "state",
        "district_climate_available": False,
        "district_yield_profiles": district_yield_profiles,
        "state_yield_profiles": state_yield_profiles,
        "fertilizer_products": [
            {"name": "Urea", "n": .46, "p": 0, "k": 0, "cost": 6.2, "max_kg_ha": 220},
            {"name": "DAP", "n": .18, "p": .46, "k": 0, "cost": 27.0, "max_kg_ha": 180},
            {"name": "MOP", "n": 0, "p": 0, "k": .60, "cost": 17.0, "max_kg_ha": 160},
            {"name": "NPK 15-15-15", "n": .15, "p": .15, "k": .15, "cost": 24.0, "max_kg_ha": 250},
        ],
        "fertilizer_safe_targets_npk": {c: float(data[c].quantile(.65)) for c in ["N", "P", "K"]},
        "fertilizer_safe_targets_by_crop": {
            str(crop): {
                "N": min(150.0, max(45.0, float(group.N.quantile(.65)))),
                "P": min(110.0, max(35.0, float(group.P.quantile(.65)))),
                "K": min(150.0, max(35.0, float(group.K.quantile(.65)))),
            }
            for crop, group in data.groupby("crop")
        },
        "yield_model_metrics": yield_metrics,
        "training_rows": int(len(data)),
        "last_training_weather_year": int(climate.year.max()),
        "note": "Dataset-free inference bundle. Climate output is a deterministic scenario projection built from state trends plus bootstrapped historical annual residuals; it is not a year-specific weather forecast, and no district weather source is present. District crop yields use ICRISAT historical medians where available. Crop costs and validated fertilizer-treatment rates are absent; provide crop_cost_per_ha for profit estimates, and treat fertilizer quantities as empirical nutrient-target estimates. Mandi prices are historical, use region-specific records when available, and exclude quotes outside 0.2x-5x of each commodity's historical median.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"model": str(MODEL_PATH), "metadata": str(METADATA_PATH),
                      "crop_count": len(crops), "states": len(states), "metrics": yield_metrics,
                      "onnx_inputs": input_names, "onnx_outputs": output_names}, indent=2))


if __name__ == "__main__":
    main()
