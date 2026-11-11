"""AgriVision: single-file Kaggle starter for crop, climate-risk, and fertilizer engines.

Run as a Kaggle notebook cell/script. Results are deterministic and are written as
one JSON file; no charts or intermediate output files are generated. Add your data
with Kaggle Input or set AGRIVISION_DATA_DIR to its directory.

This is a decision-support prototype, not agronomic advice. Replace demo data and
validate locally before using recommendations in the field.
"""
from __future__ import annotations

import os
import random
import warnings
import hashlib
from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# --------------------------- Configuration ---------------------------------
DATA_DIR = os.environ.get("AGRIVISION_DATA_DIR", "/kaggle/input")
OUTPUT_JSON_PATH = os.environ.get("AGRIVISION_JSON_PATH", "/kaggle/working/agrivision_outputs.json")
N_MONTE_CARLO = 10_000

# %% [KAGGLE INPUT CELL — copy this block into its own Kaggle cell and edit it]
# Edit the values in this one dictionary for each run. Keep the field names intact.
# Replace only the values you want to change; use None to leave an optional value unset.
# Use exact state/district/season names from your CSVs. Set unknown values to None.
# risk_tolerance options: "low", "medium", "high".
# climate_scenario options: "normal", "drought-prone", "flood-prone".
# optimization_goal options: "minimize_cost", "maximize_yield", "balance_roi".
# land_unit options: "acres", "hectares". Confidence is a fraction: 0.90 or 0.95.
# Crop/fertilizer include/exclude fields accept lists, e.g. ["Rice", "Wheat"].
# Prices are INR/tonne. Budgets are INR/acre. Soil N/P/K are kg/ha; pH is unitless.
USER_INPUTS = {
    "location": {"state": None, "district": None},
    "season": None,                 # "Kharif", "Rabi", "Summer", or None
    "land_size": 1.0,
    "land_unit": "acres",           # "acres" or "hectares"
    "crop_budget_per_acre": None,   # INR per acre; None disables crop budget filter
    "risk_tolerance": "medium",     # "low", "medium", or "high"
    "include_crops": [],             # e.g. ["Rice", "Maize"]
    "exclude_crops": [],             # e.g. ["Cotton"]
    "price_override": None,          # INR/tonne; or {"Rice": 22000, ...}
    "crop_cost_per_ha": {},            # Optional local production costs by crop; empty means cost is unknown
    "target_year": 2027,             # None uses the horizon below
    "horizon_months": 12,            # used only when weather records are monthly
    "horizon_years": 1,              # used when source weather is annual (current files are annual)
    "climate_scenario": "normal",    # "normal", "drought-prone", "flood-prone"
    "confidence_level": 0.90,        # 0.90 or 0.95; adverse confidence bound feeds risk score
    "manual_weather": {              # optional override; use None to ignore each reading
        "temperature": None, "rainfall": None, "humidity": None,
    },
    "fertilizer_crop": None,         # None uses the top crop recommendation
    "soil_test": {"N": None, "P": None, "K": None, "ph": None},
    "fertilizer_budget_per_acre": None,
    "optimization_goal": "balance_roi",  # "minimize_cost", "maximize_yield", "balance_roi"
    "preferred_fertilizers": [],     # e.g. ["Urea"]
    "excluded_fertilizers": [],      # e.g. ["DAP"]
    "fertilizer_prices_per_kg": {    # local prices; edit to current supplier prices (INR/kg)
        "Urea": 6.2, "DAP": 27.0, "MOP": 17.0, "NPK 15-15-15": 24.0,
    },
}
# %% [END KAGGLE INPUT CELL]

# Expected crop_yield.csv columns: N, P, K, temperature, humidity, ph, rainfall,
# crop (or label), yield_t_ha; optional price_per_ton, cost_per_ha.
# Expected weather.csv: date, temperature, rainfall; optional humidity.


def _find_csv(filename: str) -> Optional[str]:
    """Find a named CSV under Kaggle input or AGRIVISION_DATA_DIR."""
    direct = os.path.join(DATA_DIR, filename)
    if os.path.isfile(direct):
        return direct
    wanted = filename.lower()
    for root, _, files in os.walk(DATA_DIR):
        for name in files:
            if name.lower() == wanted:
                return os.path.join(root, name)
    return None


def _find_csv_containing(*tokens: str) -> Optional[str]:
    """Find a CSV by filename keywords when Kaggle renamed a source file."""
    wanted = [t.lower() for t in tokens]
    for root, _, files in os.walk(DATA_DIR):
        for name in files:
            low = name.lower()
            if low.endswith(".csv") and all(t in low for t in wanted):
                return os.path.join(root, name)
    return None


def make_demo_data(n: int = 1800) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create plausible synthetic data for a reproducible end-to-end demo."""
    rng = np.random.default_rng(RANDOM_STATE)
    crops = np.array(["rice", "maize", "chickpea", "cotton", "banana", "apple"])
    centers = {
        "rice": (95, 42, 42, 27, 82, 6.2, 210), "maize": (78, 48, 20, 24, 62, 6.5, 95),
        "chickpea": (42, 62, 78, 21, 48, 7.1, 68), "cotton": (115, 48, 38, 27, 58, 7.0, 78),
        "banana": (100, 82, 52, 27, 78, 6.1, 115), "apple": (22, 135, 198, 16, 58, 6.0, 115),
    }
    crop = rng.choice(crops, n, p=[.22, .22, .16, .16, .13, .11])
    cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = np.vstack([np.array(centers[c]) + rng.normal(0, [22, 18, 25, 5, 13, .65, 38]) for c in crop])
    df = pd.DataFrame(X, columns=cols).clip(lower=[0, 0, 0, 3, 10, 3.5, 0], upper=[180, 160, 220, 42, 100, 9, 320])
    df["crop"] = crop
    ideal = np.vstack([np.array(centers[c]) for c in crop])
    dist = np.mean(((X - ideal) / np.array([50, 45, 60, 12, 25, 1.5, 100])) ** 2, axis=1)
    df["yield_t_ha"] = np.maximum(.3, (5.0 + 4.0 * np.exp(-2.7 * dist) + rng.normal(0, .65, n)))
    prices = {"rice": 320, "maize": 230, "chickpea": 720, "cotton": 1450, "banana": 510, "apple": 980}
    df["price_per_ton"] = df.crop.map(prices) * rng.lognormal(0, .08, n)
    df["cost_per_ha"] = rng.normal(850, 110, n).clip(500, 1300)
    dates = pd.date_range("2016-01-01", periods=120, freq="MS")
    t = np.arange(len(dates))
    weather = pd.DataFrame({"date": dates,
        "temperature": 25 + 5*np.sin(2*np.pi*t/12) + .025*t + rng.normal(0, 1.6, len(t)),
        "rainfall": np.maximum(0, 75 + 65*np.sin(2*np.pi*(t-2)/12) + rng.normal(0, 26, len(t))),
        "humidity": np.clip(62 + 18*np.sin(2*np.pi*(t-1)/12) + rng.normal(0, 7, len(t)), 20, 100)})
    return df, weather


FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def _canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map common Indian agriculture dataset headers to this script's schema."""
    d = frame.copy()
    def key(s: object) -> str:
        return "".join(ch for ch in str(s).lower() if ch.isalnum())
    aliases = {
        "N": ["n", "nitrogen", "nkgperha", "nitrogenkgperha", "nitrogencontent", "nitrogenkg", "nitrogenlevel"],
        "P": ["p", "phosphorus", "phosphorous", "pkgperha", "phosphoruskgperha", "phosphorouscontent", "phosphoruskg", "phosphoruslevel"],
        "K": ["k", "potassium", "kkgperha", "potassiumkgperha", "potassiumcontent", "potassiumkg", "potassiumlevel"],
        "temperature": ["temperature", "temp", "avgtemp", "avgtempc", "meantemperature", "temperaturec", "meantemp", "temperaturemean"],
        "humidity": ["humidity", "relativehumidity", "humiditypercent", "avghumiditypercent"],
        "ph": ["ph", "soilph", "soilphvalue"],
        "rainfall": ["rainfall", "annualrainfall", "precipitation", "rain", "rainfallmm", "totalrainfallmm"],
        "crop": ["crop", "croptype", "cropname", "commodity", "cropcategory"],
        "yield_t_ha": ["yield", "yieldtha", "yieldperhectare", "yieldtonneshectare", "yieldkgperha", "cropyield", "productionyield"],
        "production": ["production", "productiontonnes", "productiontons"],
        "area": ["area", "areahectare", "areahectares", "cultivatedarea"],
        "date": ["date", "datetime", "month", "yearmonth", "arrivaldate"],
        "price_per_ton": ["price", "modalprice", "priceperton", "marketprice", "modalpriceinrsquintal"],
        "cost_per_ha": ["costperha", "costperhectare", "cost"],
        "state": ["state", "statename", "province", "region"],
        "district": ["district", "districtname", "tehsil", "taluk"],
        "season": ["season", "cropseason", "growingseason"],
        "year": ["year", "cropyear", "harvestyear"],
    }
    existing = {key(c): c for c in d.columns}
    renames = {}
    for target, names in aliases.items():
        if target not in d.columns:
            for name in names:
                if name in existing:
                    renames[existing[name]] = target
                    break
    d = d.rename(columns=renames)
    # Numeric conversion is tolerant of commas, currency symbols, and blanks.
    for c in [*FEATURES, "yield_t_ha", "production", "area", "price_per_ton", "cost_per_ha", "year"]:
        if c in d:
            d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", "", regex=False).str.replace("₹", "", regex=False), errors="coerce")
    if "yield_t_ha" not in d and {"production", "area"}.issubset(d.columns):
        d["yield_t_ha"] = d.production / d.area.replace(0, np.nan)
    return d


def _crop_key(value: object) -> str:
    """Normalize common crop-name spelling variants for price/cost joins."""
    key = "".join(ch for ch in str(value).casefold() if ch.isalnum())
    return {
        "cashewnut": "cashewnuts",
        "cardamom": "cardamoms",
        "horsegram": "kulthihorsegram",
        "kulthihorsegram": "kulthihorsegram",
    }.get(key, key)


def _location_key(value: object) -> str:
    """Normalize common state spelling/punctuation variants for dataset joins."""
    key = "".join(ch for ch in str(value).casefold() if ch.isalnum())
    return {
        "chattisgarh": "chhattisgarh",
        "jammukashmir": "jammuandkashmir",
        "orissa": "odisha",
        "pondicherry": "puducherry",
        "uttaranchal": "uttarakhand",
    }.get(key, key)


def _prepare_training_data(raw: pd.DataFrame, weather: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Use the provided fertilizer table for NPK suitability when historical yield
    data has no soil nutrients; attach crop-level historical yield where possible."""
    d = _canonicalize_columns(raw)
    required = set(FEATURES + ["crop", "yield_t_ha"])
    if required.issubset(d.columns):
        return d
    fertilizer_path = (_find_csv("fertilizer_recommendation.csv")
                       or _find_csv_containing("fertilizer", "recommend")
                       or _find_csv_containing("fertilizer"))
    # Prefer actual crop/yield observations and join the state soil/weather tables.
    # A separate fertilizer-label table is not a yield dataset and must not be used
    # to invent crop yields by copying crop-level averages onto every fertilizer row.
    if {"crop", "yield_t_ha"}.issubset(d.columns):
        fertilizer_path = None
    if fertilizer_path:
        fert = _canonicalize_columns(pd.read_csv(fertilizer_path))
        # Typical fertilizer dataset uses crop type plus nitrogen/phosphorous/potassium.
        if {"crop", "N", "P", "K"}.issubset(fert.columns):
            d_crop = "crop" if "crop" in d else None
            d_yield = "yield_t_ha" if "yield_t_ha" in d else None
            if d_crop and d_yield:
                yield_by_crop = d.groupby(d.crop.astype(str).str.strip().str.lower()).yield_t_ha.median()
                fert["crop_key"] = fert.crop.astype(str).str.strip().str.lower()
                fert["yield_t_ha"] = fert.crop_key.map(yield_by_crop)
                global_yield = float(d.yield_t_ha.median())
                fert["yield_t_ha"] = fert.yield_t_ha.fillna(global_yield)
            elif "yield_t_ha" in fert:
                pass
            else:
                fert["yield_t_ha"] = float(d.yield_t_ha.median()) if "yield_t_ha" in d else 3.0
            # Use observed climate medians as context when fertilizer rows lack climate.
            if weather is not None:
                w = _canonicalize_columns(weather)
                for col in ["temperature", "humidity", "rainfall"]:
                    if col in w and w[col].notna().any():
                        fert[col] = fert[col].fillna(float(w[col].median())) if col in fert else float(w[col].median())
            for col, default in [("temperature", 25.0), ("humidity", 60.0), ("ph", 6.5), ("rainfall", 100.0)]:
                if col not in fert: fert[col] = float(d[col].median()) if col in d and d[col].notna().any() else default
                else: fert[col] = fert[col].fillna(float(d[col].median()) if col in d and d[col].notna().any() else default)
            # Keep original state/yield records for diagnostics and use NPK observations
            # from the fertilizer dataset as the crop recommendation/nutrient training set.
            return fert
    # Fallback for Kaggle setups where only crop-yield/weather files were added
    # as notebook inputs. Use state soil measurements, if present, for nutrient
    # features; remaining climate gaps use observed weather medians. The fallback
    # is explicitly reported because it is weaker than a crop-labeled fertilizer table.
    soil_path = _find_csv("state_soil_data.csv") or _find_csv_containing("soil", "data")
    soil = _canonicalize_columns(pd.read_csv(soil_path)) if soil_path else pd.DataFrame()
    weather_path = _find_csv("state_weather_data_1997_2020.csv") or _find_csv_containing("weather", "1997")
    if weather_path and "state" in d:
        regional_weather = _canonicalize_columns(pd.read_csv(weather_path))
        weather_cols = [c for c in ["temperature", "humidity", "rainfall"] if c in regional_weather]
        if "state" in regional_weather and weather_cols:
            # State spellings commonly differ by capitalization or whitespace
            # between public CSVs. Normalize both sides before the state/year join.
            d["_state_join_key"] = d.state.map(_location_key)
            regional_weather["_state_join_key"] = regional_weather.state.map(_location_key)
            join_keys = ["_state_join_key"]
            if "year" in d and "year" in regional_weather:
                d["_year_join_key"] = pd.to_numeric(d.year, errors="coerce").astype("Int64")
                regional_weather["_year_join_key"] = pd.to_numeric(regional_weather.year, errors="coerce").astype("Int64")
                join_keys.append("_year_join_key")
            regional_weather = regional_weather[join_keys + weather_cols].groupby(join_keys, as_index=False).mean(numeric_only=True)
            regional_weather = regional_weather.rename(columns={c: f"{c}_weather" for c in weather_cols})
            d = d.merge(regional_weather, on=join_keys, how="left")
            d = d.drop(columns=[c for c in ["_state_join_key", "_year_join_key"] if c in d])
            for col in weather_cols:
                joined = f"{col}_weather"
                if joined in d:
                    if col not in d: d[col] = d[joined]
                    else: d[col] = d[col].fillna(d[joined])
    if "state" in d and "state" in soil:
        soil_cols = [c for c in ["state", "N", "P", "K", "ph"] if c in soil]
        if len(soil_cols) > 1:
            soil_state = soil[soil_cols].groupby("state", as_index=False).median(numeric_only=True)
            d["_state_key"] = d.state.map(_location_key)
            soil_state["_state_key"] = soil_state.state.map(_location_key)
            d = d.merge(soil_state.drop(columns="state"), on="_state_key", how="left", suffixes=("", "_soil"))
            d = d.drop(columns="_state_key")
    for col in ["N", "P", "K", "ph", "temperature", "humidity", "rainfall"]:
        if col not in d:
            if col in soil and soil[col].notna().any():
                d[col] = float(soil[col].median())
            elif col in ("temperature", "humidity", "rainfall") and weather is not None and col in weather and weather[col].notna().any():
                d[col] = float(weather[col].median())
            else:
                d[col] = {"N": 70.0, "P": 45.0, "K": 45.0, "ph": 6.5,
                          "temperature": 25.0, "humidity": 60.0, "rainfall": 100.0}[col]
        elif d[col].isna().any():
            if col in soil and soil[col].notna().any():
                fill_value = float(soil[col].median())
            elif weather is not None and col in weather and weather[col].notna().any():
                fill_value = float(weather[col].median())
            else:
                fill_value = float(d[col].median()) if d[col].notna().any() else 0.0
            d[col] = d[col].fillna(fill_value)
    if soil_path and {"crop", "yield_t_ha"}.issubset(d.columns):
        print(f"Using observed crop/yield rows enriched with state soil measurements from: {soil_path}")
    else:
        print("No usable fertilizer recommendation table found; missing NPK/pH/climate fields were filled with neutral defaults. Treat nutrient optimizer results as demonstration only.")
    return d


def _district_yield_history(location: Dict[str, object], known_crops: Sequence[str]) -> pd.DataFrame:
    """Load real district/year crop yields from the optional ICRISAT source.

    ICRISAT reports yield in kg/ha and has no soil or weather columns. Keep only
    its overlapping weather years, convert units explicitly, and enrich it with
    state soil/weather context through _prepare_training_data.
    """
    district = location.get("district")
    if not district:
        return pd.DataFrame()
    path = _find_csv("ICRISAT-District_Level_Data.csv")
    if not path:
        return pd.DataFrame()
    raw = pd.read_csv(path)
    required = {"Dist Name", "State Name", "Year"}
    if not required.issubset(raw.columns):
        return pd.DataFrame()
    mask = raw["Dist Name"].map(_location_key).eq(_location_key(district))
    state = location.get("state")
    if state:
        mask &= raw["State Name"].map(_location_key).eq(_location_key(state))
    raw = raw.loc[mask].copy()
    if raw.empty:
        return pd.DataFrame()
    crop_names = {
        "RICE": "Rice", "WHEAT": "Wheat", "KHARIF SORGHUM": "Sorghum",
        "RABI SORGHUM": "Sorghum", "SORGHUM": "Sorghum", "PEARL MILLET": "Pearl millet",
        "MAIZE": "Maize", "FINGER MILLET": "Finger millet", "BARLEY": "Barley",
        "CHICKPEA": "Chickpea", "PIGEONPEA": "Pigeonpea", "MINOR PULSES": "Pulses",
        "GROUNDNUT": "Groundnut", "SESAMUM": "Sesame", "RAPESEED AND MUSTARD": "Mustard",
        "SAFFLOWER": "Safflower", "CASTOR": "Castor", "LINSEED": "Linseed",
        "SUNFLOWER": "Sunflower", "SOYABEAN": "Soybean", "OILSEEDS": "Oilseeds",
        "SUGARCANE": "Sugarcane", "COTTON": "Cotton",
    }
    def comparable_crop_key(value: object) -> str:
        key = _crop_key(value)
        return {"arhartur": "pigeonpea", "soyabean": "soybean", "bajra": "pearlmillet"}.get(key, key)
    known = {comparable_crop_key(name): str(name) for name in known_crops}
    yield_columns = [c for c in raw.columns if str(c).strip().upper().endswith(" YIELD (KG PER HA)")]
    rows = []
    for col in yield_columns:
        label = str(col).strip().upper().rsplit(" YIELD (KG PER HA)", 1)[0]
        canonical = crop_names.get(label)
        if canonical is None:
            continue
        key = comparable_crop_key(canonical)
        if key not in known:
            continue
        values = pd.to_numeric(raw[col], errors="coerce")
        years = pd.to_numeric(raw["Year"], errors="coerce")
        valid = years.between(1997, 2020) & values.gt(0) & values.le(250000)
        if not valid.any():
            continue
        rows.append(pd.DataFrame({
            "state": raw.loc[valid, "State Name"].astype(str).str.strip().to_numpy(),
            "district": raw.loc[valid, "Dist Name"].astype(str).str.strip().to_numpy(),
            "year": years.loc[valid].to_numpy(), "crop": known[key],
            "yield_t_ha": (values.loc[valid].to_numpy() / 1000.0),
            "_district_yield_proxy": True,
        }))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).groupby(
        ["state", "district", "year", "crop", "_district_yield_proxy"], as_index=False
    ).yield_t_ha.mean()


def _prepare_weather(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardize common state/district weather files to monthly date/temp/rain."""
    w = _canonicalize_columns(raw)
    cols = {"date": next((c for c in w if "date" in c.lower()), None),
            "year": next((c for c in w if c.lower() in ("year", "yearofrecord")), None),
            "month": next((c for c in w if c.lower() in ("month", "monthname", "monthnumber")), None)}
    # Handle split min/max temperature columns.
    if "temperature" not in w:
        tcols = [c for c in w if any(k in c.lower() for k in ("temp", "tmax", "tmin"))]
        if tcols:
            w["temperature"] = w[tcols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    if "rainfall" not in w:
        rcols = [c for c in w if "rain" in c.lower() or "precip" in c.lower()]
        if rcols: w["rainfall"] = w[rcols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    if "date" not in w:
        if cols["year"] and cols["month"]:
            year_text = pd.to_numeric(w[cols["year"]], errors="coerce").astype("Int64").astype(str)
            month_text = pd.to_numeric(w[cols["month"]], errors="coerce").astype("Int64").astype(str)
            w["date"] = pd.to_datetime(year_text + "-" + month_text + "-01", errors="coerce")
        elif cols["year"]:
            annual = w.groupby(cols["year"], as_index=False).agg({c: "mean" for c in ["temperature", "humidity", "rainfall"] if c in w})
            year_text = pd.to_numeric(annual[cols["year"]], errors="coerce").astype("Int64").astype(str)
            annual["date"] = pd.to_datetime(year_text + "-01-01", errors="coerce")
            # ARIMA routine consumes monthly series. Annual observations are retained
            # as dated observations and forecast support remains functional.
            w = annual
    if "date" not in w or "temperature" not in w or "rainfall" not in w:
        raise ValueError("Weather CSV needs a date/year plus temperature and rainfall columns; recognized common aliases automatically.")
    if "humidity" not in w: w["humidity"] = np.nan
    return w[["date", "temperature", "rainfall", "humidity"]].dropna(subset=["date"])


class CropProfitEngine:
    """Observed-yield model and cost-aware crop ranking."""
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.features = FEATURES
        self.crop_model = None  # No labeled crop-suitability outcome exists in the source data.
        self.yield_model_name = "Per-crop validation-selected model or crop-median baseline"
        self.metrics: Dict[str, float] = {}

    def fit(self) -> "CropProfitEngine":
        d = self.data.dropna(subset=["crop", "yield_t_ha"] + [c for c in self.features if c in self.data])
        # This dataset's Coconut yield field mixes nut counts with tonnes/ha and
        # reaches values above 20,000. Such values are not comparable as t/ha.
        invalid_yield = d.yield_t_ha > 250
        self.invalid_yield_rows = int(invalid_yield.sum())
        d = d.loc[~invalid_yield].copy()
        self.data = d
        if d.empty:
            missing_counts = self.data[self.features + ["crop", "yield_t_ha"]].isna().sum().to_dict()
            raise ValueError(
                "Crop model has zero complete training rows after dropping missing values. "
                f"Input rows={len(self.data)}; missing by required column={missing_counts}. "
                "Check the crop-yield, soil, and state/year weather joins."
            )
        # Crop identity in a historical yield table is not a label for crop
        # suitability. A classifier trained on those rows learns past planting
        # patterns, so it must not be presented as a suitability probability.
        # Train yield models on the full, state-enriched data instead of only the
        # selected state: otherwise N/P/K/pH become constants and farmer soil
        # overrides are silently dropped from the model's active features.
        self.features = [c for c in FEATURES if c in d and d[c].nunique(dropna=True) > 1]
        if not self.features:
            self.features = [c for c in FEATURES if c in d]
        self.classifier_training_source = "not_used_crop_identity_is_not_a_suitability_label"
        self.classifier_training_rows = 0
        self.classifier_is_informative = False
        self.metrics["crop_accuracy"] = None
        self.metrics["crop_majority_baseline_accuracy"] = None
        self.yield_models = {}
        self.yield_baselines = {}
        self.per_crop_yield_metrics = {}
        for crop_name, group in d.groupby("crop"):
            crop_key = str(crop_name)
            baseline = float(group.yield_t_ha.median())
            self.yield_baselines[crop_key] = baseline
            if len(group) >= 25:
                if "year" in group and pd.to_numeric(group.year, errors="coerce").nunique() >= 4:
                    years = sorted(pd.to_numeric(group.year, errors="coerce").dropna().unique())
                    cutoff = years[max(1, int(len(years) * .8) - 1)]
                    train_mask = pd.to_numeric(group.year, errors="coerce") <= cutoff
                    test_mask = ~train_mask
                    if train_mask.sum() < 10 or test_mask.sum() < 5:
                        train_mask, test_mask = None, None
                else:
                    train_mask, test_mask = None, None
                if train_mask is None:
                    train_idx, test_idx = train_test_split(np.arange(len(group)), test_size=.2, random_state=RANDOM_STATE)
                    train_group, test_group = group.iloc[train_idx], group.iloc[test_idx]
                    validation_split = "deterministic_random_fallback"
                else:
                    train_group, test_group = group.loc[train_mask], group.loc[test_mask]
                    validation_split = "chronological_holdout_by_year"
                X_train_crop, X_test_crop = train_group[self.features], test_group[self.features]
                y_train_crop, y_test_crop = train_group.yield_t_ha, test_group.yield_t_ha
                model = HistGradientBoostingRegressor(max_iter=180, learning_rate=.05, max_leaf_nodes=15,
                    l2_regularization=2.0, random_state=RANDOM_STATE)
                model.fit(X_train_crop, y_train_crop)
                model_mae = float(mean_absolute_error(y_test_crop, np.maximum(0, model.predict(X_test_crop))))
                baseline_mae = float(mean_absolute_error(y_test_crop, np.full(len(y_test_crop), float(y_train_crop.median()))))
                self.per_crop_yield_metrics[crop_key] = {
                    "model_mae_t_ha": model_mae, "median_baseline_mae_t_ha": baseline_mae,
                    "validation_split": validation_split,
                    "selected_model": "histogram_gradient_boosting" if model_mae < baseline_mae * .95 else "crop_median",
                    "selected_mae_t_ha": model_mae if model_mae < baseline_mae * .95 else baseline_mae,
                }
                if model_mae < baseline_mae * .95:
                    model.fit(group[self.features], group.yield_t_ha)
                    self.yield_models[crop_key] = model
            elif len(group) >= 5:
                if "year" in group and pd.to_numeric(group.year, errors="coerce").nunique() >= 4:
                    years = sorted(pd.to_numeric(group.year, errors="coerce").dropna().unique())
                    cutoff = years[max(1, int(len(years) * .8) - 1)]
                    train_group = group.loc[pd.to_numeric(group.year, errors="coerce") <= cutoff]
                    test_group = group.loc[pd.to_numeric(group.year, errors="coerce") > cutoff]
                    if train_group.empty or test_group.empty:
                        train_idx, test_idx = train_test_split(np.arange(len(group)), test_size=.2, random_state=RANDOM_STATE)
                        train_group, test_group = group.iloc[train_idx], group.iloc[test_idx]
                else:
                    train_idx, test_idx = train_test_split(np.arange(len(group)), test_size=.2, random_state=RANDOM_STATE)
                    train_group, test_group = group.iloc[train_idx], group.iloc[test_idx]
                y_train_crop, y_test_crop = train_group.yield_t_ha, test_group.yield_t_ha
                baseline_mae = float(mean_absolute_error(y_test_crop, np.full(len(y_test_crop), float(y_train_crop.median()))))
                self.per_crop_yield_metrics[crop_key] = {
                    "model_mae_t_ha": None, "median_baseline_mae_t_ha": baseline_mae,
                    "validation_split": "chronological_or_deterministic_holdout",
                    "selected_model": "crop_median", "selected_mae_t_ha": baseline_mae,
                }
        selected_maes = [m["selected_mae_t_ha"] for m in self.per_crop_yield_metrics.values()]
        self.metrics["per_crop_macro_mae_t_ha"] = float(np.mean(selected_maes)) if selected_maes else float("nan")
        return self

    def recommend(self, soil_weather: Dict[str, float], top_n: int = 5,
                  include_crops: Optional[Sequence[str]] = None,
                  exclude_crops: Optional[Sequence[str]] = None,
                  price_override: Optional[object] = None,
                  budget_per_ha: Optional[float] = None,
                  risk_tolerance: str = "medium",
                  reference_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        row = pd.DataFrame([{k: soil_weather[k] for k in self.features}])
        local_data = reference_data if reference_data is not None and not reference_data.empty else self.data
        available_crops = set(local_data.crop.astype(str).unique())
        candidates = pd.DataFrame({"crop": sorted(available_crops), "suitability_probability": [None] * len(available_crops)})
        recommendation_basis = "yield_proxy_cost_data_missing"
        if include_crops:
            allowed = {str(c).strip().lower() for c in include_crops}
            candidates = candidates[candidates.crop.astype(str).str.lower().isin(allowed)]
        if exclude_crops:
            excluded = {str(c).strip().lower() for c in exclude_crops}
            candidates = candidates[~candidates.crop.astype(str).str.lower().isin(excluded)]
        if candidates.empty:
            raise ValueError("No candidate crops remain after applying the include/exclude preferences.")
        if self.classifier_is_informative:
            candidates = candidates.sort_values("suitability_probability", ascending=False).head(top_n)
        crops = candidates.crop.to_numpy()
        probs_for_crops = candidates.suitability_probability.to_numpy()
        # Model is trained globally; produce a yield estimate and profit distribution
        # for each suggested crop using that crop's observed median yield as a baseline.
        out = []
        tolerance = str(risk_tolerance).strip().lower()
        risk_weight = {"low": .75, "medium": .45, "high": .20}.get(tolerance, .45)
        for crop, prob in zip(crops, probs_for_crops):
            subset = local_data[local_data.crop.astype(str) == str(crop)]
            if subset.empty:
                subset = self.data[self.data.crop.astype(str) == str(crop)]
            sample = row.copy()
            sample["N"], sample["P"], sample["K"] = soil_weather["N"], soil_weather["P"], soil_weather["K"]
            yield_model = self.yield_models.get(str(crop))
            estimated_yield = (float(max(0, yield_model.predict(sample[self.features])[0]))
                               if yield_model is not None else float(self.yield_baselines[str(crop)]))
            # When district-level historical yield rows are supplied, adjust the
            # state-trained estimate by the observed district/state yield ratio.
            if "_district_yield_proxy" in subset and subset["_district_yield_proxy"].fillna(False).any():
                district_median = float(subset.yield_t_ha.median())
                state_rows = self.data[self.data.crop.astype(str) == str(crop)]
                state_median = float(state_rows.yield_t_ha.median()) if not state_rows.empty else 0.0
                if state_median > 0:
                    estimated_yield *= district_median / state_median
            price = float(subset.price_per_ton.median()) if "price_per_ton" in subset and subset.price_per_ton.notna().any() else None
            price_source = str(subset.price_source.dropna().iloc[0]) if "price_source" in subset and subset.price_source.notna().any() else ("crop_yield_csv" if price is not None else "unavailable")
            price_date = subset.price_observation_date.dropna().max() if "price_observation_date" in subset and subset.price_observation_date.notna().any() else None
            if isinstance(price_override, dict):
                override = next((v for k, v in price_override.items() if _crop_key(k) == _crop_key(crop)), None)
                if override is not None:
                    price = float(override); price_source = "user_override"; price_date = None
            elif price_override is not None:
                price = float(price_override); price_source = "user_override"; price_date = None
            cost = float(subset.cost_per_ha.median()) if "cost_per_ha" in subset and subset.cost_per_ha.notna().any() else None
            cost_source = str(subset.cost_source.dropna().iloc[0]) if "cost_source" in subset and subset.cost_source.notna().any() else ("crop_yield_csv" if cost is not None else "unavailable")
            revenue = estimated_yield * price if price is not None else None
            sim = self.simulate_profit(estimated_yield, price, cost) if price is not None and cost is not None else None
            risk_adjusted = sim["mean_profit"] + risk_weight * (sim["p10_profit"] - sim["mean_profit"]) if sim else None
            out.append({"crop": crop, "suitability_probability": prob, "estimated_yield_t_ha": estimated_yield,
                        "recommendation_basis": recommendation_basis,
                        "estimated_cost_per_ha": cost, "cost_source": cost_source,
                        "price_per_ton": price, "price_source": price_source, "price_observation_date": price_date,
                        "expected_gross_revenue_per_ha": revenue,
                        "expected_profit_per_ha": sim["mean_profit"] if sim else None,
                        "profit_p10": sim["p10_profit"] if sim else None,
                        "profit_p90": sim["p90_profit"] if sim else None,
                        "probability_of_loss": sim["probability_loss"] if sim else None,
                        "risk_adjusted_profit_per_ha": risk_adjusted,
                        "within_budget": (None if cost is None else True if budget_per_ha is None else cost <= budget_per_ha)})
        result = pd.DataFrame(out)
        result["_has_profit"] = result.expected_profit_per_ha.notna()
        # A price is revenue data; without crop production costs it cannot support
        # a profit ranking. Use the yield proxy and label that limitation.
        has_profit = bool(result.expected_profit_per_ha.notna().any())
        result["ranking_basis"] = "risk_adjusted_profit_with_supplied_costs" if has_profit else "estimated_yield_proxy_cost_data_missing"
        ranking_value = result.risk_adjusted_profit_per_ha.where(result.risk_adjusted_profit_per_ha.notna(), result.estimated_yield_t_ha)
        result["_ranking_value"] = ranking_value
        result = result.sort_values(["_has_profit", "_ranking_value", "estimated_yield_t_ha"],
                                    ascending=[False, False, False], na_position="last").drop(columns=["_has_profit", "_ranking_value"]).reset_index(drop=True)
        if budget_per_ha is not None and result.within_budget.eq(True).any():
            result = result[(result.within_budget.eq(True)) | (result.within_budget.isna())].reset_index(drop=True)
        return result.head(top_n).reset_index(drop=True)

    @staticmethod
    def simulate_profit(yield_t_ha: float, price_per_ton: float, cost_per_ha: float,
                        price_volatility: float = .15, yield_volatility: float = .12,
                        n: int = N_MONTE_CARLO) -> Dict[str, float]:
        rng = np.random.default_rng(RANDOM_STATE)
        # Lognormal shocks preserve positive prices and yields; expected values are centered.
        price = price_per_ton * rng.lognormal(-.5*price_volatility**2, price_volatility, n)
        yld = yield_t_ha * rng.lognormal(-.5*yield_volatility**2, yield_volatility, n)
        profit = price * yld - cost_per_ha
        return {"mean_profit": float(np.mean(profit)), "median_profit": float(np.median(profit)),
                "p10_profit": float(np.percentile(profit, 10)), "p90_profit": float(np.percentile(profit, 90)),
                "probability_loss": float(np.mean(profit < 0))}


class WeatherClimateRiskEngine:
    """ARIMA climate trend, reproducible weather scenarios, and exposure index."""
    def __init__(self, weather: pd.DataFrame):
        w = weather.copy()
        w["date"] = pd.to_datetime(w.date)
        numeric = [c for c in ["temperature", "rainfall", "humidity"] if c in w]
        w = w.groupby("date", as_index=False)[numeric].mean(numeric_only=True)
        deltas = w.date.sort_values().diff().dt.days.dropna()
        self.is_annual = bool(len(deltas) and float(deltas.median()) > 180)
        if self.is_annual:
            w["date"] = pd.to_datetime(w.date.dt.year.astype(str) + "-01-01")
            self.weather = w.groupby("date")[numeric].mean().sort_index().asfreq("YS")
        else:
            self.weather = w.sort_values("date").set_index("date")[numeric].resample("MS").mean()
        self.weather[["temperature", "rainfall"]] = self.weather[["temperature", "rainfall"]].interpolate().ffill().bfill()

    def _risk_features(self) -> pd.DataFrame:
        w = self.weather.copy()
        w["temp_anomaly"] = w.temperature - w.temperature.rolling(12, min_periods=3).mean()
        w["rain_z"] = (w.rainfall - w.rainfall.rolling(12, min_periods=3).mean()) / (w.rainfall.rolling(12, min_periods=3).std() + 1e-6)
        w["temp_change"] = w.temperature.diff().fillna(0)
        X = w[["temperature", "rainfall", "temp_anomaly", "rain_z", "temp_change"]].replace([np.inf, -np.inf], 0).fillna(0)
        return X

    def fit(self) -> "WeatherClimateRiskEngine":
        # There are no observed crop-loss labels in the source, so do not train a
        # circular classifier from threshold labels generated from the same weather.
        self._X = self._risk_features()
        return self

    def forecast(self, periods: int = 12, confidence_level: float = .95,
                 climate_scenario: str = "normal") -> pd.DataFrame:
        confidence_level = min(.999, max(.50, float(confidence_level)))
        rows = []
        # Forecast humidity too. Previously it was filled with one historical
        # median in score(), which made every reported period look identical.
        forecast_columns = [c for c in ["temperature", "rainfall", "humidity"]
                            if c in self.weather and self.weather[c].notna().any()]
        for col in forecast_columns:
            s = self.weather[col].dropna()
            try:
                from statsmodels.tsa.arima.model import ARIMA
                # ARIMA is retained for annual point forecasts because a rolling
                # one-year holdout across the available states gave lower MAE
                # (0.404 °C) than a recent linear trend (0.423 °C). Smooth annual
                # means are expected; uncertainty appears in the confidence bands.
                order = (1, 1, 0) if self.is_annual else (1, 1, 1)
                seasonal_order = (0, 0, 0, 0) if self.is_annual else (1, 0, 1, 12)
                fitted = ARIMA(s, order=order, seasonal_order=seasonal_order).fit()
                f = fitted.get_forecast(periods, alpha=1-confidence_level)
                mean = np.asarray(f.predicted_mean)
                ci = np.asarray(f.conf_int())
            except Exception:
                # Drift + widening uncertainty fallback. Do not cycle annual data
                # into repeated values when ARIMA cannot fit.
                steps = np.arange(1, periods+1)
                if self.is_annual and len(s) > 1:
                    slope = float((s.iloc[-1] - s.iloc[0]) / max(1, len(s)-1))
                    mean = float(s.iloc[-1]) + slope * steps
                else:
                    seasonal = min(12, len(s))
                    mean = np.array([s.iloc[-seasonal + (i % seasonal)] for i in range(periods)])
                z = NormalDist().inv_cdf(.5 + confidence_level/2)
                spread = max(float(s.std()), 1.0) * np.sqrt(steps) * .25 * z/1.96
                ci = np.column_stack([mean - 1.96*spread, mean + 1.96*spread])
            if self.is_annual and len(s) >= 6:
                # A future annual mean alone is nearly a straight line and can
                # look like the same weather every year. Add a reproducible
                # bootstrap of observed detrended annual anomalies. This is a
                # scenario path, not a claim to know exact future weather.
                history = s.to_numpy(dtype=float)
                trend = np.polyval(np.polyfit(np.arange(len(history)), history, 1), np.arange(len(history)))
                anomalies = history - trend
                anomalies -= anomalies.mean()
                seed_material = col.encode("utf-8") + np.round(history, 6).astype(np.float64).tobytes()
                seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
                annual_shocks = np.random.default_rng(seed).choice(anomalies, size=periods, replace=True)
                mean = np.asarray(mean, dtype=float) + annual_shocks
                ci = np.asarray(ci, dtype=float) + annual_shocks[:, None]
            scenario = str(climate_scenario).strip().lower()
            if col == "temperature" and scenario == "drought-prone":
                mean = mean + 1.0; ci = ci + 1.0
            elif col == "rainfall" and scenario == "drought-prone":
                mean = mean * .70; ci = ci * .70
            elif col == "rainfall" and scenario == "flood-prone":
                mean = mean * 1.30; ci = ci * 1.30
            if col == "rainfall":
                mean = np.maximum(mean, 0)
                ci = np.maximum(ci, 0)
            if col == "humidity":
                # Relative humidity is bounded; keep both forecast and intervals
                # physically meaningful without replacing the time series by a median.
                mean = np.clip(mean, 0, 100)
                ci = np.clip(ci, 0, 100)
            if self.is_annual:
                dates = pd.date_range(self.weather.index[-1] + pd.offsets.YearBegin(), periods=periods, freq="YS")
            else:
                dates = pd.date_range(self.weather.index[-1] + pd.offsets.MonthBegin(), periods=periods, freq="MS")
            for date, val, lo, hi in zip(dates, mean, ci[:, 0], ci[:, 1]):
                rows.append({"date": date, "variable": col, "forecast": val, "lower_95": lo, "upper_95": hi,
                             "confidence_level": confidence_level, "scenario": climate_scenario})
        return pd.DataFrame(rows)

    def score(self, forecast: Optional[pd.DataFrame] = None,
              manual_humidity: Optional[float] = None) -> Tuple[float, pd.DataFrame]:
        """Return aggregate score and period breakdown; periods follow source frequency."""
        fc = forecast if forecast is not None else self.forecast()
        temps = fc[fc.variable == "temperature"].forecast.to_numpy()
        rain = fc[fc.variable == "rainfall"].forecast.to_numpy()
        historical_t = self.weather.temperature
        historical_r = self.weather.rainfall
        t_rows = fc[fc.variable == "temperature"].sort_values("date")
        r_rows = fc[fc.variable == "rainfall"].sort_values("date")
        t_hi = t_rows.upper_95.to_numpy() if "upper_95" in t_rows else temps
        t_lo = t_rows.lower_95.to_numpy() if "lower_95" in t_rows else temps
        r_hi = r_rows.upper_95.to_numpy() if "upper_95" in r_rows else rain
        r_lo = r_rows.lower_95.to_numpy() if "lower_95" in r_rows else rain
        t_scale = max(float(historical_t.std()), 1.0)
        r_scale = max(float(historical_r.std()), float(historical_r.mean()) * .1, 1.0)
        # Score climate exposures using the selected confidence interval's adverse
        # bound: hot bound for heat, dry bound for drought, wet bound for flood.
        heat = np.clip((t_hi - 30) / 12, 0, 1)
        if self.is_annual:
            # Annual totals cannot be compared with monthly mm thresholds. Score
            # unusually dry/wet years against the selected location's history.
            drought = np.clip((historical_r.mean() - r_lo) / (2 * r_scale), 0, 1)
            flood = np.clip((r_hi - historical_r.mean()) / (2 * r_scale), 0, 1)
        else:
            drought = np.clip((25 - r_lo) / 25, 0, 1)
            flood = np.clip((r_hi - 150) / 120, 0, 1)
        # Use the point forecast for the anomaly component. Using the farthest
        # confidence-bound edge made the same interval width dominate every year
        # and could flatten distinct years to an identical risk score.
        t_anomaly = np.abs((temps-historical_t.mean())/t_scale)
        r_anomaly = np.abs((rain-historical_r.mean())/r_scale)
        anomaly = np.clip((t_anomaly + r_anomaly) / 5, 0, 1)
        h_rows = fc[fc.variable == "humidity"].sort_values("date")
        if len(h_rows):
            humidity = h_rows.forecast.to_numpy()
        else:
            # Explicitly mark missing humidity rather than implying a fixed 60%.
            humidity = np.full(len(temps), np.nan)
        if manual_humidity is not None and len(humidity): humidity[0] = float(manual_humidity)
        humidity_extreme = np.nan_to_num(np.clip(np.maximum(humidity - 80, 20 - humidity) / 20, 0, 1), nan=0.0)
        # Continuous components allow risk scores to respond to forecast changes.
        raw = 100 * (.25*heat + .20*drought + .20*flood + .25*anomaly + .10*humidity_extreme)
        classes = np.where(raw >= 60, "High", np.where(raw >= 30, "Moderate", "Low"))
        monthly = pd.DataFrame({"date": sorted(fc.date.unique()), "temperature": temps, "rainfall": rain, "humidity": humidity, "risk_score": raw, "risk_class": classes})
        return float(monthly.risk_score.mean()), monthly


@dataclass
class Fertilizer:
    name: str
    n: float
    p: float
    k: float
    cost: float  # cost per kg of product
    max_kg_ha: float


class NutrientOptimizer:
    """Associational yield model plus a nutrient-target fertilizer cost proxy."""
    def __init__(self, data: pd.DataFrame, fertilizers: Optional[Sequence[Fertilizer]] = None):
        self.data = data.copy()
        self.fertilizers = list(fertilizers or [
            Fertilizer("Urea", .46, 0, 0, 6.2, 220), Fertilizer("DAP", .18, .46, 0, 27.0, 180),
            Fertilizer("MOP", 0, 0, .60, 17.0, 160), Fertilizer("NPK 15-15-15", .15, .15, .15, 24.0, 250),
        ])
        self.model = make_pipeline(SimpleImputer(), StandardScaler(), MLPRegressor(hidden_layer_sizes=(48, 24), activation="relu", alpha=.02, max_iter=1200, early_stopping=True, random_state=RANDOM_STATE))
        self.nutrient_columns = ["N", "P", "K"]

    def fit(self) -> "NutrientOptimizer":
        # Feed soil nutrients plus weather context if present; keeps same schema at inference.
        cols = self.nutrient_columns + [c for c in ["temperature", "rainfall", "ph"] if c in self.data.columns]
        self.input_columns = cols
        self.model.fit(self.data[cols], self.data.yield_t_ha)
        self.validation_mae = float("nan")
        return self

    def optimize(self, soil: Dict[str, float], max_generations: int = 90, population_size: int = 72,
                 max_total_kg_ha: float = 420, overdose_penalty: float = 35,
                 budget_per_ha: Optional[float] = None, goal: str = "balance_roi",
                 preferred_products: Optional[Sequence[str]] = None,
                 excluded_products: Optional[Sequence[str]] = None,
                 expected_price_per_ton: float = 300.0) -> Dict[str, object]:
        rng = np.random.default_rng(RANDOM_STATE)
        preferred_products = {str(v).strip().lower() for v in (preferred_products or [])}
        excluded_products = {str(v).strip().lower() for v in (excluded_products or [])}
        fert = [f for f in self.fertilizers if f.name.lower() not in excluded_products]
        if not fert: raise ValueError("All fertilizer products were excluded; allow at least one product.")
        caps = np.array([f.max_kg_ha for f in fert])
        matrix = np.array([[f.n, f.p, f.k] for f in fert]) / 100.0
        costs = np.array([f.cost for f in fert])
        base = np.array([soil.get(k, float(self.data[k].median())) for k in self.nutrient_columns])
        # Guardrails are explicit and editable; incremental application caps prevent excessive nutrient additions.
        safe_targets = np.array([min(150, max(45, float(self.data.N.quantile(.65)))),
                                 min(110, max(35, float(self.data.P.quantile(.65)))),
                                 min(150, max(35, float(self.data.K.quantile(.65))))])
        x_base = {c: soil.get(c, float(self.data[c].median())) for c in self.input_columns}
        baseline_sample = dict(x_base)
        baseline_yield = float(max(0, self.model.predict(pd.DataFrame([baseline_sample])[self.input_columns])[0]))
        def evaluate(genes: np.ndarray) -> Tuple[float, float, np.ndarray]:
            applied = genes @ matrix
            final_nutrients = base + applied * 100
            sample = dict(x_base)
            sample.update(dict(zip(self.nutrient_columns, final_nutrients)))
            yield_hat = float(max(0, self.model.predict(pd.DataFrame([sample])[self.input_columns])[0]))
            cost = float(genes @ costs)
            # The yield table is observational and contains no fertilizer
            # treatment labels, so its response estimate is not causal. Select
            # rates by cost and distance from empirical nutrient guardrails;
            # use the yield model for reporting only.
            target_distance = float(np.sum((final_nutrients - safe_targets) ** 2))
            penalty = overdose_penalty * target_distance + max(0, float(genes.sum()) - max_total_kg_ha) * 100
            goal_key = str(goal).strip().lower()
            preferred_bonus = .02 * float(np.sum(genes[[i for i, f in enumerate(fert) if f.name.lower() in preferred_products]])) if preferred_products else 0.0
            budget_excess = max(0.0, cost - budget_per_ha) if budget_per_ha is not None else 0.0
            if goal_key in ("minimize_cost", "minimize cost", "cost"):
                fitness = -cost - penalty
            elif goal_key in ("maximize_yield", "maximize yield", "yield"):
                fitness = -target_distance - .0002 * cost - penalty
            else:
                # ROI cannot be learned from observational crop yields without
                # crop production costs and fertilizer-treatment comparisons.
                fitness = -cost - penalty
            fitness += preferred_bonus
            fitness -= budget_excess * 100
            return fitness, yield_hat, final_nutrients
        pop = rng.uniform(0, caps, size=(population_size, len(fert)))
        best = None
        for _ in range(max_generations):
            scores = np.array([evaluate(g)[0] for g in pop])
            elite_idx = np.argsort(scores)[-max(2, population_size//8):]
            elite = pop[elite_idx]
            if best is None or scores[elite_idx[-1]] > best[0]:
                best = (float(scores[elite_idx[-1]]), elite[-1].copy())
            children = [g.copy() for g in elite]
            while len(children) < population_size:
                a, b = elite[rng.integers(len(elite), size=2)]
                child = np.where(rng.random(len(fert)) < .5, a, b)
                if rng.random() < .85:
                    child += rng.normal(0, caps*.10)
                children.append(np.clip(child, 0, caps))
            pop = np.asarray(children)
        genes = best[1]
        # Report and recompute from the displayed application rates. This keeps
        # product quantities, individual N/P/K contributions, totals, costs and
        # final soil values arithmetically consistent after rounding.
        reported_genes = np.array([round(float(q), 2) if q > .1 else 0.0 for q in genes])
        product_contributions = np.round(reported_genes[:, None] * matrix * 100, 2)
        applied = product_contributions.sum(axis=0)
        final = base + applied
        final_sample = dict(x_base)
        final_sample.update(dict(zip(self.nutrient_columns, final)))
        predicted_yield = float(max(0, self.model.predict(pd.DataFrame([final_sample])[self.input_columns])[0]))
        reported_cost = float(reported_genes @ costs)
        contribution_breakdown = {
            f.name: {k: float(v) for k, v in zip(self.nutrient_columns, row)}
            for f, q, row in zip(fert, reported_genes, product_contributions) if q > 0
        }
        goal_key = str(goal).strip().lower()
        if goal_key == "balance_roi":
            optimization_note = "ROI cannot be estimated without production costs and fertilizer-treatment response labels; rates use a nutrient-target cost proxy."
        elif goal_key in ("maximize_yield", "maximize yield", "yield"):
            optimization_note = "Yield maximization cannot be inferred causally from these observational rows; rates use nutrient-target matching, not a yield-response objective."
        else:
            optimization_note = "Rates minimize the supplied product costs while matching empirical nutrient targets; targets are not validated dosage prescriptions."
        return {"fertilizer_kg_ha": {f.name: float(q) for f, q in zip(fert, reported_genes) if q > 0},
                "nutrient_contributions_by_product_kg_ha": contribution_breakdown,
                "total_product_kg_ha": round(float(reported_genes.sum()), 2), "estimated_cost_per_ha": round(reported_cost, 2),
                "predicted_yield_t_ha": round(predicted_yield, 3),
                "final_soil_NPK": {k: round(float(v), 2) for k, v in zip(self.nutrient_columns, final)},
                "starting_soil_NPK": {k: round(float(v), 2) for k, v in zip(self.nutrient_columns, base)},
                "added_soil_NPK": {k: round(float(v), 2) for k, v in zip(self.nutrient_columns, applied)},
                "guardrail_targets_NPK": {k: round(float(v), 2) for k, v in zip(self.nutrient_columns, safe_targets)},
                "optimization_goal": goal, "budget_per_ha": budget_per_ha,
                "within_budget": budget_per_ha is None or reported_cost <= budget_per_ha,
                "baseline_predicted_yield_t_ha": round(baseline_yield, 3),
                "guardrail_target_basis_note": "Targets use observed soil-data quantiles; the source does not contain validated crop-specific fertilizer-rate labels.",
                "optimization_note": optimization_note,
                "warning": "Estimated rates are not verified agronomic prescriptions. Confirm soil-test units, local crop guidance, and product labels before application."}


def collect_user_inputs() -> Dict[str, object]:
    """Guided Kaggle input form. Press Enter to accept each displayed default."""
    def ask(label: str, default: str = "") -> str:
        value = input(f"{label}" + (f" [{default}]" if default != "" else " [optional]") + ": ").strip()
        return value or default
    def number(label: str, default: Optional[float] = None) -> Optional[float]:
        raw = ask(label, "" if default is None else str(default))
        if not raw: return None
        try: return float(raw)
        except ValueError:
            print(f"  Please enter a number. Using {default}.")
            return default
    print("\n" + "="*72 + "\nAGRIVISION FARM INPUTS  |  blank fields use dataset defaults\n" + "="*72)
    state = ask("State")
    district = ask("District")
    season = ask("Season (Kharif/Rabi/Summer)", "Kharif")
    land_size = number("Land size", 1.0) or 1.0
    land_unit = ask("Land unit (acres/hectares)", "acres").lower()
    budget_crop = number("Maximum fertilizer/pesticide spend per acre (INR)")
    risk = ask("Risk tolerance (low/medium/high)", "medium").lower()
    include = ask("Crops to consider (comma-separated)")
    exclude = ask("Crops to exclude (comma-separated)")
    override = number("Mandi price override (INR per tonne; blank uses price dataset)")
    print("\nWEATHER & CLIMATE RISK")
    target_year = ask("Target forecast year (blank uses 12-month horizon)")
    horizon = number("Forecast horizon in months", 12) or 12
    scenario = ask("Climate scenario (normal/drought-prone/flood-prone)", "normal").lower()
    confidence = number("Forecast confidence level (90 or 95)", 95) or 95
    manual_temp = number("Current temperature °C")
    manual_rain = number("Current rainfall mm")
    manual_humidity = number("Current relative humidity %")
    print("\nFERTILIZER & NUTRIENT OPTIMIZER")
    crop_choice = ask("Crop choice (blank uses top crop recommendation)")
    n = number("Soil test Nitrogen N (kg/ha)")
    p = number("Soil test Phosphorus P (kg/ha)")
    k = number("Soil test Potassium K (kg/ha)")
    ph = number("Soil pH")
    fert_budget = number("Maximum fertilizer spend per acre (INR)", budget_crop)
    goal = ask("Optimization goal (minimize_cost/maximize_yield/balance_roi)", "balance_roi").lower()
    print("Supported optimizer products: Urea, DAP, MOP, NPK 15-15-15")
    prefer = ask("Preferred fertilizer products (comma-separated)")
    avoid = ask("Fertilizer products to avoid (comma-separated)")
    def split(s: str) -> list:
        return [x.strip() for x in s.split(",") if x.strip()]
    return {"location": {"state": state, "district": district}, "season": season,
            "land_size": land_size, "land_unit": land_unit, "crop_budget_per_acre": budget_crop,
            "risk_tolerance": risk, "include_crops": split(include), "exclude_crops": split(exclude),
            "price_override": override, "target_year": target_year, "horizon_months": int(horizon),
            "climate_scenario": scenario, "confidence_level": float(confidence)/100.0,
            "manual_weather": {k: v for k, v in {"temperature": manual_temp, "rainfall": manual_rain, "humidity": manual_humidity}.items() if v is not None},
            "fertilizer_crop": crop_choice, "soil_test": {k: v for k, v in {"N": n, "P": p, "K": k, "ph": ph}.items() if v is not None},
            "fertilizer_budget_per_acre": fert_budget, "optimization_goal": goal,
            "preferred_fertilizers": split(prefer), "excluded_fertilizers": split(avoid)}


def _hectares(size: float, unit: str) -> float:
    return float(size) * (0.4046856422 if str(unit).lower().startswith("acre") else 1.0)


def _normalize_user_inputs(raw: Optional[Dict[str, object]]) -> Dict[str, object]:
    """Fill omitted fields and coerce common Kaggle edit mistakes safely."""
    import copy
    import math
    defaults = {
        "location": {"state": None, "district": None}, "season": None,
        "land_size": 1.0, "land_unit": "acres", "crop_budget_per_acre": None,
        "risk_tolerance": "medium", "include_crops": [], "exclude_crops": [],
        "price_override": None, "crop_cost_per_ha": {}, "target_year": 2027, "horizon_months": 12,
        "horizon_years": 1, "climate_scenario": "normal", "confidence_level": 0.90,
        "manual_weather": {"temperature": None, "rainfall": None, "humidity": None},
        "fertilizer_crop": None, "soil_test": {"N": None, "P": None, "K": None, "ph": None},
        "fertilizer_budget_per_acre": None, "optimization_goal": "balance_roi",
        "preferred_fertilizers": [], "excluded_fertilizers": [],
        "fertilizer_prices_per_kg": {"Urea": 6.2, "DAP": 27.0, "MOP": 17.0, "NPK 15-15-15": 24.0},
    }
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("user_inputs must be a dictionary, for example USER_INPUTS.")
    merged = {**defaults, **raw}
    for key in ("location", "manual_weather", "soil_test", "fertilizer_prices_per_kg"):
        if not isinstance(merged.get(key), dict):
            merged[key] = {}
        merged[key] = {**defaults[key], **merged[key]}
    if not isinstance(merged.get("crop_cost_per_ha"), dict):
        merged["crop_cost_per_ha"] = {}
    try:
        merged["crop_cost_per_ha"] = {str(k): float(v) for k, v in merged["crop_cost_per_ha"].items() if v is not None}
    except (TypeError, ValueError):
        merged["crop_cost_per_ha"] = {}
    if any(not math.isfinite(v) or v < 0 for v in merged["crop_cost_per_ha"].values()):
        raise ValueError("crop_cost_per_ha values must be finite and nonnegative INR/ha.")
    def as_list(value: object) -> list:
        if value is None: return []
        if isinstance(value, str): return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, (list, tuple, set)): return [str(part).strip() for part in value if str(part).strip()]
        return [str(value).strip()]
    for key in ("include_crops", "exclude_crops", "preferred_fertilizers", "excluded_fertilizers"):
        merged[key] = as_list(merged.get(key))
    for key in ("land_size", "crop_budget_per_acre", "fertilizer_budget_per_acre"):
        value = merged.get(key)
        if value is not None:
            try:
                merged[key] = float(value)
                if not math.isfinite(merged[key]): raise ValueError
            except (TypeError, ValueError): merged[key] = defaults[key]
    if merged["land_size"] <= 0:
        raise ValueError("land_size must be greater than zero.")
    for key in ("crop_budget_per_acre", "fertilizer_budget_per_acre"):
        if merged[key] is not None and merged[key] < 0:
            raise ValueError(f"{key} cannot be negative.")
    for key in ("horizon_months", "horizon_years"):
        try: merged[key] = max(1, int(merged.get(key) or defaults[key]))
        except (TypeError, ValueError): merged[key] = defaults[key]
    try:
        confidence = float(merged.get("confidence_level", defaults["confidence_level"]))
        merged["confidence_level"] = confidence / 100 if confidence > 1 else confidence
        if not 0 < merged["confidence_level"] < 1: raise ValueError
    except (TypeError, ValueError): merged["confidence_level"] = defaults["confidence_level"]
    for key in ("temperature", "rainfall", "humidity"):
        value = merged["manual_weather"].get(key)
        if value is not None:
            try:
                merged["manual_weather"][key] = float(value)
                if not math.isfinite(merged["manual_weather"][key]): raise ValueError
            except (TypeError, ValueError):
                raise ValueError(f"manual_weather.{key} must be a finite number.")
    for key, lower, upper in (("temperature", -10, 60), ("rainfall", 0, 10000), ("humidity", 0, 100)):
        value = merged["manual_weather"].get(key)
        if value is not None and not lower <= value <= upper:
            raise ValueError(f"manual_weather.{key} must be between {lower} and {upper}.")
    for key in ("N", "P", "K", "ph"):
        value = merged["soil_test"].get(key)
        if value is not None:
            try:
                merged["soil_test"][key] = float(value)
                if not math.isfinite(merged["soil_test"][key]): raise ValueError
            except (TypeError, ValueError):
                raise ValueError(f"soil_test.{key} must be a finite number.")
    for key in ("N", "P", "K"):
        value = merged["soil_test"].get(key)
        if value is not None and not 0 <= value <= 1000:
            raise ValueError(f"soil_test.{key} must be between 0 and 1000 kg/ha.")
    ph = merged["soil_test"].get("ph")
    if ph is not None and not 3 <= ph <= 10:
        raise ValueError("soil_test.ph must be between 3 and 10; check the soil-test value and units.")
    try:
        merged["fertilizer_prices_per_kg"] = {k: float(v) for k, v in merged["fertilizer_prices_per_kg"].items() if v is not None}
    except (TypeError, ValueError): merged["fertilizer_prices_per_kg"] = defaults["fertilizer_prices_per_kg"]
    if any(not math.isfinite(v) or v <= 0 for v in merged["fertilizer_prices_per_kg"].values()):
        raise ValueError("fertilizer_prices_per_kg values must be finite and greater than zero.")
    if not isinstance(merged.get("price_override"), (dict, int, float, type(None))):
        try: merged["price_override"] = float(merged["price_override"])
        except (TypeError, ValueError): merged["price_override"] = None
    override_values = (merged["price_override"].values() if isinstance(merged["price_override"], dict)
                      else [merged["price_override"]] if merged["price_override"] is not None else [])
    if any(value is not None and (not math.isfinite(float(value)) or float(value) <= 0) for value in override_values):
        raise ValueError("price_override values must be finite and greater than zero INR/tonne.")
    if not isinstance(merged["location"], dict):
        merged["location"] = {"state": None, "district": None}
    for key in ("state", "district"):
        val = merged["location"].get(key)
        merged["location"][key] = str(val).strip() if val is not None and str(val).strip() else None
    if merged.get("season") is not None:
        merged["season"] = str(merged["season"]).strip() or None
    for key, allowed in (("risk_tolerance", {"low", "medium", "high"}),
                         ("climate_scenario", {"normal", "drought-prone", "flood-prone"}),
                         ("optimization_goal", {"minimize_cost", "maximize_yield", "balance_roi"})):
        value = str(merged.get(key, defaults[key])).strip().lower()
        merged[key] = value if value in allowed else defaults[key]
    merged["land_unit"] = "hectares" if str(merged.get("land_unit", "acres")).lower().startswith("hect") else "acres"
    return merged


def _input_support_diagnostics(training_data: pd.DataFrame, inputs: Dict[str, object]) -> Dict[str, object]:
    """Flag user-entered model features outside the central training range."""
    supplied = {}
    for key, value in (inputs.get("soil_test") or {}).items():
        if value is not None and key in {"N", "P", "K", "ph"}:
            supplied[key] = float(value)
    for key, value in (inputs.get("manual_weather") or {}).items():
        if value is not None and key in {"temperature", "rainfall", "humidity"}:
            supplied[key] = float(value)
    warnings_out = []
    for key, value in supplied.items():
        if key not in training_data:
            continue
        observed = pd.to_numeric(training_data[key], errors="coerce").dropna()
        if observed.empty:
            continue
        lower, upper = float(observed.quantile(.01)), float(observed.quantile(.99))
        if value < lower or value > upper:
            warnings_out.append({"field": key, "value": value, "training_q01": lower,
                                 "training_q99": upper,
                                 "message": "Input is outside the central 98% training range; the model is extrapolating."})
    return {"within_training_support": not warnings_out, "warnings": warnings_out,
            "note": "Training-range checks are diagnostic only; they do not replace a lab soil test or local agronomic review."}


def _run_pipeline_impl(data: Optional[pd.DataFrame] = None, weather: Optional[pd.DataFrame] = None,
                      user_inputs: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Run the three engines with a website-ready structured farmer input object.

    Call ``collect_user_inputs()`` for Kaggle prompts, or pass the same dictionary
    shape directly from a future API/frontend.
    """
    inputs = _normalize_user_inputs(USER_INPUTS if user_inputs is None else user_inputs)
    location = inputs.get("location", {}) or {}
    weather_state_matched = False
    weather_district_matched = False
    if data is None:
        path = _find_csv("crop_yield.csv")
        if path:
            data = pd.read_csv(path); print(f"Loaded crop data: {path}")
        else:
            data, demo_weather = make_demo_data()
            weather = weather if weather is not None else demo_weather
            print("No crop_yield.csv found; using reproducible synthetic demo data. Add your dataset to Kaggle Input for real-data results.")
    if weather is None:
        path = _find_csv("weather.csv") or _find_csv("state_weather_data_1997_2020.csv")
        if path:
            try:
                raw_weather = _canonicalize_columns(pd.read_csv(path))
                for field, requested in [("state", location.get("state")), ("district", location.get("district"))]:
                    if requested and field in raw_weather:
                        match = raw_weather[field].map(_location_key).eq(_location_key(requested))
                        if match.any():
                            raw_weather = raw_weather.loc[match].copy()
                            if field == "state": weather_state_matched = True
                            if field == "district": weather_district_matched = True
                        elif field == "state":
                            raise ValueError(f"State '{requested}' is not present in the weather source. Choose a state listed in the dataset.")
                        else:
                            print(f"No exact {field} match for '{requested}' in weather history; retaining all available rows.")
                weather = _prepare_weather(raw_weather); print(f"Loaded and standardized weather data: {path}")
            except ValueError as exc:
                if "is not present in the weather source" in str(exc):
                    raise
                print(f"Weather file could not be mapped ({exc}); trying crop-yield weather columns.")
        if weather is None:
            try:
                weather = _prepare_weather(data)
                print("Built weather history from real crop-yield weather columns.")
            except Exception as exc:
                raise FileNotFoundError(f"Could not find/derive real weather history under {DATA_DIR}: {exc}")
    if weather is not None and ("state" in _canonicalize_columns(weather) or "district" in _canonicalize_columns(weather)):
        weather_raw = _canonicalize_columns(weather)
        for field, requested in [("state", location.get("state")), ("district", location.get("district"))]:
            if requested and field in weather_raw:
                mask = weather_raw[field].map(_location_key).eq(_location_key(requested))
                if mask.any():
                    weather_raw = weather_raw.loc[mask].copy()
                    if field == "state": weather_state_matched = True
                    if field == "district": weather_district_matched = True
                elif field == "state":
                    raise ValueError(f"State '{requested}' is not present in the weather source.")
        weather = _prepare_weather(weather_raw)
    elif not {"date", "temperature", "rainfall"}.issubset(weather.columns):
        weather = _prepare_weather(weather)

    # Match local yield history when the requested geographic/season columns exist.
    all_source = _canonicalize_columns(data)
    source = all_source.copy()
    location_match = {"state": False, "district": False, "season": False,
                      "district_yield_history": False,
                      "weather_state": weather_state_matched,
                      "weather_district": weather_district_matched}
    for field, requested in [("state", location.get("state")), ("district", location.get("district")), ("season", inputs.get("season"))]:
        if requested and field in source:
            if field == "season":
                mask = source[field].astype(str).str.strip().str.casefold().eq(str(requested).strip().casefold())
            else:
                mask = source[field].map(_location_key).eq(_location_key(requested))
            if mask.any():
                source = source.loc[mask].copy(); location_match[field] = True
            elif field == "season":
                raise ValueError(f"Season '{requested}' is not present in crop history. Choose a season listed in the dataset.")
            else:
                print(f"No exact {field} match for '{requested}' in crop history; retaining all available rows.")
    # Fit against all regions so N/P/K/pH vary during training. Keep a separate
    # location/season slice for candidate crops and local historical baselines.
    regional_data = _prepare_training_data(source, weather)
    district_history = _district_yield_history(location, all_source.crop.astype(str).unique()) if "crop" in all_source else pd.DataFrame()
    if not district_history.empty:
        if inputs.get("season") and "season" in source and "crop" in source:
            eligible = {_crop_key(c) for c in source.crop.dropna().unique()}
            district_history = district_history[district_history.crop.map(_crop_key).isin(eligible)]
        if not district_history.empty:
            regional_data = _prepare_training_data(district_history, weather)
            location_match["district"] = True
            location_match["district_yield_history"] = True
    data = _prepare_training_data(all_source, weather)
    missing = sorted(set(FEATURES + ["crop", "yield_t_ha"]) - set(data.columns))
    if missing:
        raise ValueError(f"Model table is missing {missing}. Available crop columns: {list(data.columns)}. Ensure Kaggle Add Input includes crop_yield.csv and fertilizer/soil CSVs.")
    price_path = _find_csv("Price_Agriculture_commodities_Week.csv") or _find_csv_containing("price", "commodities")
    if price_path:
        market_all = _canonicalize_columns(pd.read_csv(price_path))
        if "price_per_ton" not in data:
            data["price_per_ton"] = np.nan
        if "price_source" not in data:
            data["price_source"] = np.where(data.price_per_ton.notna(), "crop_yield_csv", "unavailable")
        if "price_observation_date" not in data:
            data["price_observation_date"] = pd.NaT
        if "crop" in market_all and "price_per_ton" in market_all:
            market_all = market_all.copy()
            market_all["_crop_key"] = market_all.crop.map(_crop_key)
            market_all["_price_inr_per_tonne"] = market_all.price_per_ton * 10.0  # source prices are INR/quintal
            if "date" in market_all:
                market_all["_market_date"] = pd.to_datetime(market_all.date, dayfirst=True, errors="coerce")
            else:
                market_all["_market_date"] = pd.NaT
            # The source has isolated extreme modal-price entries (for example,
            # one Wheat quote 10x the commodity's usual range). Exclude values
            # outside a broad 5x band around the historical crop median before
            # taking latest-date market medians.
            historical_medians = market_all.groupby("_crop_key")._price_inr_per_tonne.transform("median")
            plausible_price = (market_all._price_inr_per_tonne > 0) & (market_all._price_inr_per_tonne.between(historical_medians / 5, historical_medians * 5))
            market_all = market_all.loc[plausible_price].copy()
            def latest_crop_prices(frame: pd.DataFrame) -> pd.DataFrame:
                valid = frame.dropna(subset=["_crop_key", "_price_inr_per_tonne"]).copy()
                if valid["_market_date"].notna().any():
                    latest_dates = valid.groupby("_crop_key")._market_date.transform("max")
                    dated = valid[valid._market_date.eq(latest_dates)]
                    return dated.groupby("_crop_key", as_index=True).agg(
                        price_per_ton=("_price_inr_per_tonne", "median"),
                        price_observation_date=("_market_date", "max"))
                return valid.groupby("_crop_key", as_index=True).agg(
                    price_per_ton=("_price_inr_per_tonne", "median"))
            market_state = market_all
            state_price_match = False
            if location.get("state") and "state" in market_all:
                state_mask = market_all.state.map(_location_key).eq(_location_key(location["state"]))
                if state_mask.any():
                    market_state = market_all.loc[state_mask].copy()
                    state_price_match = True
                else:
                    print(f"No exact state match for '{location['state']}' in market history; checking national crop prices.")
            market_region = market_state
            district_price_match = False
            if location.get("district") and "district" in market_state and (state_price_match or not location.get("state")):
                district_mask = market_state.district.map(_location_key).eq(_location_key(location["district"]))
                if district_mask.any():
                    market_region = market_state.loc[district_mask].copy()
                    district_price_match = True
                else:
                    print(f"No exact district match for '{location['district']}' in market history; using state-level prices where available.")
            regional_prices = latest_crop_prices(market_region)
            state_prices = latest_crop_prices(market_state)
            national_prices = latest_crop_prices(market_all)
            crop_keys = data.crop.map(_crop_key)
            regional_values = crop_keys.map(regional_prices.price_per_ton)
            state_values = crop_keys.map(state_prices.price_per_ton)
            national_values = crop_keys.map(national_prices.price_per_ton)
            regional_dates = crop_keys.map(regional_prices.price_observation_date) if "price_observation_date" in regional_prices else pd.Series(pd.NaT, index=data.index)
            state_dates = crop_keys.map(state_prices.price_observation_date) if "price_observation_date" in state_prices else pd.Series(pd.NaT, index=data.index)
            national_dates = crop_keys.map(national_prices.price_observation_date) if "price_observation_date" in national_prices else pd.Series(pd.NaT, index=data.index)
            missing_price = data.price_per_ton.isna()
            use_regional = missing_price & regional_values.notna()
            data.loc[use_regional, "price_per_ton"] = regional_values[use_regional]
            data.loc[use_regional, "price_source"] = "selected_district_latest_market" if district_price_match else "selected_state_latest_market" if state_price_match else "latest_market"
            data.loc[use_regional, "price_observation_date"] = regional_dates[use_regional]
            missing_price = data.price_per_ton.isna()
            use_state = missing_price & state_values.notna()
            data.loc[use_state, "price_per_ton"] = state_values[use_state]
            data.loc[use_state, "price_source"] = "selected_state_latest_market"
            data.loc[use_state, "price_observation_date"] = state_dates[use_state]
            missing_price = data.price_per_ton.isna()
            use_national = missing_price & national_values.notna()
            data.loc[use_national, "price_per_ton"] = national_values[use_national]
            data.loc[use_national, "price_source"] = "national_latest_market"
            data.loc[use_national, "price_observation_date"] = national_dates[use_national]
            print(f"Joined the latest crop-matched mandi prices from: {price_path}; unmatched crops remain unpriced instead of receiving a global median.")

    crop_costs = {_crop_key(k): float(v) for k, v in (inputs.get("crop_cost_per_ha", {}) or {}).items()}
    if "cost_per_ha" not in data:
        data["cost_per_ha"] = np.nan
    data["cost_source"] = np.where(data.cost_per_ha.notna(), "crop_yield_csv", "unavailable")
    cost_keys = data.crop.map(_crop_key)
    cost_values = cost_keys.map(crop_costs)
    use_input_cost = data.cost_per_ha.isna() & cost_values.notna()
    data.loc[use_input_cost, "cost_per_ha"] = cost_values[use_input_cost]
    data.loc[use_input_cost, "cost_source"] = "user_input"

    # The location slice keeps its crop history and yield levels, while market
    # price/cost data are mapped by crop from the fully enriched training frame.
    financial_columns = [c for c in ["price_per_ton", "price_source", "price_observation_date", "cost_per_ha", "cost_source"] if c in data]
    if not regional_data.empty and financial_columns:
        aggregations = {col: ("median" if col in ("price_per_ton", "cost_per_ha") else "max" if col == "price_observation_date" else "first")
                        for col in financial_columns}
        financial = data.assign(_crop_key=data.crop.map(_crop_key)).groupby("_crop_key", as_index=True)[financial_columns].agg(aggregations)
        local_keys = regional_data.crop.map(_crop_key)
        for col in financial_columns:
            if col not in regional_data:
                regional_data[col] = np.nan
            mapped = local_keys.map(financial[col])
            regional_data[col] = regional_data[col].where(regional_data[col].notna(), mapped)

    crop = CropProfitEngine(data).fit()
    soil_source = regional_data if not regional_data.empty else data
    soil = {c: float(soil_source[c].median()) for c in FEATURES}
    soil.update({k: float(v) for k, v in (inputs.get("soil_test", {}) or {}).items() if k in FEATURES and v is not None})
    manual = {k: v for k, v in (inputs.get("manual_weather", {}) or {}).items() if v is not None}
    soil.update({k: float(v) for k, v in manual.items() if k in FEATURES and v is not None})
    land_unit = str(inputs.get("land_unit", "acres"))
    land_size = float(inputs.get("land_size", 1.0) or 1.0)
    area_ha = _hectares(land_size, land_unit)
    budget_crop = inputs.get("crop_budget_per_acre")
    crop_budget_ha = float(budget_crop) / .4046856422 if budget_crop is not None else None

    weather_engine = WeatherClimateRiskEngine(weather).fit()
    if weather_engine.is_annual:
        print("Weather source resolution: annual. This dataset cannot support honest monthly climate-risk scores; reports will use annual periods.")
    target_year = str(inputs.get("target_year", "")).strip()
    if weather_engine.is_annual:
        periods = int(inputs.get("horizon_years", 1) or 1)
        if target_year:
            try:
                year = int(target_year[:4]); last = weather_engine.weather.index[-1]
                periods = max(1, min(30, year-last.year))
            except Exception: pass
    else:
        periods = int(inputs.get("horizon_months", 12) or 12)
        if target_year:
            try:
                year = int(target_year[:4]); last = weather_engine.weather.index[-1]
                periods = max(1, min(120, (year-last.year)*12 + (12-last.month+1)))
            except Exception: pass
    confidence = float(inputs.get("confidence_level", .95) or .95)
    scenario = str(inputs.get("climate_scenario", "normal"))
    forecast = weather_engine.forecast(periods, confidence_level=confidence, climate_scenario=scenario)
    # Manual readings describe current conditions. Apply them to the first forecast
    # period only for a true monthly source; annual records cannot support monthly overlays.
    if not weather_engine.is_annual:
        for variable in ["temperature", "rainfall"]:
            value = manual.get(variable)
            if value is not None:
                idx = forecast[forecast.variable == variable].index
                if len(idx): forecast.loc[idx[0], "forecast"] = float(value)
    risk_score, risk_monthly = weather_engine.score(forecast, manual_humidity=manual.get("humidity"))
    risk_level = "High" if risk_score >= 60 else "Moderate" if risk_score >= 30 else "Low"
    current_risk_score = None
    if "temperature" in manual and "rainfall" in manual:
        # Manual readings form a separate current-conditions assessment; they do
        # not overwrite the annual/multi-month forecast series.
        current_t = float(manual.get("temperature", weather_engine.weather.temperature.iloc[-1]))
        current_r = float(manual.get("rainfall", weather_engine.weather.rainfall.iloc[-1]))
        as_of = pd.Timestamp.now().normalize()
        current_fc = pd.DataFrame([
            {"date": as_of, "variable": var, "forecast": val, "lower_95": val, "upper_95": val}
            for var, val in [("temperature", current_t), ("rainfall", current_r)]
        ])
        current_risk_score, _ = weather_engine.score(current_fc, manual_humidity=manual.get("humidity"))
    # Feed the same scenario-adjusted forecast context to the crop and fertilizer
    # engines. Before this, climate_scenario changed the risk section only, while
    # crop and nutrient estimates kept using historical median weather.
    for variable in ("temperature", "rainfall", "humidity"):
        if variable not in manual and variable in risk_monthly and risk_monthly[variable].notna().any():
            soil[variable] = float(risk_monthly[variable].mean())
    recommendations = crop.recommend(soil, include_crops=inputs.get("include_crops") or None,
        exclude_crops=inputs.get("exclude_crops") or None, price_override=inputs.get("price_override"),
        budget_per_ha=crop_budget_ha, risk_tolerance=str(inputs.get("risk_tolerance", "medium")),
        reference_data=regional_data)
    best = recommendations.iloc[0]
    simulation = (crop.simulate_profit(float(best.estimated_yield_t_ha), float(best.price_per_ton), float(best.estimated_cost_per_ha))
                  if pd.notna(best.price_per_ton) and pd.notna(best.estimated_cost_per_ha) else None)
    risk_advice = []
    if risk_level == "High":
        risk_advice.append("High climate risk: consider staged sowing, contingency seed, and a locally approved crop insurance review.")
    elif risk_level == "Moderate":
        risk_advice.append("Moderate climate risk: monitor forecast updates and keep irrigation/drainage options ready.")
    else:
        risk_advice.append("Low climate risk in this modeled horizon; continue routine weather monitoring.")
    if weather_engine.is_annual:
        annual_rain_q10, annual_rain_q90 = weather_engine.weather.rainfall.quantile([.10, .90]).tolist()
        if risk_monthly.rainfall.min() < annual_rain_q10:
            risk_advice.append("Forecast annual rainfall is below the location's recent 10th percentile; verify irrigation and water conservation plans.")
        if risk_monthly.rainfall.max() > annual_rain_q90:
            risk_advice.append("Forecast annual rainfall is above the location's recent 90th percentile; check drainage and waterlogging risk.")
    else:
        if risk_monthly.rainfall.min() < 25:
            risk_advice.append("Dry months appear in the forecast: plan water conservation and verify irrigation availability before sowing.")
        if risk_monthly.rainfall.max() > 150:
            risk_advice.append("Very wet months appear in the forecast: check field drainage and avoid waterlogging-sensitive crops where practical.")
    if risk_monthly.temperature.max() > 35:
        risk_advice.append("Hot months appear in the forecast: check local heat-tolerant varieties and crop-stage-specific advisories.")

    # Produce a separate nutrient/fertilizer plan for each of the top three
    # recommended crops. A manually chosen fertilizer_crop is included first
    # only when it is not already among those recommendations.
    top_crops = [str(name) for name in recommendations.crop.head(3).tolist()]
    requested_fertilizer_crop = inputs.get("fertilizer_crop")
    if requested_fertilizer_crop:
        requested_name = str(requested_fertilizer_crop)
        if requested_name.casefold() not in {name.casefold() for name in top_crops}:
            top_crops.insert(0, requested_name)
    top_crops = top_crops[:3]
    local_fertilizer_prices = inputs.get("fertilizer_prices_per_kg", {}) or {}
    fertilizer_soil = dict(soil)
    fert_budget_acre = inputs.get("fertilizer_budget_per_acre")
    fert_budget_ha = float(fert_budget_acre) / .4046856422 if fert_budget_acre is not None else None
    requested_goal = str(inputs.get("optimization_goal", "balance_roi"))
    fertilizer_plans = []
    nutrient_engines = []
    for crop_name in top_crops:
        crop_rows = data[data.crop.astype(str).str.casefold() == crop_name.casefold()]
        nutrient_training = crop_rows if len(crop_rows) >= 25 else data
        nutrient_engine = NutrientOptimizer(nutrient_training).fit()
        nutrient_engines.append(nutrient_engine)
        for product in nutrient_engine.fertilizers:
            if product.name in local_fertilizer_prices:
                product.cost = float(local_fertilizer_prices[product.name])
        if len(crop_rows) < 25:
            print(f"Only {len(crop_rows)} observed rows for {crop_name}; nutrient response uses all crops and is less crop-specific.")
        crop_price = (float(crop_rows.price_per_ton.median()) if "price_per_ton" in crop_rows and crop_rows.price_per_ton.notna().any()
                      else None)
        effective_goal = requested_goal
        plan = nutrient_engine.optimize(fertilizer_soil, budget_per_ha=fert_budget_ha,
            goal=effective_goal,
            preferred_products=inputs.get("preferred_fertilizers") or None,
            excluded_products=inputs.get("excluded_fertilizers") or None,
            expected_price_per_ton=crop_price if crop_price is not None else 0.0)
        plan["requested_optimization_goal"] = requested_goal
        plan["crop_price_used_per_ton"] = None
        plan["crop_price_reference_per_ton"] = crop_price
        plan["yield_response_note"] = "Yield change is an associational model estimate; the source has no fertilizer-treatment labels, so it is not used to choose application rates."
        plan["crop"] = crop_name
        plan["estimated_yield_gain_t_ha"] = round(float(plan["predicted_yield_t_ha"] - plan["baseline_predicted_yield_t_ha"]), 3)
        plan["whole_farm_cost"] = float(plan["estimated_cost_per_ha"] * area_ha)
        plan["whole_farm_fertilizer_kg"] = {k: float(v)*area_ha for k, v in plan["fertilizer_kg_ha"].items()}
        fertilizer_plans.append(plan)
    selected_crop = fertilizer_plans[0]["crop"] if fertilizer_plans else str(best.crop)
    fertilizer = fertilizer_plans[0] if fertilizer_plans else {}
    nutrient = nutrient_engines[0] if nutrient_engines else NutrientOptimizer(data).fit()

    results = {"crop_engine": crop, "recommendations": recommendations, "profit_simulation": simulation,
        "farm_profit_expected": float(best.expected_profit_per_ha * area_ha) if pd.notna(best.expected_profit_per_ha) else None,
        "farm_profit_p10": float(best.profit_p10 * area_ha) if pd.notna(best.profit_p10) else None,
        "farm_profit_p90": float(best.profit_p90 * area_ha) if pd.notna(best.profit_p90) else None,
        "farm_revenue_expected": float(best.expected_gross_revenue_per_ha * area_ha) if pd.notna(best.expected_gross_revenue_per_ha) else None,
        "land_area_ha": area_ha, "weather_engine": weather_engine, "weather_forecast": forecast,
        "risk_score": risk_score, "risk_level": risk_level, "risk_monthly": risk_monthly,
        "risk_advice": risk_advice, "current_manual_weather_risk_score": current_risk_score,
        "nutrient_optimizer": nutrient, "fertilizer_plan": fertilizer,
        "fertilizer_plans": fertilizer_plans,
        "fertilizer_crop": selected_crop, "fertilizer_total_cost": fertilizer["whole_farm_cost"],
        "location_match": location_match, "training_rows": len(data),
        "input_diagnostics": _input_support_diagnostics(data, inputs)}
    return results


def _json_safe(value: object) -> object:
    """Convert pandas/NumPy values and non-finite numbers to strict JSON values."""
    import math
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.DataFrame):
        return [_json_safe(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in list(value)]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def run_pipeline(data: Optional[pd.DataFrame] = None, weather: Optional[pd.DataFrame] = None,
                 user_inputs: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Run all engines, save one stable JSON report, and print only that JSON."""
    import io
    import json
    from contextlib import redirect_stdout

    # Suppress progress messages so stdout contains only the website-ready report.
    with redirect_stdout(io.StringIO()):
        raw = _run_pipeline_impl(data=data, weather=weather, user_inputs=user_inputs)
    report = {
        "schema_version": "agrivision_outputs_v1",
        "crop_profit_ranking": {
            "recommendations": raw["recommendations"],
            "profit_simulation": raw["profit_simulation"],
            "farm_profit_expected_inr": raw["farm_profit_expected"],
            "farm_profit_p10_inr": raw["farm_profit_p10"],
            "farm_profit_p90_inr": raw["farm_profit_p90"],
            "farm_revenue_expected_inr": raw["farm_revenue_expected"],
            "land_area_ha": raw["land_area_ha"],
            "training_rows": raw["training_rows"],
            "classifier_training_rows": raw["crop_engine"].classifier_training_rows,
            "classifier_training_source": raw["crop_engine"].classifier_training_source,
            "crop_suitability_model_status": "not_estimated_no_suitability_labels_in_source_data",
            "yield_rows_excluded_as_invalid_units": raw["crop_engine"].invalid_yield_rows,
            "yield_model": raw["crop_engine"].yield_model_name,
            "yield_model_selection": "Per-crop histogram gradient boosting is used only when chronological holdout MAE improves on the crop-median baseline by at least 5%; otherwise the median is used.",
            "model_metrics": raw["crop_engine"].metrics,
            "per_crop_validation_metrics": raw["crop_engine"].per_crop_yield_metrics,
            "features_used": raw["crop_engine"].features,
            "ranking_basis": (raw["recommendations"].iloc[0].get("ranking_basis") if len(raw["recommendations"]) else None),
            "location_data_note": ("District/year crop-yield history from ICRISAT is used when that district is present."
                                   if raw["location_match"].get("district_yield_history") else
                                   "District crop-yield history is unavailable for this selection; local/state crop records are used where present."),
            "financial_data_note": "The source yield dataset has no production cost column. Profit is omitted unless crop_cost_per_ha is supplied; unpriced crops are not assigned a dataset-wide median price.",
            "market_data_note": "Prices use latest crop-matched mandi records, preferring district then state then national data. Quotes outside 0.2x-5x of the commodity's historical median are excluded. These are historical inputs, not live quotes.",
        },
        "weather_climate_risk": {
            "forecast": raw["weather_forecast"],
            "risk_by_period": raw["risk_monthly"],
            "risk_score": raw["risk_score"],
            "risk_level": raw["risk_level"],
            "risk_advice": raw["risk_advice"],
            "current_manual_weather_risk_score": raw["current_manual_weather_risk_score"],
            "source_resolution": "annual" if raw["weather_engine"].is_annual else "monthly",
            "forecast_method": ("ARIMA trend plus deterministic bootstrap of historical annual anomalies; this is a scenario projection, not a year-specific weather forecast."
                                if raw["weather_engine"].is_annual else "Seasonal ARIMA forecast with model uncertainty."),
            "geographic_resolution": "state" if raw["location_match"].get("weather_state") else "multi_state_average_or_unspecified",
            "district_level_weather_available": False,
            "resolution_note": ("Weather observations are annual and state-level. District and season do not alter this climate estimate; a state filter was applied only when an exact state match was found. Annual anomaly variation is bootstrapped deterministically from historical residuals."),
            "risk_score_note": "The 0-100 value is a modeled climate-exposure index, not a probability of crop loss.",
        },
        "fertilizer_nutrient_plan": {
            "crop": raw["fertilizer_crop"],
            "plan": raw["fertilizer_plan"],
            "top_three_plans": raw["fertilizer_plans"],
            "whole_farm_cost_inr": raw["fertilizer_total_cost"],
        },
        "inputs": _normalize_user_inputs(USER_INPUTS if user_inputs is None else user_inputs),
        "location_match": raw["location_match"],
        "input_diagnostics": raw["input_diagnostics"],
    }
    json_text = json.dumps(_json_safe(report), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    output_path = os.path.abspath(OUTPUT_JSON_PATH)
    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(json_text + "\n")
    print(json_text)
    return report


# Kaggle runs scripts/notebook cells top-to-bottom. Importing this file leaves
# the reusable classes available without training; direct execution runs the pipeline.
if __name__ == "__main__" and os.environ.get("AGRIVISION_AUTO_RUN", "1") != "0":
    RESULTS = run_pipeline(user_inputs=USER_INPUTS)
