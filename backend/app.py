"""Small API around AgriVision's dataset-free ONNX inference adapter."""
from __future__ import annotations

import os
import sys
import json
import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
os.environ.setdefault("AGRIVISION_ONNX_PATH", str(ROOT / "model" / "agrivision_website_model.onnx"))
os.environ.setdefault("AGRIVISION_ONNX_METADATA", str(ROOT / "model" / "agrivision_onnx_metadata.json"))

sys.path.insert(0, str(BASE))
from . import auth_store  # noqa: E402
from .auth_routes import frontend_origins, optional_user, require_csrf, router as account_router  # noqa: E402

logger = logging.getLogger("agrivision.api")


@lru_cache(maxsize=1)
def _get_model() -> Any:
    """Load inference dependencies on demand so a model error cannot crash the site."""
    try:
        from agrivision_onnx_inference import AgriVisionONNX

        return AgriVisionONNX()
    except Exception as exc:
        logger.exception("The ONNX inference model could not be loaded.")
        raise HTTPException(status_code=503, detail="The inference model is temporarily unavailable.") from exc


@lru_cache(maxsize=1)
def _get_metadata() -> dict[str, Any]:
    """Read public model metadata without importing ONNX Runtime."""
    try:
        metadata_path = Path(os.environ["AGRIVISION_ONNX_METADATA"])
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.exception("The public inference metadata could not be loaded.")
        raise HTTPException(status_code=503, detail="Planner metadata is temporarily unavailable.") from exc

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Keep the established public inference service available if the additive
    # account store is unavailable; account endpoints will surface their errors.
    try:
        auth_store.app_secret()
        auth_store.initialize_database()
    except Exception:
        logger.exception("Account storage could not initialize; guest predictions remain available.")
    yield


app = FastAPI(title="AgriVision inference API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(frontend_origins()),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-CSRF-Token"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    if os.getenv("APP_ENV", "development").lower() in {"production", "prod"}:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    if request.url.path == "/api/predict" or request.url.path.startswith(("/api/auth/", "/api/admin/", "/api/membership", "/api/premium/")):
        response.headers["Cache-Control"] = "no-store"
    return response

app.include_router(account_router)


class Location(BaseModel):
    state: str = Field(min_length=2)
    district: str = Field(min_length=2, max_length=80)


class SoilTest(BaseModel):
    N: float = Field(ge=0, le=500)
    P: float = Field(ge=0, le=500)
    K: float = Field(ge=0, le=500)
    ph: float = Field(ge=0, le=14)


class PredictRequest(BaseModel):
    location: Location
    season: str
    land_size: float = Field(gt=0, le=100000)
    land_unit: str
    soil_test: SoilTest
    preferred_fertilizers: list[str] = Field(default_factory=list)
    excluded_fertilizers: list[str] = Field(default_factory=list)
    risk_tolerance: str
    crop_budget_per_acre: float | None = Field(default=None, ge=0, le=10_000_000)
    fertilizer_budget_per_acre: float | None = Field(default=None, ge=0, le=10_000_000)
    crop_cost_per_ha: dict[str, float] = Field(default_factory=dict)
    target_year: int = Field(ge=2021, le=2050)
    optimization_goal: str = "balance_roi"

    @field_validator("season")
    @classmethod
    def valid_season(cls, value: str) -> str:
        if value not in {"Kharif", "Rabi", "Summer"}:
            raise ValueError("Choose Kharif, Rabi, or Summer.")
        return value

    @field_validator("land_unit")
    @classmethod
    def valid_unit(cls, value: str) -> str:
        if value not in {"acres", "hectares"}:
            raise ValueError("Choose acres or hectares.")
        return value

    @field_validator("risk_tolerance")
    @classmethod
    def valid_risk(cls, value: str) -> str:
        if value not in {"low", "medium", "high"}:
            raise ValueError("Choose low, medium, or high.")
        return value


@app.get("/api/health")
def health() -> dict[str, str]:
    _get_model()
    return {"status": "ready", "model": "AgriVision ONNX"}


@app.get("/api/metadata")
def get_metadata() -> dict[str, Any]:
    # Keep the browser payload compact; the complete metadata stays server-side.
    metadata = _get_metadata()
    return {
        "states": metadata["states"],
        "crops": metadata["crops"],
        "fertilizer_products": [p["name"] for p in metadata["fertilizer_products"]],
        "crop_count": len(metadata["crops"]),
        "climate_summary": {
            "training_weather_through": metadata["last_training_weather_year"],
            "yield_mae_t_ha": metadata["yield_model_metrics"]["holdout_mae_t_ha"],
        },
    }


@app.post("/api/predict")
def predict(request: Request, payload: PredictRequest, user: dict[str, Any] | None = Depends(optional_user)) -> dict[str, Any]:
    try:
        if user:
            # Prediction remains guest-accessible, but report persistence is a
            # state change and therefore needs the same CSRF/Origin check as account APIs.
            require_csrf(request)
        inputs = payload.model_dump()
        report = _get_model().predict(inputs)
        if user:
            now = auth_store.timestamp()
            with auth_store.db_session() as connection:
                connection.execute(
                    "INSERT INTO planner_reports(user_id, inputs_json, report_json, created_at) VALUES (?, ?, ?, ?)",
                    (user["id"], auth_store.json_dump(inputs), auth_store.json_dump(report), now),
                )
                keep = 1000 if user["membership_status"] == "approved" else 10
                connection.execute(
                    "DELETE FROM planner_reports WHERE user_id = ? AND id NOT IN "
                    "(SELECT id FROM planner_reports WHERE user_id = ? ORDER BY created_at DESC, id DESC LIMIT ?)",
                    (user["id"], user["id"], keep),
                )
                auth_store.add_activity(connection, user["id"], "field_report_created", {
                    "state": inputs["location"]["state"], "season": inputs["season"],
                })
        return report
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # surfaced as a service error without exposing internals
        raise HTTPException(status_code=500, detail="Prediction could not be completed. Check the inference service logs.") from exc


# In the free single-host deployment, Next.js is exported as static files and
# served from this ASGI process so browser API calls and session cookies stay
# same-origin. Local development continues to use `next dev` on port 3000.
web_directory = ROOT / "out"
if web_directory.is_dir():
    app.mount("/", StaticFiles(directory=web_directory, html=True), name="website")

