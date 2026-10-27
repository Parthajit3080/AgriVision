"""Account, membership, saved-planner and admin APIs.

Browser authentication uses opaque, random, server-side sessions in HttpOnly cookies.
The database stores only a SHA-256 digest of each session secret so sessions can be
revoked immediately. Mutating account/admin routes require a signed CSRF token and
an allow-listed Origin.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import smtplib
import sqlite3
from datetime import timedelta
from email.message import EmailMessage
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import auth_store as store

router = APIRouter(prefix="/api")
logger = logging.getLogger("agrivision.auth")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SignupBody(StrictModel):
    email: str = Field(min_length=3, max_length=254)
    display_name: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def clean_email(cls, value: str) -> str:
        return store.validate_email(value)

    @field_validator("display_name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Enter your name.")
        return value

    @field_validator("password")
    @classmethod
    def valid_password_bytes(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 512:
            raise ValueError("Password is too long.")
        return value


class LoginBody(StrictModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def clean_email(cls, value: str) -> str:
        return store.validate_email(value)


class AdminLoginBody(StrictModel):
    username: str = Field(min_length=2, max_length=80)
    password: str = Field(min_length=1, max_length=128)


class ForgotPasswordBody(StrictModel):
    email: str = Field(min_length=3, max_length=254)

    @field_validator("email")
    @classmethod
    def clean_email(cls, value: str) -> str:
        return store.validate_email(value)


class ResetPasswordBody(StrictModel):
    token: str = Field(min_length=32, max_length=256)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("password")
    @classmethod
    def valid_password_bytes(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 512:
            raise ValueError("Password is too long.")
        return value


class ProfileBody(StrictModel):
    display_name: str = Field(min_length=1, max_length=80)

    @field_validator("display_name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Enter your name.")
        return value


class PreferencesBody(StrictModel):
    form: dict[str, Any] = Field(default_factory=dict)
    preferred: list[str] = Field(default_factory=list, max_length=30)
    excluded: list[str] = Field(default_factory=list, max_length=30)

    @field_validator("form")
    @classmethod
    def validate_form(cls, value: dict[str, Any]) -> dict[str, Any]:
        allowed = {"state", "district", "season", "land_size", "land_unit", "N", "P", "K", "ph",
                   "risk_tolerance", "crop_budget_per_acre", "fertilizer_budget_per_acre", "crop_cost_per_ha",
                   "target_year", "optimization_goal"}
        if set(value) - allowed:
            raise ValueError("Preferences contain unsupported fields.")
        if len(json.dumps(value, ensure_ascii=False)) > 5000:
            raise ValueError("Preferences are too large.")
        result = {key: str(item)[:120] for key, item in value.items()}
        if result.get("season") not in {None, "Kharif", "Rabi", "Summer"}:
            raise ValueError("Choose a valid season.")
        if result.get("land_unit") not in {None, "acres", "hectares"}:
            raise ValueError("Choose a valid land unit.")
        if result.get("risk_tolerance") not in {None, "low", "medium", "high"}:
            raise ValueError("Choose a valid risk tolerance.")
        return result

    @field_validator("preferred", "excluded")
    @classmethod
    def validate_products(cls, value: list[str]) -> list[str]:
        if any(not item.strip() or len(item) > 80 for item in value):
            raise ValueError("Product names must be between 1 and 80 characters.")
        return list(dict.fromkeys(item.strip() for item in value))


class MembershipBody(StrictModel):
    plan_code: Literal["field-plus"] = "field-plus"
    message: str = Field(default="", max_length=500)


class ReviewBody(StrictModel):
    decision: Literal["approved", "rejected"]
    note: str = Field(default="", max_length=500)


class RoleBody(StrictModel):
    role: Literal["user", "admin", "superadmin"]


class AccountStatusBody(StrictModel):
    status: Literal["active", "suspended"]


def frontend_origins() -> set[str]:
    configured = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
    return {origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()}


def _cookie_samesite() -> Literal["lax", "strict", "none"]:
    value = os.getenv("AGRIVISION_COOKIE_SAMESITE", "lax").lower()
    if value not in {"lax", "strict", "none"}:
        value = "lax"
    if value == "none" and not store.session_cookie_secure():
        raise RuntimeError("SameSite=None requires AGRIVISION_COOKIE_SECURE=true.")
    return value  # type: ignore[return-value]


def _csrf_token() -> str:
    issued = str(int(store.utc_now().timestamp()))
    nonce = secrets.token_urlsafe(24)
    signed = f"{issued}.{nonce}"
    signature = hmac.new(store.app_secret(), signed.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{signed}.{signature}"


def require_csrf(request: Request) -> None:
    origin = request.headers.get("origin", "").rstrip("/")
    if not origin or origin not in frontend_origins():
        raise HTTPException(status_code=403, detail="Request origin is not allowed.")
    cookie = request.cookies.get(store.CSRF_COOKIE_NAME, "")
    header = request.headers.get("X-CSRF-Token", "")
    if not cookie or not header or not hmac.compare_digest(cookie, header):
        raise HTTPException(status_code=403, detail="Refresh the page and try again.")
    try:
        issued, nonce, signature = cookie.split(".", 2)
        issued_at = int(issued)
    except (ValueError, TypeError):
        raise HTTPException(status_code=403, detail="Refresh the page and try again.") from None
    age = int(store.utc_now().timestamp()) - issued_at
    signed = f"{issued}.{nonce}"
    expected = hmac.new(store.app_secret(), signed.encode("ascii"), hashlib.sha256).hexdigest()
    if age < -60 or age > 3600 or not hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=403, detail="Refresh the page and try again.")


def set_csrf_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        store.CSRF_COOKIE_NAME,
        token,
        max_age=3600,
        path="/",
        secure=store.session_cookie_secure(),
        httponly=False,
        samesite=_cookie_samesite(),
    )


def set_session_cookie(response: Response, token: str, expires_at: str) -> None:
    max_age = max(int((store.parse_timestamp(expires_at) - store.utc_now()).total_seconds()), 0)
    response.set_cookie(
        store.SESSION_COOKIE_NAME,
        token,
        max_age=max_age,
        path="/",
        secure=store.session_cookie_secure(),
        httponly=True,
        samesite=_cookie_samesite(),
    )
    response.headers["Cache-Control"] = "no-store"


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        store.SESSION_COOKIE_NAME,
        path="/",
        secure=store.session_cookie_secure(),
        httponly=True,
        samesite=_cookie_samesite(),
    )
    response.headers["Cache-Control"] = "no-store"


def clear_csrf_cookie(response: Response) -> None:
    response.delete_cookie(
        store.CSRF_COOKIE_NAME,
        path="/",
        secure=store.session_cookie_secure(),
        httponly=False,
        samesite=_cookie_samesite(),
    )


def _user_from_request(request: Request, *, required: bool) -> dict[str, Any] | None:
    token = request.cookies.get(store.SESSION_COOKIE_NAME)
    if not token:
        if required:
            raise HTTPException(status_code=401, detail="Sign in to continue.")
        return None
    try:
        with store.db_session() as connection:
            result = store.lookup_session(connection, token)
            if not result:
                if required:
                    raise HTTPException(status_code=401, detail="Sign in to continue.")
                return None
            session, user = result
            public = store.public_user(user)
            public["_session_id"] = session["id"]
            public["_session_token_hash"] = session["token_hash"]
            return public
    except sqlite3.Error as exc:
        if required:
            raise HTTPException(status_code=503, detail="Account services are temporarily unavailable.") from exc
        logger.warning("Session validation is unavailable; handling the request as a guest.")
        return None


def current_user(request: Request) -> dict[str, Any]:
    user = _user_from_request(request, required=True)
    assert user is not None
    return user


def optional_user(request: Request) -> dict[str, Any] | None:
    return _user_from_request(request, required=False)


def admin_user(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    if user["role"] not in {"admin", "superadmin"}:
        raise HTTPException(status_code=403, detail="Administrator access is required.")
    return user


def superadmin_user(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    if user["role"] != "superadmin":
        raise HTTPException(status_code=403, detail="Superadmin access is required.")
    return user


def premium_user(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    if user["membership_status"] != "approved":
        raise HTTPException(status_code=403, detail="Premium membership is required.")
    return user


def _rate_limit(request: Request, scope: str, identifier: str, limit: int, window: int) -> None:
    ip = request.client.host if request.client else "unknown"
    # Commit each attempt independently of the login/reset transaction. Otherwise a
    # rejected request would roll back its own rate-limit counter with its HTTP error.
    with store.db_session() as rate_connection:
        rate_connection.execute("BEGIN IMMEDIATE")
        store.enforce_rate_limit(rate_connection, scope=scope, bucket=f"ip:{ip}", limit=limit, window_seconds=window)
        if identifier:
            store.enforce_rate_limit(rate_connection, scope=f"{scope}-identity", bucket=identifier, limit=limit, window_seconds=window)


def _new_session(connection: sqlite3.Connection, request: Request, response: Response, user: sqlite3.Row) -> dict[str, Any]:
    token, _, expires_at = store.create_session(connection, user["id"], request)
    set_session_cookie(response, token, expires_at)
    response.headers["Cache-Control"] = "no-store"
    return store.public_user(user)


@router.get("/auth/csrf")
def issue_csrf(response: Response) -> dict[str, str]:
    token = _csrf_token()
    set_csrf_cookie(response, token)
    response.headers["Cache-Control"] = "no-store"
    return {"csrf_token": token}


@router.post("/auth/signup", status_code=201)
def signup(body: SignupBody, request: Request, response: Response, _csrf: None = Depends(require_csrf)) -> dict[str, Any]:
    with store.db_session() as connection:
        _rate_limit(request, "signup", body.email, 10, 3600)
        existing = connection.execute("SELECT id FROM users WHERE email = ? COLLATE NOCASE", (body.email,)).fetchone()
        if existing:
            raise HTTPException(status_code=409, detail="An account with that email already exists.")
        password_hash = store.hash_password(body.password)
        now = store.timestamp()
        cursor = connection.execute(
            "INSERT INTO users(email, display_name, password_hash, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (body.email, body.display_name, password_hash, now, now),
        )
        user = connection.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone()
        store.add_activity(connection, user["id"], "account_created")
        public = _new_session(connection, request, response, user)
    return {"user": public}


@router.post("/auth/login")
def login(body: LoginBody, request: Request, response: Response, _csrf: None = Depends(require_csrf)) -> dict[str, Any]:
    with store.db_session() as connection:
        _rate_limit(request, "login", body.email, 12, 900)
        user = connection.execute("SELECT * FROM users WHERE email = ? COLLATE NOCASE", (body.email,)).fetchone()
        encoded = user["password_hash"] if user else store._DUMMY_HASH
        valid = store.verify_password(body.password, encoded)
        if not user or not valid or user["account_status"] != "active":
            raise HTTPException(status_code=401, detail="Email or password is incorrect.")
        store.add_activity(connection, user["id"], "signed_in")
        public = _new_session(connection, request, response, user)
    return {"user": public}


@router.post("/admin/auth/login")
def admin_login(body: AdminLoginBody, request: Request, response: Response, _csrf: None = Depends(require_csrf)) -> dict[str, Any]:
    identity = body.username.strip()
    with store.db_session() as connection:
        _rate_limit(request, "admin-login", identity, 8, 900)
        user = connection.execute(
            "SELECT * FROM users WHERE (username = ? COLLATE NOCASE OR email = ? COLLATE NOCASE) "
            "AND role IN ('admin', 'superadmin')",
            (identity, identity),
        ).fetchone()
        encoded = user["password_hash"] if user else store._DUMMY_HASH
        valid = store.verify_password(body.password, encoded)
        if not user or not valid or user["account_status"] != "active":
            raise HTTPException(status_code=401, detail="Admin ID or password is incorrect.")
        store.add_activity(connection, user["id"], "admin_signed_in")
        public = _new_session(connection, request, response, user)
    return {"user": public}


@router.post("/auth/logout", status_code=204)
def logout(request: Request, response: Response, user: dict[str, Any] | None = Depends(optional_user), _csrf: None = Depends(require_csrf)) -> Response:
    if user:
        with store.db_session() as connection:
            connection.execute("UPDATE sessions SET revoked_at = ? WHERE id = ? AND user_id = ?", (store.timestamp(), user["_session_id"], user["id"]))
            store.add_activity(connection, user["id"], "signed_out")
    clear_session_cookie(response)
    response.status_code = 204
    return response


@router.post("/auth/logout-all", status_code=204)
def logout_all(request: Request, response: Response, user: dict[str, Any] = Depends(current_user), _csrf: None = Depends(require_csrf)) -> Response:
    with store.db_session() as connection:
        connection.execute("UPDATE sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL", (store.timestamp(), user["id"]))
        store.add_activity(connection, user["id"], "signed_out_all_sessions")
    clear_session_cookie(response)
    response.status_code = 204
    return response


@router.get("/auth/me")
def me(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    user.pop("_session_id", None)
    user.pop("_session_token_hash", None)
    return {"user": user}


@router.patch("/auth/profile")
def update_profile(body: ProfileBody, user: dict[str, Any] = Depends(current_user), _csrf: None = Depends(require_csrf)) -> dict[str, Any]:
    with store.db_session() as connection:
        connection.execute("UPDATE users SET display_name = ?, updated_at = ? WHERE id = ?", (body.display_name, store.timestamp(), user["id"]))
        store.add_activity(connection, user["id"], "profile_updated")
        updated = connection.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()
        return {"user": store.public_user(updated)}


@router.put("/auth/preferences")
def save_preferences(body: PreferencesBody, user: dict[str, Any] = Depends(current_user), _csrf: None = Depends(require_csrf)) -> dict[str, Any]:
    preferences = {"form": body.form, "preferred": body.preferred, "excluded": body.excluded}
    with store.db_session() as connection:
        connection.execute("UPDATE users SET saved_preferences_json = ?, updated_at = ? WHERE id = ?", (store.json_dump(preferences), store.timestamp(), user["id"]))
        store.add_activity(connection, user["id"], "planner_preferences_saved")
    return {"preferences": preferences}


@router.get("/auth/sessions")
def list_sessions(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    with store.db_session() as connection:
        rows = connection.execute(
            "SELECT id, created_at, expires_at, last_seen_at, user_agent, ip_hash, token_hash "
            "FROM sessions WHERE user_id = ? AND revoked_at IS NULL AND expires_at > ? ORDER BY last_seen_at DESC",
            (user["id"], store.timestamp()),
        ).fetchall()
    return {"sessions": [{
        "id": row["id"], "created_at": row["created_at"], "expires_at": row["expires_at"],
        "last_seen_at": row["last_seen_at"], "device": row["user_agent"] or "Browser session",
        "current": row["id"] == user["_session_id"],
    } for row in rows]}


@router.get("/auth/activity")
def account_activity(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    with store.db_session() as connection:
        rows = connection.execute(
            "SELECT event_type, details_json, created_at FROM activity_events "
            "WHERE user_id = ? ORDER BY created_at DESC LIMIT 50",
            (user["id"],),
        ).fetchall()
    return {"items": [
        {"event_type": row["event_type"], "details": store.json_load(row["details_json"], {}), "created_at": row["created_at"]}
        for row in rows
    ]}


@router.delete("/auth/sessions/{session_id}", status_code=204)
def revoke_session(session_id: int, request: Request, response: Response, user: dict[str, Any] = Depends(current_user), _csrf: None = Depends(require_csrf)) -> Response:
    with store.db_session() as connection:
        cursor = connection.execute("UPDATE sessions SET revoked_at = ? WHERE id = ? AND user_id = ? AND revoked_at IS NULL", (store.timestamp(), session_id, user["id"]))
        if not cursor.rowcount:
            raise HTTPException(status_code=404, detail="Session not found.")
        store.add_activity(connection, user["id"], "session_revoked", {"session_id": session_id})
    if session_id == user["_session_id"]:
        clear_session_cookie(response)
    response.status_code = 204
    return response


@router.post("/auth/forgot-password")
def forgot_password(body: ForgotPasswordBody, request: Request, background_tasks: BackgroundTasks, _csrf: None = Depends(require_csrf)) -> dict[str, Any]:
    reset_url = None
    with store.db_session() as connection:
        _rate_limit(request, "forgot-password", body.email, 5, 3600)
        user = connection.execute("SELECT id, email, account_status FROM users WHERE email = ? COLLATE NOCASE", (body.email,)).fetchone()
        if user and user["account_status"] == "active":
            raw_token = secrets.token_urlsafe(32)
            now = store.utc_now()
            expires = now + timedelta(minutes=30)
            connection.execute("UPDATE password_reset_tokens SET consumed_at = ? WHERE user_id = ? AND consumed_at IS NULL", (store.timestamp(now), user["id"]))
            connection.execute(
                "INSERT INTO password_reset_tokens(user_id, token_hash, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (user["id"], hashlib.sha256(raw_token.encode()).hexdigest(), store.timestamp(now), store.timestamp(expires)),
            )
            base_url = os.getenv("APP_PUBLIC_URL", next(iter(frontend_origins())))
            reset_url = f"{base_url.rstrip('/')}/reset-password?token={raw_token}"
            if os.getenv("SMTP_HOST"):
                background_tasks.add_task(_send_reset_email, user["email"], reset_url)
            store.add_activity(connection, user["id"], "password_reset_requested")
            if os.getenv("APP_ENV", "development").lower() in {"development", "dev", "test"} and not os.getenv("SMTP_HOST"):
                return {"message": "If the account exists, password reset instructions have been sent.", "development_reset_url": reset_url}
    return {"message": "If the account exists, password reset instructions have been sent."}


def _send_reset_email(email: str, reset_url: str) -> None:
    host = os.getenv("SMTP_HOST")
    if not host:
        return
    port = int(os.getenv("SMTP_PORT", "587"))
    sender = os.getenv("SMTP_FROM", "AgriVision <no-reply@localhost>")
    message = EmailMessage()
    message["Subject"] = "Reset your AgriVision password"
    message["From"] = sender
    message["To"] = email
    message.set_content(
        "A password reset was requested for your AgriVision account.\n\n"
        f"Use this one-time link within 30 minutes:\n{reset_url}\n\n"
        "If you did not request this change, you can ignore this message."
    )
    try:
        username = os.getenv("SMTP_USERNAME")
        password = os.getenv("SMTP_PASSWORD")
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=15) as server:
                if username:
                    server.login(username, password or "")
                server.send_message(message)
        else:
            with smtplib.SMTP(host, port, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                if username:
                    server.login(username, password or "")
                server.send_message(message)
    except Exception:
        # Do not log the email's one-time reset URL or token.
        logger.exception("Password reset email delivery failed.")


@router.post("/auth/reset-password")
def reset_password(body: ResetPasswordBody, request: Request, response: Response, _csrf: None = Depends(require_csrf)) -> dict[str, str]:
    digest = hashlib.sha256(body.token.encode()).hexdigest()
    now = store.timestamp()
    with store.db_session() as connection:
        _rate_limit(request, "reset-password", "", 10, 3600)
        token = connection.execute(
            "SELECT id, user_id FROM password_reset_tokens WHERE token_hash = ? AND consumed_at IS NULL AND expires_at > ?",
            (digest, now),
        ).fetchone()
        if not token:
            raise HTTPException(status_code=400, detail="This reset link is invalid or expired. Request a new one.")
        new_hash = store.hash_password(body.password)
        connection.execute("UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?", (new_hash, now, token["user_id"]))
        connection.execute("UPDATE password_reset_tokens SET consumed_at = ? WHERE user_id = ? AND consumed_at IS NULL", (now, token["user_id"]))
        connection.execute("UPDATE sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL", (now, token["user_id"]))
        store.add_activity(connection, token["user_id"], "password_reset_completed")
    clear_session_cookie(response)
    return {"message": "Password updated. Sign in with your new password."}


@router.get("/membership/plans")
def membership_plans() -> dict[str, Any]:
    return {"plans": [{
        "code": "field-plus",
        "name": "Field Plus",
        "review": "Admin-approved membership",
        "billing": "No payment or recurring billing is configured in this prototype.",
        "benefits": [
            "Keep up to 1,000 saved field reports (standard accounts keep the latest 10).",
            "Filter report history by crop, state, season, climate risk and date.",
            "See personal yield, crop and climate summaries across saved reports.",
            "Premium account badge and early access to future planning tools.",
        ],
    }]}


@router.get("/membership")
def membership_status(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    with store.db_session() as connection:
        latest = connection.execute(
            "SELECT id, plan_code, message, status, decision_note, created_at, updated_at, decided_at "
            "FROM membership_applications WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
            (user["id"],),
        ).fetchone()
    return {"status": user["membership_status"], "application": dict(latest) if latest else None}


@router.post("/membership/requests", status_code=201)
def request_membership(body: MembershipBody, user: dict[str, Any] = Depends(current_user), _csrf: None = Depends(require_csrf)) -> dict[str, Any]:
    now = store.timestamp()
    with store.db_session() as connection:
        if user["membership_status"] == "approved":
            raise HTTPException(status_code=409, detail="Your premium membership is already active.")
        pending = connection.execute("SELECT id FROM membership_applications WHERE user_id = ? AND status = 'pending'", (user["id"],)).fetchone()
        if pending:
            raise HTTPException(status_code=409, detail="You already have a request under review.")
        cursor = connection.execute(
            "INSERT INTO membership_applications(user_id, plan_code, message, status, created_at, updated_at) VALUES (?, ?, ?, 'pending', ?, ?)",
            (user["id"], body.plan_code, body.message.strip(), now, now),
        )
        connection.execute("UPDATE users SET membership_status = 'pending', updated_at = ? WHERE id = ?", (now, user["id"]))
        store.add_activity(connection, user["id"], "premium_request_submitted", {"application_id": cursor.lastrowid})
        application = connection.execute("SELECT * FROM membership_applications WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return {"application": dict(application)}


def _report_summary(row: sqlite3.Row) -> dict[str, Any]:
    report = store.json_load(row["report_json"], {})
    inputs = report.get("inputs") or store.json_load(row["inputs_json"], {})
    ranking = report.get("crop_profit_ranking", {})
    recommendations = ranking.get("recommendations", [])
    top = recommendations[0] if recommendations else {}
    climate = report.get("weather_climate_risk", {})
    location = inputs.get("location", {})
    return {
        "id": row["id"], "created_at": row["created_at"],
        "state": location.get("state", ""), "district": location.get("district", ""),
        "season": inputs.get("season", ""), "target_year": inputs.get("target_year"),
        "land_size": inputs.get("land_size"), "land_unit": inputs.get("land_unit", ""),
        "top_crop": top.get("crop", ""), "estimated_yield_t_ha": top.get("estimated_yield_t_ha"),
        "risk_level": climate.get("risk_level", ""),
        "risk_score": climate.get("risk_guidance", {}).get("score", climate.get("risk_score")),
    }


@router.get("/auth/history")
def report_history(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    premium = user["membership_status"] == "approved"
    limit = 1000 if premium else 10
    with store.db_session() as connection:
        rows = connection.execute(
            "SELECT id, inputs_json, report_json, created_at FROM planner_reports WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user["id"], limit),
        ).fetchall()
    return {"items": [_report_summary(row) for row in rows], "limit": limit, "premium": premium}


@router.get("/auth/history/{report_id}")
def get_saved_report(report_id: int, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    with store.db_session() as connection:
        row = connection.execute("SELECT id, report_json, created_at FROM planner_reports WHERE id = ? AND user_id = ?", (report_id, user["id"])).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Saved report not found.")
    return {"id": row["id"], "created_at": row["created_at"], "report": store.json_load(row["report_json"], {})}


@router.delete("/auth/history/{report_id}", status_code=204)
def delete_saved_report(report_id: int, user: dict[str, Any] = Depends(current_user), _csrf: None = Depends(require_csrf)) -> Response:
    with store.db_session() as connection:
        cursor = connection.execute("DELETE FROM planner_reports WHERE id = ? AND user_id = ?", (report_id, user["id"]))
        if not cursor.rowcount:
            raise HTTPException(status_code=404, detail="Saved report not found.")
        store.add_activity(connection, user["id"], "saved_report_deleted", {"report_id": report_id})
    return Response(status_code=204)


@router.get("/premium/analytics")
def premium_analytics(user: dict[str, Any] = Depends(premium_user)) -> dict[str, Any]:
    with store.db_session() as connection:
        rows = connection.execute("SELECT report_json FROM planner_reports WHERE user_id = ? ORDER BY created_at DESC LIMIT 1000", (user["id"],)).fetchall()
    crops: dict[str, int] = {}
    yields: list[float] = []
    risks: list[float] = []
    seasons: dict[str, int] = {}
    for row in rows:
        report = store.json_load(row["report_json"], {})
        inputs = report.get("inputs", {})
        season = inputs.get("season")
        if season:
            seasons[season] = seasons.get(season, 0) + 1
        ranking = report.get("crop_profit_ranking", {}).get("recommendations", [])
        if ranking:
            crop = ranking[0].get("crop")
            if crop:
                crops[crop] = crops.get(crop, 0) + 1
            try:
                value = float(ranking[0].get("estimated_yield_t_ha"))
                if 0 <= value <= 1000:
                    yields.append(value)
            except (TypeError, ValueError):
                pass
        climate = report.get("weather_climate_risk", {})
        try:
            risk = float(climate.get("risk_guidance", {}).get("score", climate.get("risk_score")))
            if 0 <= risk <= 100:
                risks.append(risk)
        except (TypeError, ValueError):
            pass
    return {
        "report_count": len(rows), "top_crop": max(crops, key=crops.get) if crops else None,
        "crop_counts": crops, "season_counts": seasons,
        "average_top_crop_yield_t_ha": round(sum(yields) / len(yields), 2) if yields else None,
        "average_climate_risk": round(sum(risks) / len(risks), 1) if risks else None,
    }


@router.get("/admin/dashboard")
def admin_dashboard(user: dict[str, Any] = Depends(admin_user)) -> dict[str, Any]:
    with store.db_session() as connection:
        counts = {
            "users": connection.execute("SELECT COUNT(*) AS count FROM users WHERE role = 'user'").fetchone()["count"],
            "admins": connection.execute("SELECT COUNT(*) AS count FROM users WHERE role IN ('admin','superadmin') AND account_status = 'active'").fetchone()["count"],
            "suspended": connection.execute("SELECT COUNT(*) AS count FROM users WHERE account_status = 'suspended'").fetchone()["count"],
            "premium_pending": connection.execute("SELECT COUNT(*) AS count FROM membership_applications WHERE status = 'pending'").fetchone()["count"],
            "premium_approved": connection.execute("SELECT COUNT(*) AS count FROM users WHERE membership_status = 'approved'").fetchone()["count"],
            "predictions": connection.execute("SELECT COUNT(*) AS count FROM planner_reports").fetchone()["count"],
        }
        events = connection.execute(
            "SELECT a.id, a.action, a.details_json, a.created_at, u.email AS actor_email "
            "FROM admin_audit_logs a LEFT JOIN users u ON u.id = a.actor_user_id ORDER BY a.created_at DESC LIMIT 8"
        ).fetchall()
    return {"metrics": counts, "model": "AgriVision ONNX", "database": "ready", "recent_admin_actions": [
        {"id": row["id"], "action": row["action"], "details": store.json_load(row["details_json"], {}), "created_at": row["created_at"], "actor_email": row["actor_email"]}
        for row in events
    ], "viewer_role": user["role"]}


@router.get("/admin/users")
def admin_users(
    q: str = Query(default="", max_length=120),
    role: str = Query(default="", pattern="^(|user|admin|superadmin)$"),
    status: str = Query(default="", pattern="^(|active|suspended)$"),
    membership: str = Query(default="", pattern="^(|none|pending|approved|rejected)$"),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=100000),
    user: dict[str, Any] = Depends(admin_user),
) -> dict[str, Any]:
    clauses: list[str] = []
    params: list[Any] = []
    if q.strip():
        escaped = q.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        clauses.append("(email LIKE ? ESCAPE '\\' OR display_name LIKE ? ESCAPE '\\' OR COALESCE(username, '') LIKE ? ESCAPE '\\')")
        params.extend([f"%{escaped}%"] * 3)
    for column, value in (("role", role), ("account_status", status), ("membership_status", membership)):
        if value:
            clauses.append(f"{column} = ?")
            params.append(value)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with store.db_session() as connection:
        total = connection.execute("SELECT COUNT(*) AS count FROM users" + where, params).fetchone()["count"]
        rows = connection.execute(
            "SELECT id, email, username, display_name, role, account_status, membership_status, created_at, last_login_at "
            "FROM users" + where + " ORDER BY created_at DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
    return {"items": [dict(row) for row in rows], "total": total, "limit": limit, "offset": offset, "viewer_role": user["role"]}


@router.get("/admin/users/{user_id}")
def admin_user_detail(user_id: int, _user: dict[str, Any] = Depends(admin_user)) -> dict[str, Any]:
    with store.db_session() as connection:
        row = connection.execute(
            "SELECT id, email, username, display_name, role, account_status, membership_status, premium_since, created_at, updated_at, last_login_at "
            "FROM users WHERE id = ?", (user_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found.")
        activities = connection.execute(
            "SELECT event_type, details_json, created_at FROM activity_events WHERE user_id = ? ORDER BY created_at DESC LIMIT 30",
            (user_id,),
        ).fetchall()
    return {"user": dict(row), "activity": [
        {"event_type": item["event_type"], "details": store.json_load(item["details_json"], {}), "created_at": item["created_at"]}
        for item in activities
    ]}


@router.get("/admin/membership-requests")
def admin_membership_requests(
    status: str = Query(default="pending", pattern="^(pending|approved|rejected|all)$"),
    _user: dict[str, Any] = Depends(admin_user),
) -> dict[str, Any]:
    with store.db_session() as connection:
        where = "" if status == "all" else "WHERE a.status = ?"
        params: tuple[Any, ...] = () if status == "all" else (status,)
        rows = connection.execute(
            "SELECT a.id, a.user_id, a.plan_code, a.message, a.status, a.decision_note, a.created_at, a.updated_at, a.decided_at, "
            "u.email, u.display_name FROM membership_applications a JOIN users u ON u.id = a.user_id "
            f"{where} ORDER BY a.created_at DESC LIMIT 250", params,
        ).fetchall()
    return {"items": [dict(row) for row in rows]}


@router.post("/admin/membership-requests/{application_id}/decision")
def review_membership(
    application_id: int,
    body: ReviewBody,
    request: Request,
    admin: dict[str, Any] = Depends(admin_user),
    _csrf: None = Depends(require_csrf),
) -> dict[str, Any]:
    now = store.timestamp()
    with store.db_session() as connection:
        application = connection.execute("SELECT * FROM membership_applications WHERE id = ?", (application_id,)).fetchone()
        if not application:
            raise HTTPException(status_code=404, detail="Membership request not found.")
        if application["status"] != "pending":
            raise HTTPException(status_code=409, detail="This membership request has already been reviewed.")
        if application["user_id"] == admin["id"]:
            raise HTTPException(status_code=403, detail="You cannot review your own membership request.")
        connection.execute(
            "UPDATE membership_applications SET status = ?, decision_by = ?, decision_note = ?, updated_at = ?, decided_at = ? WHERE id = ?",
            (body.decision, admin["id"], body.note.strip(), now, now, application_id),
        )
        premium_status = "approved" if body.decision == "approved" else "rejected"
        premium_since = now if body.decision == "approved" else None
        connection.execute("UPDATE users SET membership_status = ?, premium_since = ?, updated_at = ? WHERE id = ?", (premium_status, premium_since, now, application["user_id"]))
        store.add_activity(connection, application["user_id"], f"premium_request_{body.decision}", {"application_id": application_id})
        store.add_audit(connection, admin["id"], f"membership_{body.decision}", application["user_id"], request, {"application_id": application_id, "note": body.note.strip()})
        updated = connection.execute("SELECT id, user_id, status, decision_note, decided_at FROM membership_applications WHERE id = ?", (application_id,)).fetchone()
    return {"application": dict(updated)}


@router.patch("/admin/users/{user_id}/role")
def admin_change_role(
    user_id: int,
    body: RoleBody,
    request: Request,
    admin: dict[str, Any] = Depends(superadmin_user),
    _csrf: None = Depends(require_csrf),
) -> dict[str, Any]:
    now = store.timestamp()
    with store.db_session() as connection:
        target = connection.execute("SELECT id, role, email FROM users WHERE id = ?", (user_id,)).fetchone()
        if not target:
            raise HTTPException(status_code=404, detail="User not found.")
        if user_id == admin["id"] and body.role != "superadmin":
            raise HTTPException(status_code=409, detail="You cannot remove your own superadmin role.")
        if target["role"] == "superadmin" and body.role != "superadmin":
            count = connection.execute("SELECT COUNT(*) AS count FROM users WHERE role = 'superadmin' AND account_status = 'active'").fetchone()["count"]
            if count <= 1:
                raise HTTPException(status_code=409, detail="The last active superadmin cannot be demoted.")
        previous = target["role"]
        connection.execute("UPDATE users SET role = ?, updated_at = ? WHERE id = ?", (body.role, now, user_id))
        connection.execute("UPDATE sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL", (now, user_id))
        store.add_audit(connection, admin["id"], "user_role_changed", user_id, request, {"from": previous, "to": body.role})
        updated = connection.execute("SELECT id, email, username, display_name, role, account_status, membership_status FROM users WHERE id = ?", (user_id,)).fetchone()
    return {"user": dict(updated)}


@router.patch("/admin/users/{user_id}/status")
def admin_change_status(
    user_id: int,
    body: AccountStatusBody,
    request: Request,
    admin: dict[str, Any] = Depends(admin_user),
    _csrf: None = Depends(require_csrf),
) -> dict[str, Any]:
    now = store.timestamp()
    with store.db_session() as connection:
        target = connection.execute("SELECT id, role, account_status FROM users WHERE id = ?", (user_id,)).fetchone()
        if not target:
            raise HTTPException(status_code=404, detail="User not found.")
        if target["role"] in {"admin", "superadmin"} and admin["role"] != "superadmin":
            raise HTTPException(status_code=403, detail="Only a superadmin can manage another administrator.")
        if user_id == admin["id"] and body.status == "suspended":
            raise HTTPException(status_code=409, detail="You cannot suspend your own account.")
        if target["role"] == "superadmin" and body.status == "suspended":
            count = connection.execute("SELECT COUNT(*) AS count FROM users WHERE role = 'superadmin' AND account_status = 'active'").fetchone()["count"]
            if count <= 1:
                raise HTTPException(status_code=409, detail="The last active superadmin cannot be suspended.")
        previous = target["account_status"]
        connection.execute("UPDATE users SET account_status = ?, updated_at = ? WHERE id = ?", (body.status, now, user_id))
        if body.status == "suspended":
            connection.execute("UPDATE sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL", (now, user_id))
        store.add_audit(connection, admin["id"], f"user_{body.status}", user_id, request, {"from": previous, "to": body.status})
        updated = connection.execute("SELECT id, email, username, display_name, role, account_status, membership_status FROM users WHERE id = ?", (user_id,)).fetchone()
    return {"user": dict(updated)}


@router.get("/admin/audit-logs")
def audit_logs(
    limit: int = Query(default=100, ge=1, le=250),
    offset: int = Query(default=0, ge=0, le=100000),
    _user: dict[str, Any] = Depends(admin_user),
) -> dict[str, Any]:
    with store.db_session() as connection:
        rows = connection.execute(
            "SELECT l.id, l.action, l.details_json, l.created_at, l.actor_user_id, l.target_user_id, "
            "actor.email AS actor_email, target.email AS target_email "
            "FROM admin_audit_logs l LEFT JOIN users actor ON actor.id = l.actor_user_id "
            "LEFT JOIN users target ON target.id = l.target_user_id "
            "ORDER BY l.created_at DESC LIMIT ? OFFSET ?", (limit, offset),
        ).fetchall()
        total = connection.execute("SELECT COUNT(*) AS count FROM admin_audit_logs").fetchone()["count"]
    return {"items": [
        {"id": row["id"], "action": row["action"], "details": store.json_load(row["details_json"], {}),
         "created_at": row["created_at"], "actor_user_id": row["actor_user_id"], "target_user_id": row["target_user_id"],
         "actor_email": row["actor_email"], "target_email": row["target_email"]}
        for row in rows
    ], "total": total}
