"""SQLite/Postgres persistence and security primitives for account features."""
from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    import psycopg
    from psycopg.rows import dict_row
    DATABASE_ERRORS: tuple[type[BaseException], ...] = (sqlite3.Error, psycopg.Error)
except ImportError:  # Keep SQLite-only local environments usable without psycopg.
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]
    DATABASE_ERRORS = (sqlite3.Error,)

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
PASSWORD_ITERATIONS = 600_000
SESSION_COOKIE_NAME = "agrivision_session"
CSRF_COOKIE_NAME = "agrivision_csrf"
SESSION_TTL_HOURS = 24 * 7
EMAIL_RE = re.compile(r"^[^\s@]{1,64}@[^\s@.]+(?:\.[^\s@.]+)+$")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def timestamp(value: datetime | None = None) -> str:
    return (value or utc_now()).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def get_database_path() -> Path:
    return Path(os.getenv("AGRIVISION_DB_PATH", str(ROOT / "data" / "agrivision.sqlite3"))).expanduser().resolve()


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", os.getenv("AGRIVISION_DATABASE_URL", "")).strip()


def app_secret() -> bytes:
    secret = os.getenv("AGRIVISION_SECRET", "")
    if not secret and os.getenv("APP_ENV", "development").lower() in {"production", "prod"}:
        raise RuntimeError("AGRIVISION_SECRET must be configured in production.")
    if not secret:
        secret = "local-development-only-change-before-deployment-" + str(ROOT)
    if len(secret.encode("utf-8")) < 32:
        raise RuntimeError("AGRIVISION_SECRET must contain at least 32 bytes.")
    return secret.encode("utf-8")


def keyed_hash(value: str) -> str:
    return hmac.new(app_secret(), value.encode("utf-8"), hashlib.sha256).hexdigest()


def hash_password(password: str) -> str:
    if not isinstance(password, str) or not password:
        raise ValueError("Password must not be empty.")
    if len(password) > 128:
        raise ValueError("Password must be 128 characters or fewer.")
    if len(password.encode("utf-8")) > 512:
        raise ValueError("Password is too long.")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    import base64
    return "pbkdf2_sha256${}${}${}".format(
        PASSWORD_ITERATIONS,
        base64.urlsafe_b64encode(salt).decode("ascii").rstrip("="),
        base64.urlsafe_b64encode(digest).decode("ascii").rstrip("="),
    )


def verify_password(password: str, encoded: str) -> bool:
    import base64
    try:
        algorithm, rounds_text, salt_text, digest_text = encoded.split("$", 3)
        rounds = int(rounds_text)
        if algorithm != "pbkdf2_sha256" or rounds < 100_000 or rounds > 1_000_000:
            return False
        salt = base64.urlsafe_b64decode(salt_text + "=" * (-len(salt_text) % 4))
        expected = base64.urlsafe_b64decode(digest_text + "=" * (-len(digest_text) % 4))
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
        return hmac.compare_digest(actual, expected)
    except (AttributeError, TypeError, ValueError, UnicodeError):
        return False


def dummy_password_hash() -> str:
    # Used for nonexistent accounts so lookup failures still pay the normal hash cost.
    return hash_password(secrets.token_urlsafe(24))


def validate_email(email: str) -> str:
    normalized = email.strip().lower()
    if len(normalized) > 254 or not EMAIL_RE.fullmatch(normalized):
        raise ValueError("Enter a valid email address.")
    return normalized


class _PostgresConnection:
    """Small compatibility layer so existing account queries keep SQLite placeholders."""

    def __init__(self, connection: Any):
        self._connection = connection

    def execute(self, query: str, params: Any = ()) -> Any:
        if query.strip().upper() == "BEGIN IMMEDIATE":
            query = "BEGIN"
        return self._connection.execute(query.replace("?", "%s"), params)

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def close(self) -> None:
        self._connection.close()


def _connect(path: Path | None = None) -> Any:
    database_url = get_database_url() if path is None else ""
    if database_url:
        if psycopg is None:
            raise RuntimeError("psycopg is required when DATABASE_URL is configured.")
        connection = psycopg.connect(database_url, row_factory=dict_row, connect_timeout=10)
        return _PostgresConnection(connection)

    database = path or get_database_path()
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, timeout=10, isolation_level="DEFERRED")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


@contextlib.contextmanager
def db_session(path: Path | None = None) -> Iterator[Any]:
    connection = _connect(path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(path: Path | None = None) -> None:
    migration_dir = BASE / "migrations"
    connection = _connect(path)
    try:
        is_postgres = isinstance(connection, _PostgresConnection)
        if is_postgres:
            connection.execute("SELECT pg_advisory_xact_lock(1726, 1)")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)"
        )
        migration = migration_dir / (
            "001_identity_and_membership.postgres.sql" if is_postgres else "001_identity_and_membership.up.sql"
        )
        if not migration.exists():
            raise RuntimeError(f"Required migration is missing: {migration.name}")
        applied = connection.execute("SELECT 1 FROM schema_migrations WHERE version = 1").fetchone()
        if not applied:
            migration_sql = migration.read_text(encoding="utf-8")
            if is_postgres:
                for statement in migration_sql.split(";"):
                    if statement.strip():
                        connection.execute(statement)
                connection.execute(
                    "CREATE OR REPLACE FUNCTION prevent_admin_audit_mutation() "
                    "RETURNS trigger LANGUAGE plpgsql AS $agri$ BEGIN "
                    "RAISE EXCEPTION 'administrative audit entries are immutable'; END; $agri$"
                )
                connection.execute("DROP TRIGGER IF EXISTS admin_audit_no_update ON admin_audit_logs")
                connection.execute(
                    "CREATE TRIGGER admin_audit_no_update BEFORE UPDATE ON admin_audit_logs "
                    "FOR EACH ROW EXECUTE FUNCTION prevent_admin_audit_mutation()"
                )
                connection.execute("DROP TRIGGER IF EXISTS admin_audit_no_delete ON admin_audit_logs")
                connection.execute(
                    "CREATE TRIGGER admin_audit_no_delete BEFORE DELETE ON admin_audit_logs "
                    "FOR EACH ROW EXECUTE FUNCTION prevent_admin_audit_mutation()"
                )
                connection.execute(
                    "INSERT INTO schema_migrations(version, name, applied_at) VALUES (?, ?, ?)",
                    (1, "identity_and_membership", timestamp()),
                )
            else:
                connection.executescript(migration_sql)
        retention_cutoff = timestamp(utc_now() - timedelta(days=90))
        connection.execute(
            "DELETE FROM sessions WHERE (expires_at < ? OR revoked_at < ?) AND created_at < ?",
            (timestamp(), retention_cutoff, retention_cutoff),
        )
        connection.execute(
            "DELETE FROM password_reset_tokens WHERE expires_at < ? OR consumed_at < ?",
            (retention_cutoff, retention_cutoff),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def json_load(value: str, fallback: Any = None) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def public_user(user: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    row = dict(user)
    return {
        "id": row["id"],
        "email": row["email"],
        "username": row.get("username"),
        "display_name": row["display_name"],
        "role": row["role"],
        "account_status": row["account_status"],
        "membership_status": row["membership_status"],
        "premium_since": row.get("premium_since"),
        "created_at": row["created_at"],
        "last_login_at": row.get("last_login_at"),
        "preferences": json_load(row.get("saved_preferences_json", "{}"), {}),
    }


def session_cookie_secure() -> bool:
    configured = os.getenv("AGRIVISION_COOKIE_SECURE")
    if configured is not None:
        return configured.strip().lower() in {"1", "true", "yes"}
    return os.getenv("APP_ENV", "development").lower() in {"production", "prod"}


def session_ttl() -> timedelta:
    raw = os.getenv("AGRIVISION_SESSION_TTL_HOURS", str(SESSION_TTL_HOURS))
    try:
        hours = min(max(int(raw), 1), 24 * 30)
    except ValueError:
        hours = SESSION_TTL_HOURS
    return timedelta(hours=hours)


def create_session(connection: Any, user_id: int, request: Any) -> tuple[str, int, str]:
    token = secrets.token_urlsafe(32)
    now = utc_now()
    expires = now + session_ttl()
    ip = request.client.host if request.client else "unknown"
    agent = request.headers.get("user-agent", "")[:300]
    cursor = connection.execute(
        "INSERT INTO sessions(user_id, token_hash, created_at, expires_at, last_seen_at, ip_hash, user_agent) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING id",
        (user_id, hashlib.sha256(token.encode()).hexdigest(), timestamp(now), timestamp(expires), timestamp(now), keyed_hash(ip), agent),
    )
    connection.execute("UPDATE users SET last_login_at = ?, updated_at = ? WHERE id = ?", (timestamp(now), timestamp(now), user_id))
    return token, int(cursor.fetchone()["id"]), timestamp(expires)


def lookup_session(connection: Any, token: str | None, *, touch: bool = True) -> tuple[Any, Any] | None:
    if not token or len(token) > 256:
        return None
    digest = hashlib.sha256(token.encode()).hexdigest()
    session = connection.execute(
        "SELECT * FROM sessions WHERE token_hash = ? AND revoked_at IS NULL AND expires_at > ?",
        (digest, timestamp()),
    ).fetchone()
    if not session:
        return None
    user = connection.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    if not user or user["account_status"] != "active":
        connection.execute("UPDATE sessions SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL", (timestamp(), session["id"]))
        return None
    if touch:
        last_seen = parse_timestamp(session["last_seen_at"])
        if (utc_now() - last_seen).total_seconds() > 300:
            connection.execute("UPDATE sessions SET last_seen_at = ? WHERE id = ?", (timestamp(), session["id"]))
    return session, user


def enforce_rate_limit(
    connection: Any,
    *,
    scope: str,
    bucket: str,
    limit: int,
    window_seconds: int,
) -> None:
    now = utc_now()
    cutoff = timestamp(now - timedelta(seconds=window_seconds))
    connection.execute("DELETE FROM auth_attempts WHERE attempted_at < ?", (timestamp(now - timedelta(days=1)),))
    key = keyed_hash(f"{scope}:{bucket.lower()}")
    count = connection.execute(
        "SELECT COUNT(*) AS total FROM auth_attempts WHERE scope = ? AND key_hash = ? AND attempted_at >= ?",
        (scope, key, cutoff),
    ).fetchone()["total"]
    if count >= limit:
        from fastapi import HTTPException
        raise HTTPException(status_code=429, detail="Too many attempts. Try again later.", headers={"Retry-After": str(window_seconds)})
    connection.execute("INSERT INTO auth_attempts(scope, key_hash, attempted_at) VALUES (?, ?, ?)", (scope, key, timestamp(now)))


def request_ip_hash(request: Any) -> str | None:
    if not request.client:
        return None
    return keyed_hash(request.client.host)


def add_activity(connection: Any, user_id: int, event_type: str, details: dict[str, Any] | None = None) -> None:
    connection.execute(
        "INSERT INTO activity_events(user_id, event_type, details_json, created_at) VALUES (?, ?, ?, ?)",
        (user_id, event_type, json_dump(details or {}), timestamp()),
    )


def add_audit(
    connection: Any,
    actor_id: int,
    action: str,
    target_id: int | None,
    request: Any,
    details: dict[str, Any] | None = None,
) -> None:
    connection.execute(
        "INSERT INTO admin_audit_logs(actor_user_id, target_user_id, action, details_json, ip_hash, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (actor_id, target_id, action, json_dump(details or {}), request_ip_hash(request), timestamp()),
    )


def bootstrap_initial_admin() -> bool:
    """Create the configured first superadmin once, then forget the password in this process."""
    if not get_database_url():
        return False
    username = os.getenv("AGRIVISION_BOOTSTRAP_ADMIN_ID", "").strip()
    email = os.getenv("AGRIVISION_BOOTSTRAP_ADMIN_EMAIL", "").strip()
    password = os.getenv("AGRIVISION_BOOTSTRAP_ADMIN_PASSWORD", "")
    if not (username and email and password):
        return False

    if not re.fullmatch(r"[A-Za-z0-9_-]{3,40}", username):
        raise RuntimeError("AGRIVISION_BOOTSTRAP_ADMIN_ID is invalid.")
    email = validate_email(email)
    if len(password) > 128 or len(password.encode("utf-8")) > 512:
        raise RuntimeError("AGRIVISION_BOOTSTRAP_ADMIN_PASSWORD is too long.")

    with db_session() as connection:
        if isinstance(connection, _PostgresConnection):
            connection.execute("SELECT pg_advisory_xact_lock(1726, 2)")
        existing = connection.execute(
            "SELECT 1 FROM users WHERE role IN ('admin', 'superadmin') LIMIT 1"
        ).fetchone()
        if existing:
            _forget_bootstrap_admin_environment()
            return False
        now = timestamp()
        cursor = connection.execute(
            "INSERT INTO users(email, username, display_name, password_hash, role, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'superadmin', ?, ?) RETURNING id",
            (email, username, "AgriVision Superadmin", hash_password(password), now, now),
        )
        user_id = int(cursor.fetchone()["id"])
        add_activity(connection, user_id, "initial_superadmin_created")
    _forget_bootstrap_admin_environment()
    return True


def _forget_bootstrap_admin_environment() -> None:
    for key in (
        "AGRIVISION_BOOTSTRAP_ADMIN_ID",
        "AGRIVISION_BOOTSTRAP_ADMIN_EMAIL",
        "AGRIVISION_BOOTSTRAP_ADMIN_PASSWORD",
    ):
        os.environ.pop(key, None)


_DUMMY_HASH = dummy_password_hash()

