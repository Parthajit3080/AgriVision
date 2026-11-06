"""Vercel ASGI entry point."""
import json
import logging
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlencode

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_RUNTIME_DEPS = PROJECT_ROOT / "model" / "runtime_deps"
if PYTHON_RUNTIME_DEPS.is_dir():
    sys.path.insert(0, str(PYTHON_RUNTIME_DEPS))

try:
    from backend.app import app as _fastapi_app

    async def app(scope, receive, send):
        if scope["type"] == "http":
            query_items = parse_qsl(scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True)
            forwarded_path = next((value for key, value in query_items if key == "_asgi_path"), None)
            if forwarded_path:
                scope = dict(scope)
                scope["path"] = forwarded_path
                scope["raw_path"] = forwarded_path.encode("utf-8")
                scope["root_path"] = ""
                scope["query_string"] = urlencode(
                    [(key, value) for key, value in query_items if key != "_asgi_path"]
                ).encode("latin-1")
        await _fastapi_app(scope, receive, send)

except Exception:
    logging.exception("AgriVision FastAPI failed during import.")

    async def app(scope, receive, send):
        if scope["type"] != "http":
            return
        body = json.dumps({"error": "AgriVision backend failed to start."}).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 500,
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"cache-control", b"no-store"),
            ],
        })
        await send({"type": "http.response.body", "body": body})

__all__ = ["app"]
