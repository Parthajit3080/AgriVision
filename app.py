"""Vercel FastAPI entry point with a visible startup failure response."""
import json
import logging

try:
    from backend.app import app
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

