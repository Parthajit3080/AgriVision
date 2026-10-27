"""PythonAnywhere ASGI entry point."""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env.production", override=False)

from backend.app import app  # noqa: E402

# Vercel's Python runtime looks for `app`; PythonAnywhere's ASGI configuration
# continues to use `application`.
application = app
