"""Upload and create the free PythonAnywhere ASGI preview without storing the API token."""
from __future__ import annotations

import argparse
import getpass
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "deployment" / "AgriVision-free-preview.zip"
STARTUP = ROOT / "deployment" / "pythonanywhere-startup.sh"


def call_api(url: str, token: str, *, method: str = "GET", body: bytes | None = None,
             content_type: str = "application/json") -> tuple[int, bytes]:
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": content_type,
            "User-Agent": "AgriVision-Free-Preview-Deploy/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"PythonAnywhere API returned HTTP {error.code}: {details}") from None
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach the PythonAnywhere API: {error.reason}") from None


def upload_file(api_root: str, username: str, token: str, remote_path: str, local_path: Path) -> None:
    boundary = "----AgriVisionDeployBoundary7MA4YWxkTrZu0gW"
    filename = local_path.name
    content_type = "application/zip" if local_path.suffix == ".zip" else "text/plain"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="content"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8") + local_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode("ascii")
    endpoint = f"{api_root}/api/v0/user/{username}/files/path{remote_path}"
    call_api(endpoint, token, method="POST", body=body, content_type=f"multipart/form-data; boundary={boundary}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True, help="PythonAnywhere account username")
    parser.add_argument("--region", required=True, choices=("us", "eu"), help="Account region")
    args = parser.parse_args()

    username = args.username.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", username):
        parser.error("username may contain only letters, numbers, underscores, and hyphens")
    if not ARCHIVE.is_file() or not STARTUP.is_file():
        parser.error("the deployment archive is missing; run deployment/package-pythonanywhere.ps1 first")

    api_host = "www.pythonanywhere.com" if args.region == "us" else "eu.pythonanywhere.com"
    domain_suffix = "pythonanywhere.com" if args.region == "us" else "eu.pythonanywhere.com"
    api_root = f"https://{api_host}"
    domain = f"{username}.{domain_suffix}"
    api_token = getpass.getpass("PythonAnywhere API token (hidden): ").strip()
    if not api_token:
        parser.error("an API token is required")

    account_url = f"{api_root}/api/v0/user/{urllib.parse.quote(username, safe='')}/cpu/"
    call_api(account_url, api_token)

    list_url = f"{api_root}/api/v1/user/{urllib.parse.quote(username, safe='')}/websites/"
    _, current = call_api(list_url, api_token)
    websites = json.loads(current.decode("utf-8"))
    if any(item.get("domain_name", "").lower() == domain.lower() for item in websites):
        raise RuntimeError(f"{domain} already has a website. No files were uploaded and it was left unchanged.")

    print("Uploading the app bundle and launch script...")
    upload_file(api_root, username, api_token, f"/home/{username}/AgriVision-free-preview.zip", ARCHIVE)
    upload_file(api_root, username, api_token, f"/home/{username}/agrivision-startup.sh", STARTUP)

    command = f"/bin/bash /home/{username}/agrivision-startup.sh {domain}"
    payload = json.dumps({
        "domain_name": domain,
        "enabled": True,
        "webapp": {"command": command},
    }).encode("utf-8")
    create_url = f"{api_root}/api/v1/user/{urllib.parse.quote(username, safe='')}/websites/"
    call_api(create_url, api_token, method="POST", body=payload)

    print(f"Created the public site: https://{domain}")
    print("On the first request, the app extracts its files and installs Python dependencies.")
    print("The first load can take several minutes. Then check /api/health for status.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nDeployment cancelled.", file=sys.stderr)
        raise SystemExit(130)
    except RuntimeError as error:
        print(f"Deployment stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
