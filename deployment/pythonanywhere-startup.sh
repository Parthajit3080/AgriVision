#!/usr/bin/env bash
set -euo pipefail

app_dir="${HOME}/agrivision"
archive_path="${HOME}/AgriVision-free-preview.zip"
venv_path="${HOME}/.virtualenvs/agrivision-free-preview"
site_domain="${1:?Pass the PythonAnywhere site domain as the first argument}"
account_name="${HOME##*/}"

case "$site_domain" in
  "${account_name}.pythonanywhere.com"|"${account_name}.eu.pythonanywhere.com") ;;
  *) echo "Unexpected PythonAnywhere domain." >&2; exit 2 ;;
esac

if [ ! -f "${app_dir}/asgi.py" ]; then
  mkdir -p "$app_dir"
  unzip -o "$archive_path" -d "$app_dir"
fi

if [ ! -f "${app_dir}/.env.production" ]; then
  secret="$(python3.11 -c 'import secrets; print(secrets.token_urlsafe(48))')"
  site_origin="https://${site_domain}"
  mkdir -p "${app_dir}/data"
  cat > "${app_dir}/.env.production" <<EOF
APP_ENV=production
AGRIVISION_SECRET=${secret}
AGRIVISION_DB_PATH=${app_dir}/data/agrivision.sqlite3
AGRIVISION_COOKIE_SECURE=true
AGRIVISION_COOKIE_SAMESITE=lax
FRONTEND_ORIGIN=${site_origin}
APP_PUBLIC_URL=${site_origin}
EOF
  chmod 600 "${app_dir}/.env.production"
fi

mkdir -p "${HOME}/.virtualenvs"
if [ ! -f "${venv_path}/.agrivision-dependencies-installed" ]; then
  python3.11 -m venv "$venv_path"
  "${venv_path}/bin/python" -m pip install --disable-pip-version-check --no-cache-dir -r "${app_dir}/requirements.txt"
  touch "${venv_path}/.agrivision-dependencies-installed"
fi

: "${DOMAIN_SOCKET:?PythonAnywhere did not provide DOMAIN_SOCKET}"
exec "${venv_path}/bin/uvicorn" --app-dir "$app_dir" --uds "$DOMAIN_SOCKET" asgi:application
