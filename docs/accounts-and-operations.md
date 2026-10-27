# AgriVision accounts and operations

## Local setup

Copy `.env.example` to `.env`. In a development shell, run the FastAPI service and Next.js app as described in the repository README. The API applies the additive identity schema on startup at `AGRIVISION_DB_PATH` (defaults to `data/agrivision.sqlite3`). Keep this file and its `-wal`/`-shm` companions out of source control.

Create the first administrator interactively after configuring the API environment:

```powershell
python -m backend.management create-admin
```

The command prompts for an admin ID, email, display name, and password. It does not accept credentials on the command line or print the password. Passwords must be nonempty and are capped at 128 characters. Additional administrators are granted by an authenticated superadmin in the admin console.

## Production configuration

- Set `APP_ENV=production` and provide `AGRIVISION_SECRET` from a secret manager. Use at least 32 cryptographically random bytes and keep the value stable across API restarts. The service validates the secret on startup.
- Serve the web app and API over HTTPS. Set `AGRIVISION_COOKIE_SECURE=true`. Keep `AGRIVISION_COOKIE_SAMESITE=lax` when both origins are same-site; use `none` only for a deliberately cross-site setup, which also requires secure cookies.
- Set `FRONTEND_ORIGIN` to a comma-separated allowlist of exact browser origins. The API uses credentialed CORS and verifies the Origin plus a signed CSRF token for state-changing account and admin requests.
- Set `NEXT_PUBLIC_AGRIVISION_API` at Next.js build time and `APP_PUBLIC_URL` to the public web origin. Configure SMTP with `SMTP_HOST`, `SMTP_PORT`, `SMTP_FROM`, and optional username/password to deliver one-time password reset links. Without SMTP, reset URLs are exposed only while `APP_ENV` is development, dev, or test.
- Use a durable local disk for SQLite. This deployment is a single-host SQLite setup: SQLite WAL files must stay on the same local filesystem, and concurrent multi-host deployments should move the account store to a managed database with an equivalent migration before scaling out.
- Back up the database and test restoring it before upgrades. The `backend/migrations/001_identity_and_membership.down.sql` script is destructive; use it only when intentionally rolling back the account feature after a verified backup. No deployment pipeline or hosted database was present in the inspected project, so neither is modified here.

## Security and data behavior

Passwords are salted and hashed with PBKDF2-HMAC-SHA256 at 600,000 iterations. Browser sessions are random opaque values stored server-side by digest, expire, can be revoked, and are issued in HttpOnly cookies. Password reset tokens are single-use, expire after 30 minutes, and revoke existing sessions. Public prediction routes remain available to guests with their existing request/response contract. Predictions are saved only when a valid account session is present; standard users retain the latest 10 reports and Field Plus accounts retain up to 1,000. Accounts can update their display name, saved planner defaults, and active sessions.

Field Plus is manually reviewed and has no payment processor or billing workflow configured. It grants the higher saved-report limit, personal summaries, history filtering access, and premium account recognition. Admin and superadmin permissions are checked on the API for every protected operation. Membership decisions, account status changes, and role changes create append-only audit entries; user activity is separately recorded for account history.

## Tests

Install the test client dependency with `pip install -r requirements-dev.txt`, then run `python -m unittest discover -s tests`. Run `npm run build` to verify the frontend bundle and typed routes.
