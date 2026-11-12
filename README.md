# AgriVision

A crop-planning web app for exploring crop, yield, fertilizer, and climate scenarios using a farmer’s location, season, land, and soil-test inputs. The frontend is built with Next.js; a FastAPI service runs ONNX inference and optional account features.

## Features

- Compare crop recommendations using modeled yield, embedded market prices, risk settings, and optional cost inputs.
- Explore state-level climate scenarios and fertilizer plans based on soil inputs.
- Generate and preview downloadable PDF reports.
- Use the planner as a guest, or create an account to save reports and preferences.
- Review membership requests and manage accounts through the protected admin workspace.

## Project layout

```text
.
├── backend/                  # FastAPI API, account services, and database migrations
├── deployment/               # PythonAnywhere deployment scripts and guide
├── docs/                     # Operations notes, diagrams, and model charts
├── model/                    # ONNX inference model and runtime metadata
├── public/                   # Browser assets, including the PDF.js worker
├── scripts/                  # Build and documentation helpers
├── src/                      # Next.js pages, components, and bundled location data
├── tests/                    # Python API and model tests
├── AgriVision Final Folder/  # Model export sources and source datasets
├── app.py                    # Vercel FastAPI entry point
├── asgi.py                   # ASGI entry point for other deployments
└── vercel.json               # Static export and API routing configuration
```

The API loads the ONNX model for inference; it does not read the source CSV files during prediction. The model and compact metadata used at runtime are in `model/`.

## Run locally

### Requirements

- Python **3.12** (see `.python-version`)
- Node.js **20 or newer**
- pnpm

### Start the API

In PowerShell, from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn backend.app:app --reload --port 8000 --env-file .env
```

### Start the website

In a second terminal, from the repository root:

```powershell
pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000). The development frontend uses the API at `http://localhost:8000` by default. Check the API at [`/api/health`](http://localhost:8000/api/health); interactive API documentation is available at [`/docs`](http://localhost:8000/docs).

To create an optimized static export, run:

```powershell
pnpm build
```

This runs the Next.js build and copies the export into `public/` for deployment.

## API overview

| Route | Purpose |
| --- | --- |
| `GET /api/health` | Check that the inference service and model are available. |
| `GET /api/metadata` | Load public crop, state, fertilizer, and model summary data. |
| `POST /api/predict` | Generate a field-planning report. Guest access is supported. |
| `/api/auth/*` | Account, session, preferences, and saved-report operations. |
| `/api/membership/*` | Submit and review Field Plus membership requests. |
| `/api/admin/*` | Protected account and platform administration. |

The website includes `/signup`, `/login`, `/forgot-password`, `/reset-password`, `/account`, and `/admin/login` pages. Membership is reviewed by an administrator; billing is not configured.

## Configuration

Copy `.env.example` to `.env` for local development. Keep real credentials in environment variables or a secret manager; do not commit `.env` files.

| Variable | Use |
| --- | --- |
| `AGRIVISION_SECRET` | Stable signing secret; use at least 32 bytes in production. |
| `AGRIVISION_DB_PATH` | Local SQLite file path. SQLite is the default for local development. |
| `DATABASE_URL` | PostgreSQL connection string for persistent account and report data. |
| `APP_ENV` | Set to `production` in production deployments. |
| `AGRIVISION_COOKIE_SECURE` | Set to `true` when serving over HTTPS. |
| `FRONTEND_ORIGIN` | Exact allowed browser origin or comma-separated origin allowlist. |
| `APP_PUBLIC_URL` | Public website origin used when creating account recovery links. |
| `NEXT_PUBLIC_AGRIVISION_API` | Optional API origin, set at frontend build time; leave unset for the same-origin deployment. |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM` | Optional SMTP configuration for password-reset email. |

For production account setup, database migrations, admin bootstrapping, cookie settings, and backups, see [Account and operations guide](docs/accounts-and-operations.md).

## Deploy

### Vercel

The included `vercel.json` builds the Next.js static export and routes `/api/*` requests to the FastAPI ASGI function. Configure these Vercel environment variables:

- `APP_ENV=production`
- `AGRIVISION_SECRET` with a private value of at least 32 bytes
- `AGRIVISION_COOKIE_SECURE=true`
- `FRONTEND_ORIGIN` and `APP_PUBLIC_URL` set to the deployed site origin
- `DATABASE_URL` set to a persistent PostgreSQL connection string for accounts and saved reports
- SMTP variables if password-reset emails are required

The Vercel filesystem is ephemeral, so use PostgreSQL for durable account and report storage. The API applies the included PostgreSQL migration at startup. After deployment, check `/api/health` on the public site.

### PythonAnywhere

See the [PythonAnywhere deployment guide](deployment/pythonanywhere.md) for the static export, ASGI setup, and free-preview limitations.

## Model notes and limitations

- The recorded yield evaluation reports a holdout MAE of **1.19 t/ha** and an R² of **0.936** on 3,904 rows. These are prototype metrics, not a guarantee of field accuracy.
- The app does not include a calibrated crop-suitability classifier; suitability probabilities are unavailable.
- Embedded mandi prices are historical snapshots. Refresh them before making current financial decisions.
- Climate charts show reproducible, state-level scenarios. They are not official weather forecasts, and district-level climate observations are not included.
- Profit estimates require production-cost inputs. Without them, the planner cannot determine net profit.
- Fertilizer rates are planning estimates. Confirm them against a current soil test, local agricultural guidance, and product labels before use.

