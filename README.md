# AgriVision field planner

A Next.js field-planning interface backed by the supplied dataset-free ONNX inference adapter. The API process loads AgriVisionONNX() once and reuses its session; crop and fertilizer predictions use the ONNX model, while climate scenarios use compact state-climate summaries. No CSV is read at request time.

## Run locally

Requirements: Node.js 20+, Python 3.10+.

```powershell
npm install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.app:app --reload --port 8000
```

The web interface uses Tailwind CSS 4, Chart.js (`chart.js` + `react-chartjs-2`), jsPDF with AutoTable for report generation, and PDF.js to render the generated report in the browser before printing. The PDF.js worker is served from `public/pdf.worker.min.mjs`. District choices are bundled from the MIT-licensed directory listed in `src/data/README.md` and filtered immediately when a state is selected.

In a second terminal, run `npm run dev` and open `http://localhost:3000`. The browser calls the API at `http://localhost:8000`; set `NEXT_PUBLIC_AGRIVISION_API` at build time to use another API origin. The Python API allows that origin through `FRONTEND_ORIGIN` (defaults to localhost:3000). Copy `.env.example` to `.env` to configure account storage, browser sessions, and optional email delivery.

The ONNX model and compact metadata are in `model/`. The source adapter is in `backend/agrivision_onnx_inference.py`. `GET /api/metadata` serves only states, crop labels, fertilizer product names, and short model/climate metrics. `POST /api/predict` remains public and retains its request/response contract. Guest requests return the report without account persistence; signed-in requests also save the report to the user's history.

The site supports guest planning and optional accounts at `/signup` and `/login`. Account recovery uses `/forgot-password` and `/reset-password`. To bootstrap the first administrator on a persistent database, run `python -m backend.management create-admin`; deployments can also use the one-time `AGRIVISION_BOOTSTRAP_ADMIN_ID`, `AGRIVISION_BOOTSTRAP_ADMIN_EMAIL`, and `AGRIVISION_BOOTSTRAP_ADMIN_PASSWORD` variables. The app clears these values from the running process after the first superadmin is created. The protected admin workspace is at `/admin/login`. Membership is manually reviewed. Billing is not configured. See [accounts and operations](docs/accounts-and-operations.md) for cookie, secret, SMTP, backup, and database deployment guidance.

For a no-cost public preview, the Next.js pages can be exported and served by the FastAPI app on one origin. See the [PythonAnywhere deployment guide](deployment/pythonanywhere.md); its free tier is intended for small experiments and has renewal and resource limits.

## Vercel deployment

Vercel builds the static Next.js export and serves it from its CDN; `/api/*` requests run through the FastAPI/ONNX ASGI function. The `vercel.json` configuration bundles the exported pages, model, and migrations. Configure `APP_ENV=production`, a private `AGRIVISION_SECRET` (at least 32 bytes), `AGRIVISION_COOKIE_SECURE=true`, and `FRONTEND_ORIGIN`/`APP_PUBLIC_URL` to the public web origin. Set `DATABASE_URL` to the Neon pooled PostgreSQL connection string for durable accounts and report history. Locally, SQLite remains the default; Vercel must have `DATABASE_URL` set because its filesystem is ephemeral. The app applies the included PostgreSQL migration at startup.

## Model interpretation

- Yield holdout MAE is **1.19 t/ha**, a prototype metric, not a guarantee of field accuracy.
- Production costs are absent from source data. Profit values remain unavailable until a production cost per hectare is supplied.
- The report explains its top pick: with supplied production costs it uses risk-adjusted expected profit; otherwise it ranks by modeled gross revenue (yield estimate multiplied by embedded crop price), since profit cannot be calculated.
- Downside risk is a crop-specific composite score from 1 to 100 using state climate exposure and the yield model's relative holdout MAE; with production costs it also incorporates simulated financial loss. It is a planning index, not a calibrated probability.
- Climate charts show state-level seasonal estimates based on historical means and deterministic, reproducible simulated year-to-year variability scaled by state historical standard deviations. The package does not include raw annual climate records, so these values are illustrative scenarios, not observed weather or a calibrated forecast. No long-term trend is extrapolated without yearly data.
- Embedded mandi prices are historical snapshots. Refresh metadata before production use.
- Suitability probabilities are not calibrated. District is captured in the report, while climate and crop context are state-level.
- Fertilizer estimates need confirmation against a current local soil test, state guidance, and product labels. Climate estimates are annual and are not official weather warnings.
- An empty fertilizer product map means the optimizer selected zero rates under the submitted inputs and objective. It is not a finding that the crop needs no nutrients; confirm recommendations with a current soil test and local agronomic guidance.
