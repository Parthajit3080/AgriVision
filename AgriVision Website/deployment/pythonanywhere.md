# Free PythonAnywhere preview deployment

This setup serves the static Next.js export and FastAPI/ONNX API from one PythonAnywhere ASGI site. Keeping them on one origin lets account cookies and the SQLite account database work without a cross-site proxy or paid database.

## Build and prepare the files

From the project root, build the static site and package the runtime files:

```powershell
node .\node_modules\next\dist\bin\next build
powershell -ExecutionPolicy Bypass -File deployment/package-pythonanywhere.ps1
```

The bundle is created at `deployment/AgriVision-free-preview.zip`. It contains the static site, API, model, and migrations; it excludes local databases, `.env` files, dependencies, and source-control data.

## Configure the PythonAnywhere account

1. Create a free PythonAnywhere account and create an API token on its Account page. Keep the token private.
2. From the project root, run `python deployment/deploy_pythonanywhere.py --username YOUR_USERNAME --region us`. Use `eu` for an EU account. The script asks for the token in a hidden terminal prompt, uploads the bundle and startup script, and creates the free ASGI site. It refuses to replace a site that already exists on that domain.
3. Open the printed `https://<username>.pythonanywhere.com` URL (or the EU equivalent). On its first request, the startup script unpacks the bundle, creates the production secret and SQLite database path, installs Python requirements, then starts the site. The first load can take several minutes while packages install.
4. Check `/api/health` on the public site. A ready response confirms the API started; the root URL serves the planner.

PythonAnywhere's ASGI deployment is experimental and has no static-file mapping support, so this app serves its exported `out/` folder through FastAPI itself. Its free account provides one worker and 512 MiB of storage, and new free web apps expire after one month unless renewed. The SQLite database lives on the account filesystem, so back it up before any account reset or migration. Password-reset email is not configured by this free preview setup.

See [PythonAnywhere's ASGI deployment guide](https://help.pythonanywhere.com/pages/ASGICommandLine/) and [free-account limits](https://help.pythonanywhere.com/pages/FreeAccountsFeatures/) for current platform details.
