# APEX-1

APEX-1 is a local-first AI platform and model-research foundation. This repository starts with a working React interface, FastAPI service, secure local authentication, tenant-scoped conversations, a transparent development responder, and an experimental decoder-only Transformer core.

No benchmark, training, voice, image, video, billing, or provider capability is implied unless it is listed as implemented in [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md).

## Local development

Requirements: Python 3.12+, Node 22.13+, and optionally Docker Desktop.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,model]"
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m uvicorn apex_api.main:app --reload --port 8000
```

In a second terminal:

```powershell
Set-Location apps/web
npm install
npm run dev
```

Open `http://localhost:3000`. API documentation is at `http://localhost:8000/docs`.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
Set-Location apps/web
npm run build
```

The Docker contracts are present, but this machine's Docker daemon must be running before `docker compose up` can be verified.
