# APEX-1

APEX-1 is a local-first AI platform and model-research foundation. This repository starts with a working React interface, FastAPI service, secure local authentication, tenant-scoped conversations, a transparent development responder, and an experimental decoder-only Transformer core.

No benchmark, training, voice, image, video, billing, or provider capability is implied unless it is listed as implemented in [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md).

The requirement-by-requirement audit and milestone order are in [`docs/GAP_ANALYSIS.md`](docs/GAP_ANALYSIS.md). Kaggle and local training instructions are in [`docs/KAGGLE.md`](docs/KAGGLE.md) and [`docs/TRAINING.md`](docs/TRAINING.md).

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

On Windows, the equivalent combined command is:

```powershell
.\scripts\windows\dev.ps1
```

Other native commands:

```powershell
.\scripts\windows\test.ps1
.\scripts\windows\hardware.ps1
.\scripts\windows\train.ps1 -Steps 1
.\scripts\windows\evaluate.ps1
```

Linux/CI equivalents remain available through make dev, make test, make hardware, make train,
and make evaluate. GNU Make is not required for Windows operation.

## Verification

```powershell
.\.venv\Scripts\python.exe scripts\run_tests.py
.\.venv\Scripts\python.exe -m ruff check .
Set-Location apps/web
npm run build
```

The Docker contracts are present, but this machine's Docker daemon must be running before `docker compose up` can be verified.

## Redis modes

Default local development uses SQLite and does not require Redis. /ready reports Redis as
not_configured while the service remains ready.

To test with Redis, start Redis, set APEX_REDIS_URL=redis://localhost:6379/0, and optionally set
APEX_REDIS_REQUIRED=true. Docker Compose configures Redis as required for its production-like API
service and includes a real Redis health check. Connectivity is never fabricated.

File upload, local RAG, MCP/tool endpoints, and their capability boundaries are summarized in
docs/CURRENT_STATUS.md. npm ownership and the five optional WASM packages are documented in
docs/NPM_ARCHITECTURE.md.

## Research commands

```powershell
.\.venv\Scripts\python.exe -m apex.cli hardware
.\.venv\Scripts\python.exe -m apex.cli status
.\.venv\Scripts\python.exe -m apex.cli train-small --dataset examples/data/tiny-corpus.txt --output checkpoints/apex-tiny-run --steps 2 --dataset-license project-authored
```

The tiny training command performs real optimization and writes ignored local artifacts plus hash-backed registry records. It is an experimental pipeline check, not a model-quality claim.
