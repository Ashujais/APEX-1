# Post-readiness audit

Audit date: 2026-08-30

Capability labels in this report mean:

- PASS: the stated command or flow actually ran successfully.
- FAIL: the flow ran and failed.
- BLOCKED: the required external runtime/tool is absent, or no test facility exists.
- READY: preparation is validated, while execution still requires named external resources.

## Final readiness table

| Area | Result | Evidence |
|---|---|---|
| Frontend startup | PASS | scripts/windows/dev.ps1 served HTTP 200 on localhost:3013. |
| Backend startup | PASS | The same command started Uvicorn; /health returned ok. |
| Database | PASS | SQLite schema/startup and SELECT 1 readiness passed; API tests use isolated real SQLite files. |
| Authentication | PASS | Registration, verification, login, refresh rotation, logout, reset, and ownership tests passed. |
| Chat | PASS | Authenticated conversation, SSE stream, persistence, and tenant isolation tests passed. |
| Files | PASS | Streaming upload, validation, size limit, metadata, download, auth, and ownership tests passed. |
| RAG | PASS | tiny-corpus upload, extraction, chunking, embedding, indexing, retrieval, reranking, context, and citations passed end to end. |
| MCP registry | PASS | Authenticated built-in registration/discovery/schema routes and current protocol client fixture passed. |
| Remote external MCP | BLOCKED | No external MCP server or credential is configured on this machine. |
| Tools | PASS | Permission checks, JSON validation, execution, timeout/retry controls, tenant isolation, RAG tool, and metadata-only audits passed. |
| Function-call architecture | PASS | Provider-independent structured call/result loop and maximum-call budget tests passed. |
| Redis, local without | PASS | /ready reports not_configured and remains ready because Redis is optional. |
| Redis, local with | BLOCKED | Port 6379 is closed and Docker Desktop's daemon is not running. Required/unavailable readiness behavior is tested. |
| Local CPU model | PASS | Real tiny model initialization, tokenizer, dataset, forward, loss, backward, parameter update, optimizer, scheduler, checkpoint, resume, and evaluation passed. |
| Trained APEX model | BLOCKED | No trained APEX model exists; apex-dev remains a deterministic platform-test responder. |
| Hardware | PASS | Windows hardware command measured i3-7020U, 3.79 GiB RAM, CPU PyTorch, and no CUDA. |
| Kaggle APEX-100M preparation | READY | Explicit 100,092,672-parameter config, notebook, provider, CUDA/VRAM advice, artifacts, checkpoint/resume, and preflight paths are validated. |
| Kaggle execution | BLOCKED | This host is not Kaggle, has no CUDA GPU, and no licensed Kaggle dataset was supplied. |
| Python tests | PASS | 27 passed in 80.58 seconds. |
| Python lint | PASS | ruff check . completed with zero findings. |
| npm ci | PASS | Node 24.19.0; 558 packages installed from apps/web/package-lock.json. |
| npm lint | PASS | oxlint completed with zero errors. |
| npm build | PASS | vinext production build completed all five stages. |
| Frontend unit tests | BLOCKED | apps/web defines no unit-test script or test runner. |
| pnpm absence | PASS | No pnpm-lock.yaml or pnpm-workspace.yaml exists. |
| npm lock architecture | PASS | apps/web is standalone npm with package-lock.json; see NPM_ARCHITECTURE.md. |
| GNU Make commands on this host | BLOCKED | GNU Make is not installed on Windows. The Makefile remains for Linux/CI. |
| Native Windows dev/hardware/train/evaluate | PASS | All four commands actually ran; train was limited to one tiny CPU step. |
| Native Windows test | PASS | One command completed 27 Python tests, Ruff, npm ci, oxlint, and the five-stage production build. |

## Test inventory

The 27-test Python suite covers API health, auth, chat, files, RAG, MCP/tools, Redis readiness,
function calling, model, training/resume/evaluation, tokenizer, registry, and hardware/provider
behavior. The live Windows development command additionally verified real frontend and backend
listeners together.

No large model was instantiated or trained. The one-step local result is pipeline evidence only,
not a benchmark or quality claim.

## Remaining external or dependency work

- Start Docker Desktop or a real Redis service to verify local-with-Redis connectivity.
- Run scripts/kaggle/preflight.py inside Kaggle with a GPU and licensed dataset, then run only
  the documented tiny smoke before authorizing APEX-100M training.
- Add a frontend unit-test runner if component-level browser/unit coverage is required.
- Review the eleven npm advisories through a separately scoped dependency-upgrade change.
- Configure a real allowlisted MCP server to verify remote interoperability.
