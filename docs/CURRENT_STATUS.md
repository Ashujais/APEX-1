# Current status

## Implemented

- Phase 0 hardware/toolchain inventory and feasibility recommendation.
- React/TypeScript workspace shell with responsive APEX-1 design tokens.
- FastAPI application foundation, health/readiness endpoints, and request IDs.
- Local email/password registration, verification, login, refresh rotation, logout, and Argon2id password hashing.
- Tenant-scoped conversation creation/history and authenticated server-sent-event chat streaming.
- Deterministic development responder clearly identified as non-model behavior.
- Streaming, size-limited local file upload with validation, metadata, checksums, ownership,
  conversation/project association, status, listing, and authenticated retrieval.
- Tenant-scoped local RAG ingestion for text, Markdown, CSV, JSON, PDF, DOCX, XLSX, and source
  files, with cleaning, chunking, deterministic feature-hash embeddings, SQLite indexing,
  vector/lexical reranking, context assembly, and citations.
- Authenticated built-in MCP registry, tool discovery, JSON schemas, invocation permissions,
  bounded timeouts, metadata-only audit logs, and tenant ownership enforcement.
- Provider-independent structured function-call loop with call and round budgets.
- Optional Redis readiness reporting; Redis is not required for the default local SQLite mode.
- Native Windows dev, test, hardware, tiny-train, and evaluation commands under scripts/windows.
- Byte-level BPE tokenizer training, serialization, encoding, and decoding.
- Configurable decoder-only PyTorch Transformer with RMSNorm, RoPE, SwiGLU, grouped-query attention, causal attention, weight tying, KV cache, and checkpoint helpers.
- Measurement-based hardware inventory through script, CLI, and Make target.

## Experimental

- Tiny CPU-only Transformer forward/training/inference smoke path.
- SQLite local persistence; PostgreSQL is the intended production database.
- Model and provider routing interfaces.
- Remote MCP 2026-07-28 Streamable HTTP discovery/invocation. The protocol client is tested
  with bounded fixtures; a real external MCP server is not configured on this machine.
- Built-in tool execution and provider-independent function-calling interfaces. The current
  deterministic development responder does not emit tool calls.
- Conservative `HardwareAdvisor` and local CPU/local GPU/Kaggle compute-provider contracts.
- Real tiny-model training, warmup/linear/cosine scheduling, gradient accumulation, resumable checkpoint metadata, evaluation loss, and PyTorch generation timing.
- Atomic local model/dataset/experiment registries with lifecycle validation and SHA-256 artifact records.
- Kaggle setup/preflight/tiny-train/evaluate scripts, APEX-100M configuration, runtime
  validation, CUDA/VRAM selection tests, documentation, and notebook; execution still requires
  an actual Kaggle GPU session and a licensed dataset.

## Planned

- Email provider integration, magic links, OTP, OAuth/OIDC, MFA, passkeys,
  organizations/RBAC, background document workers, agents, API keys, billing, admin, SDKs,
  observability, and full public documentation routes.
- PostgreSQL/object-storage shared registries, training job orchestration, dataset streaming/quality pipeline, model promotion, and production inference adapters.

## Requires GPU cluster

- Development-model pretraining at 100M–350M scale, distributed training, large-model evaluation, and production inference.

## Requires datasets

- Pretraining, SFT, preference optimization, tokenizer quality evaluation, model benchmarks, and safety evaluation.

## Requires external provider

- Transactional email, social OAuth, payment processing, managed object storage, and optional external model/image/audio/video providers.

## Known limitations

- The development responder is not an LLM and does not claim intelligence.
- The integrated GPU is not suitable for the planned PyTorch training workload.
- Docker configuration is unverified until the local Docker daemon is started.
- Redis-backed production features are not implemented; Redis currently participates only in
  explicit readiness checks when configured.
- The frontend has no separate unit-test runner; clean install, lint, production build, and
  live HTTP startup are the verified frontend gates.
- npm reports eleven dependency advisories and five extraneous optional WASM runtime bundles.
  The WASM packages are recreated by npm ci from optional Tailwind/Rolldown fallback bundles;
  they are not direct dependencies or a package-manager switch.
- Starlette emits a deprecation warning for its current TestClient httpx integration.
- Kaggle scripts are not locally presented as GPU-verified; Kaggle execution remains `REQUIRES_KAGGLE`.
- Local filesystem registries are single-process research storage, not a production control plane.

## Known bugs

- None confirmed in the verified first milestone.
