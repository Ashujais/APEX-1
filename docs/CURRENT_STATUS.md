# Current status

## Implemented

- Phase 0 hardware/toolchain inventory and feasibility recommendation.
- React/TypeScript workspace shell with responsive APEX-1 design tokens.
- FastAPI application foundation, health/readiness endpoints, and request IDs.
- Local email/password registration, verification, login, refresh rotation, logout, and Argon2id password hashing.
- Tenant-scoped conversation creation/history and authenticated server-sent-event chat streaming.
- Deterministic development responder clearly identified as non-model behavior.
- Byte-level BPE tokenizer training, serialization, encoding, and decoding.
- Configurable decoder-only PyTorch Transformer with RMSNorm, RoPE, SwiGLU, grouped-query attention, causal attention, weight tying, KV cache, and checkpoint helpers.
- Measurement-based hardware inventory through script, CLI, and Make target.

## Experimental

- Tiny CPU-only Transformer forward/training/inference smoke path.
- SQLite local persistence; PostgreSQL is the intended production database.
- Model and provider routing interfaces.
- Conservative `HardwareAdvisor` and local CPU/local GPU/Kaggle compute-provider contracts.
- Real tiny-model training, warmup/linear/cosine scheduling, gradient accumulation, resumable checkpoint metadata, evaluation loss, and PyTorch generation timing.
- Atomic local model/dataset/experiment registries with lifecycle validation and SHA-256 artifact records.
- Kaggle setup/train/evaluate scripts, runtime validation, configuration, documentation, and notebook; execution still requires an actual Kaggle GPU session.

## Planned

- Email provider integration, magic links, OTP, OAuth/OIDC, MFA, passkeys, organizations/RBAC, file processing/RAG, MCP, tools, agents, API keys, billing, admin, SDKs, observability, and full public documentation routes.
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
- Kaggle scripts are not locally presented as GPU-verified; Kaggle execution remains `REQUIRES_KAGGLE`.
- Local filesystem registries are single-process research storage, not a production control plane.

## Known bugs

- None confirmed in the verified first milestone.
