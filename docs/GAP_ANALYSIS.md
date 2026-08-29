# APEX-1 master specification gap analysis

Audit date: 2026-08-29. Branch at inspection: `main`. Starting worktree: clean except for test-run temporary output created during this audit. Starting commits: `3fadad1` (foundation) and `4f33892` (Sites metadata).

This matrix uses the statuses required by the master specification. A primary status describes the repository as inspected plus the milestone implemented with this audit; constraints in the evidence column remain binding. `IMPLEMENTED` means working code exists locally, not that every production-scale variation in that section exists.

## Repository evidence

- Platform: React 19/TypeScript Vinext frontend and FastAPI/SQLAlchemy API.
- Intelligence: byte BPE, decoder-only PyTorch Transformer, training/checkpoint helpers, generation with KV cache, hardware/compute modules, and research registries.
- Infrastructure: SQLite local default, PostgreSQL/Redis Docker contracts, API Dockerfile, and a private Sites frontend project. Docker Engine was unavailable during inspection.
- Preserved P0 path: registration, email verification development flow, login, refresh rotation, logout, password reset, tenant-scoped conversations, SSE streaming, and persistence.
- No RAG, memory, file ingestion, agent, MCP, plugin, multimodal, billing, worker, migration, Kubernetes, shared registry, or GPU-training implementation was found.
- Running services at inspection: no application, PostgreSQL, or Redis listener on the documented ports. Docker CLI was installed; its daemon was not running.

## Requirement matrix

| # | Requirement | Primary status | Evidence and exact gap |
|---:|---|---|---|
| 1 | Ultimate objective | PARTIALLY_IMPLEMENTED | A real platform/model foundation exists; most product and research capabilities remain absent. |
| 2 | Eight-plane architecture | PARTIALLY_IMPLEMENTED | Platform, intelligence, and deployment are modular; the requested control/data/inference/observability/security/research planes are not fully separated services. |
| 3 | Authentication | PARTIALLY_IMPLEMENTED | Email/password, verification, reset, hashed rotating refresh sessions work. OAuth/OIDC, SSO/SAML/SCIM, MFA/TOTP, recovery codes, passkeys, and device UI are missing. |
| 4 | Users/organizations | MISSING | Personal user tenancy exists, but organizations, teams, projects, invitations, policy, RBAC, and quotas do not. |
| 5 | Chat system | PARTIALLY_IMPLEMENTED | Authenticated create/list/get and SSE persistence work. Folders, edits, branches, cancellation, search/export/delete/archive mutations, sharing, and modes do not. |
| 6 | Multimodal AI | REQUIRES_EXTERNAL_API | No image/audio/video implementation. Provider abstractions and/or trained encoders plus evaluation are required. |
| 7 | File system | MISSING | No upload, validation, malware scan, object storage, parser, chunker, or async file pipeline. |
| 8 | RAG | MISSING | No embedding, sparse/dense/hybrid retrieval, reranking, citations, or caches. |
| 9 | Memory | MISSING | No persistent user-controlled memory model or privacy/export/delete flow. |
| 10 | Tools/function calling | MISSING | No schema tool runtime, permission boundary, budgets, sandbox, or audit records. |
| 11 | MCP | MISSING | No MCP registry, client/server connection, permission, secret, health, or audit implementation. |
| 12 | Plugin platform | MISSING | No manifest registry, isolation, lifecycle, or plugin runtime. |
| 13 | Agent platform | MISSING | UI navigation label only; no planner/executor/state/verifier runtime. |
| 14 | Multi-agent/council | REQUIRES_RESEARCH | No candidate, critic, ranker, or verifier experiment implementation. |
| 15 | Model router | EXPERIMENTAL | A provider registry routes only to the clearly non-LLM `apex-dev`; policy/capability/cost/privacy routing is missing. |
| 16 | Custom LLM core | EXPERIMENTAL | Configurable tiny decoder exists. No trained/evaluated 100M+ artifact exists; scaling requires data and GPU evidence. |
| 17 | Transformer architecture | EXPERIMENTAL | RMSNorm, RoPE, SwiGLU, GQA, SDPA causal attention, cache, and tying exist. MQA/FlashAttention selection and research variants are missing. |
| 18 | Mixture of Experts | REQUIRES_RESEARCH | No experts, router, capacity policy, auxiliary loss, or utilization metrics. |
| 19 | Tokenizer | PARTIALLY_IMPLEMENTED | Byte BPE training/serialization/special tokens/chat template work. Formal version registry and quality/multilingual benchmarks remain. |
| 20 | Pretraining | EXPERIMENTAL | A real tiny causal-LM loop, accumulation, checkpoint/resume, and evaluation path exist. Streaming/sharding/packing/distributed production training require GPU/data work. |
| 21 | Optimizer | PARTIALLY_IMPLEMENTED | AdamW betas/epsilon/decay/clip and constant/linear/cosine warmup schedules are implemented. Large-run validation is absent. |
| 22 | LoRA/QLoRA | MISSING | No adapters, quantized training, or merge/unmerge. |
| 23 | Alignment/preference training | REQUIRES_RESEARCH | SFT/DPO/IPO/ORPO/GRPO/reward training and verified datasets are absent. |
| 24 | Reasoning training | REQUIRES_RESEARCH | No candidate/verifier/reward/curriculum training framework. |
| 25 | Synthetic data | REQUIRES_EXTERNAL_API | No teacher/critic/verifier pipeline; provenance records alone do not implement generation. |
| 26 | Knowledge distillation | REQUIRES_RESEARCH | No response/logit/tool/reasoning distillation objective. |
| 27 | Dataset quality | MISSING | Registry metadata exists, but validation, license checks, PII/toxicity, dedup, quality scoring, packing/sharding, and dashboard do not. |
| 28 | Dataset provenance | EXPERIMENTAL | Local records capture source/license/date/processing/filters/transforms/parent/version/hash/creator/config; enforcement beyond local registration is incomplete. |
| 29 | Data mixture optimizer | REQUIRES_RESEARCH | No mixture search or benchmark impact analysis. |
| 30 | Contamination detection | REQUIRES_RESEARCH | No overlap detector or contamination report. |
| 31 | Quantization | REQUIRES_GPU | No conversion implementation or measured artifacts; model records can preserve future quantization metadata. |
| 32 | Inference optimization | EXPERIMENTAL | PyTorch SDPA, KV reuse, and a measured fallback benchmark exist. Fused kernels, compile graphs, paged caches, batching, speculation, and cache services do not. |
| 33 | Inference engines | PARTIALLY_IMPLEMENTED | PyTorch fallback exists. vLLM/SGLang/TensorRT-LLM/llama.cpp adapters are missing. |
| 34 | Performance targets | EXPERIMENTAL | Local latency and generated tokens/sec can be measured without fabricated results. TTFT/TPOT/throughput/GPU/queue/cache metrics are incomplete. |
| 35 | Hardware detection | IMPLEMENTED | `scripts/detect_hardware.py`, `apex hardware`, and `make hardware` measure OS/CPU/logical cores/RAM/display adapters/CUDA/PyTorch/disk without assumed GPU values. |
| 36 | Hardware advisor | EXPERIMENTAL | Conservative model/precision/batch/accumulation/context/engine advice uses measured free CUDA VRAM; allocation calibration on real GPUs remains. |
| 37 | Kaggle provider/files | REQUIRES_KAGGLE | Real adapter, config, setup/train/evaluate scripts, documentation, and notebook exist; Kaggle GPU execution cannot be verified locally. |
| 38 | Kaggle resource optimization | REQUIRES_KAGGLE | Runtime measurement and safe recommendation precede training; hardware microbenchmark and adaptive dataloader tuning still need Kaggle evidence. |
| 39 | GitHub↔Kaggle | PARTIALLY_IMPLEMENTED | Workflow and artifact boundaries are documented; no GitHub remote or durable registry connector is configured. |
| 40 | Fault-tolerant Kaggle training | EXPERIMENTAL | Checkpoints include model/optimizer/scheduler/scaler/RNG/steps/cursor/config/hashes/Git/hardware and resume locally. Preemption recovery needs Kaggle verification. |
| 41 | Remote compute abstraction | PARTIALLY_IMPLEMENTED | Local CPU, local GPU, and Kaggle providers exist. Remote/cloud/multi-GPU/distributed/Kubernetes providers are missing. |
| 42 | Distributed training | REQUIRES_GPU | No tested DDP/FSDP/DeepSpeed or multi-node implementation. |
| 43 | Training job system | MISSING | No persistent job model, queue, controller, logs, budgets, cancellation, or full status machine. |
| 44 | Experiment tracking | EXPERIMENTAL | Atomic local records capture the required core metadata. No concurrent/shared backend, charts, or remote synchronization. |
| 45 | Reproducibility | PARTIALLY_IMPLEMENTED | Seeds, dependency subset, Git commit, hashes, hardware, config, RNG, and cursor are captured. Environment lock/SBOM and deterministic GPU validation remain. |
| 46 | Research notebooks | PARTIALLY_IMPLEMENTED | Kaggle training notebook exists; the requested architecture/tokenizer/PEFT/alignment/quantization/ablation suite does not. |
| 47 | Benchmark system | EXPERIMENTAL | Real PyTorch fallback timing exists and emits no committed score. Modular quality suites and recognized benchmark adapters are missing. |
| 48 | Failure analysis | MISSING | No categorizer or FailureDataset registry specialization. |
| 49 | Improvement flywheel | REQUIRES_RESEARCH | No automated dataset-generation/evaluation loop or approval workflow. |
| 50 | Model registry | EXPERIMENTAL | Atomic hash-backed local records cover required metadata/lifecycle vocabulary. Shared storage, safety gates, promotion, comparison, and deployment coupling are missing. |
| 51 | Model deployment | REQUIRES_CLOUD | No validate/convert/canary/rollback deployment pipeline. |
| 52 | Online production | REQUIRES_CLOUD | A private frontend preview exists; API, databases, workers, object/vector storage, and inference are not deployed persistently. |
| 53 | Production frontend | PARTIALLY_IMPLEMENTED | React build, API URL, SSE, assets, metadata, favicon, and OpenGraph image exist. Monitoring/analytics/sitemap/robots and a reachable backend are missing. |
| 54 | API platform | PARTIALLY_IMPLEMENTED | Versioned FastAPI auth/chat/models endpoints and SSE exist. OpenAI compatibility, embeddings/files/tools/agents/MCP/webhooks/SDKs/keys are missing. |
| 55 | API keys | MISSING | No creation, one-time display, hashing, scope, limit, rotation, or usage flow. |
| 56 | Secret vault | REQUIRES_PAID_INFRASTRUCTURE | Configuration uses secret types/environment boundaries, but encryption-at-rest vault, rotation, access policy, and audit are absent. |
| 57 | Billing/usage | REQUIRES_PAID_INFRASTRUCTURE | No metering, plan, quota, spending, invoice, or payment adapter. |
| 58 | Redis | REQUIRES_PAID_INFRASTRUCTURE | Compose contract exists; application cache/rate-limit/queue/lock integration is absent and Docker daemon was unavailable. |
| 59 | PostgreSQL | PARTIALLY_IMPLEMENTED | SQLAlchemy and production URL/compose health contract exist. Migrations, backup, pool tuning, indexes audit, and replica extension are incomplete. |
| 60 | Async workers | MISSING | No queue, worker process, retries, dead letters, state, or cancellation. |
| 61 | Observability | PARTIALLY_IMPLEMENTED | Request IDs, security headers, logs, health/readiness, and a basic metrics endpoint exist. Prometheus instrumentation, OTel traces, GPU/training metrics, and dashboards do not. |
| 62 | Performance dashboard | MISSING | No metrics UI or time-series backend. |
| 63 | Cost-aware compute | MISSING | No cost estimator, price source, budget, or selection policy. |
| 64 | Security | PARTIALLY_IMPLEMENTED | Password/token/session/tenant/CORS/header controls exist. TLS edge, WAF, rate limits, scanning, SBOM/signing, secret scanning, and audit sink are missing. |
| 65 | Privacy modes | MISSING | Tenant scoping exists, but no STANDARD/PRIVATE/ENTERPRISE_PRIVATE/LOCAL routing policy or provider egress enforcement. |
| 66 | Enterprise | REQUIRES_EXTERNAL_API | No SSO/SAML/OIDC/SCIM/domain policy/private deployment/audit implementation. |
| 67 | Kubernetes | REQUIRES_CLOUD | No manifests, Helm chart, autoscaling, GPU scheduling, or cluster validation. |
| 68 | CI/CD | MISSING | No GitHub Actions pipeline for lint/type/test/security/build/container/deploy. |
| 69 | Model CI/CD | REQUIRES_RESEARCH | No benchmark/safety gate pipeline or promotion controller. |
| 70 | Disaster recovery | MISSING | No backup/restore/retention verification implementation or runbook. |
| 71 | CLI | PARTIALLY_IMPLEMENTED | Real hardware/status/train-small/evaluate/benchmark/model/dataset/experiment commands exist. Generic train/deploy/serve/inspect remain absent. |
| 72 | Make commands | PARTIALLY_IMPLEMENTED | Real hardware/status/test/lint/train-small/evaluate/benchmark targets exist. Remote/inference/deploy/migrate/backup targets await real systems. |
| 73 | Research auto-optimizer | REQUIRES_RESEARCH | No experiment generator/search scheduler or approval gate. |
| 74 | Hardware-aware auto-config | EXPERIMENTAL | Local CPU/CUDA/Kaggle detection and conservative settings exist. Remote/multi-GPU/cloud/Kubernetes calibration is missing. |
| 75 | Research performance loop | PARTIALLY_IMPLEMENTED | Train/evaluate/benchmark evidence loop exists at tiny scale; profiling, bottleneck classification, and repeat optimization are manual. |
| 76 | Long context | REQUIRES_RESEARCH | RoPE/configurable context exists, but only 128 tokens is exercised for `apex-tiny`; no 8K+ claim or mechanism validation. |
| 77 | Model memory/cache | PARTIALLY_IMPLEMENTED | Per-request Transformer KV cache exists. Prompt/prefix/embedding/retrieval/response caches and hit metrics do not. |
| 78 | Response quality | REQUIRES_RESEARCH | No factuality/reasoning/code/math/citation/tool/safety evaluation suite. |
| 79 | Safety | PARTIALLY_IMPLEMENTED | Server-side auth/tenant boundaries and refusal to expose hidden reasoning are architectural rules. Injection/file/tool/agent sandbox defenses are absent. |
| 80 | No fake implementations | IMPLEMENTED | Development response is explicitly non-LLM; unavailable GPU/Kaggle/cloud/provider paths refuse or retain required status labels. |
| 81 | Documentation corpus | PARTIALLY_IMPLEMENTED | Core architecture/security/status/roadmap/environment/Kaggle/training/gap documents exist; many named production subsystem documents remain. |
| 82 | Documentation status | PARTIALLY_IMPLEMENTED | Current status and this matrix use explicit labels. Per-feature generated status and all requested labels across every document remain. |
| 83 | Testing | PARTIALLY_IMPLEMENTED | Local auth/chat/tenancy/tokenizer/model/checkpoint/generation/hardware/provider/registry/training/evaluation tests exist. External/GPU and absent subsystems cannot be tested. |
| 84 | Milestone workflow | IMPLEMENTED | This milestone follows inspect→plan→implement→test→run→benchmark→verify→document→status→commit; later work must repeat it. |
| 85 | Git safety | IMPLEMENTED | Work began clean, preserved existing systems, keeps secrets/checkpoints/data ignored, and does not rewrite history. |
| 86 | Final architecture | PARTIALLY_IMPLEMENTED | Local platform/model/data contracts exist; internet edge, workers, shared storage, production inference, observability, and cloud compute do not. |
| 87 | Development/compute modes | PARTIALLY_IMPLEMENTED | LOCAL_DEV works; KAGGLE_RESEARCH adapter exists but requires Kaggle. Other modes are missing. |
| 88 | Kaggle primary free compute | PARTIALLY_IMPLEMENTED | Repository workflow design treats Kaggle as research-only primary free GPU; persistent registry/remote production handoff is incomplete. |
| 89 | Acceptance criteria | PARTIALLY_IMPLEMENTED | Clone/config/local dev/hardware/auth/chat/tiny train/checkpoint/resume/evaluate/local registry and benchmark paths exist. Remaining criteria map to missing rows above. |
| 90 | Engineering principle | PARTIALLY_IMPLEMENTED | Modular boundaries and reproducibility are real; cloud-native online operation is still a target, not a claim. |
| 91 | Start-now workflow | IMPLEMENTED | Repository/status/tests/dependencies/features/Git were inspected; gaps/conflicts/milestones were recorded; a P1 code milestone was implemented. |

No section is classified `BROKEN` after the audited verification. The first raw Pytest invocation was blocked by permissions on the host-global Pytest temp/cache directories; using a fresh workspace-local `--basetemp` isolates that host issue. Docker integration remains unverified because the daemon is stopped, not because the compose contract was proven broken.

## Conflicts and duplication findings

- `/v1/models` is a runtime provider catalog, not a trained-artifact registry. The new `apex.registry` module deliberately remains in the intelligence/research layer; it does not duplicate or silently replace the API router.
- Personal `tenant_id` values are server-created. Future organization/project models must extend server-side scopes rather than accepting client ownership identifiers.
- The Sites frontend repository nested under `apps/web/.git` is deployment metadata. Root Git remains the source-of-truth application repository; no platform code was forked.
- SQLite and local JSON registries are zero-service development backends. They must not be presented as PostgreSQL/object-storage production implementations.
- Kaggle is training/research compute only. It must never be wired as the only persistent production inference service.

## Implementation milestones

1. **P0 preservation — verified foundation:** keep auth, tenancy, streaming, tokenizer, Transformer, and builds green; repair only evidence-backed regressions.
2. **P1A — implemented in this milestone:** complete hardware inventory/advisor, compute providers, Kaggle workflow contracts, restartable tiny training/evaluation/benchmark, and local hash-backed registries.
3. **P1B — next:** Alembic migrations; organizations/projects/RBAC; tenant-scoped API keys with one-time secret display; Redis rate limiting/queue boundary; control-plane registry API backed by PostgreSQL/object storage.
4. **P1C:** dataset streaming/packing/validation/provenance enforcement; tokenizer version entity; 100M Kaggle allocation smoke test, checkpoint resume test, and committed run manifest without model weights.
5. **P1D:** production inference adapter contract, OpenAI-compatible text API, batching/metrics, remote GPU provider, and persistent online backend deployment.
6. **P2:** files/RAG/memory, then schema tools/MCP/plugins/agents behind permissions, budgets, sandboxing, and audits; multimodal only with real providers or evaluated encoders.
7. **P2/P3:** benchmark/failure-analysis automation, PEFT/alignment, distributed/cloud/Kubernetes, long context, MoE, and other advanced research after data/compute/evaluation gates.
