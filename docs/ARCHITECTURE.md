# Architecture

APEX-1 is divided into three independently evolvable layers.

- **Platform:** `apps/web` and `apps/api` contain the user interface, authentication, API, conversations, and future product services.
- **Intelligence:** `packages/apex-core` contains tokenizer, model, training, inference, hardware, and later reasoning/multimodal code. It does not depend on the web application.
- **Infrastructure:** `deployment`, `.env.example`, and later migrations/observability configuration define local and production services.

The first vertical slice uses SQLite for zero-service local startup. SQLAlchemy keeps the persistence boundary compatible with PostgreSQL. Redis, object storage, queues, vector storage, and GPU inference remain explicit infrastructure milestones.

Chat currently routes only to `apex-dev`, a deterministic development responder. It exists to verify streaming, storage, authentication, and tenancy without presenting an untrained model as useful AI. The Transformer core is separately tested and remains experimental until trained and evaluated artifacts are registered.
