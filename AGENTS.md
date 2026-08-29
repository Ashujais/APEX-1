# APEX-1 repository guidance

- Preserve explicit capability labels: Implemented, Experimental, Planned, or Requires external infrastructure.
- Never add benchmark or superiority claims without reproducible evidence committed with the project.
- Keep tenant ownership checks in server-side queries; never trust a tenant or user identifier supplied by a client.
- Never log tokens, passwords, provider credentials, secrets, or uploaded private content.
- Keep the platform (`apps/`), intelligence (`packages/apex-core/`), and infrastructure (`deployment/`) modular.
- Run the relevant Python tests and web build before marking a milestone implemented.
