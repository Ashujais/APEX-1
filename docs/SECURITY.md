# Security foundation

Implemented controls include Argon2id password hashing, short-lived signed access tokens, opaque refresh tokens stored only as SHA-256 hashes, refresh rotation, session revocation, generic credential errors, server-side tenant scoping, request IDs, CORS allowlisting, and development-only exposure of verification/reset tokens.

Production mode refuses to start without an explicit `APEX_SECRET_KEY`. Development mode creates a process-local random key, so sessions intentionally become invalid after restart. Secrets are never returned by configuration endpoints or written to logs.

Not yet production-complete: email delivery, MFA, WebAuthn/passkeys, OAuth/OIDC, SAML/SCIM, CSRF protection for cookie-based flows, rate limiting, malware scanning, encrypted provider vaults, external audit sinks, penetration testing, and independent security review.
