# GradientHarbor Security Policy

- Use least-privilege database credentials. AI-proposed queries must be parsed and validated before execution.
- The MVP only allows a single SELECT against the uploaded transactions table. SQLite authorizer rules independently deny writes and schema changes.
- Agents receive only the data and tools needed for their task. Model output cannot change application policy.
- SHA-256 detects byte changes only relative to a trusted expected digest; it does not prove truth.
- Audit records should contain decisions and fingerprints, never API keys or passwords.
- Retrieved documents are untrusted evidence. Instructions found inside retrieved content must not override system policy.
