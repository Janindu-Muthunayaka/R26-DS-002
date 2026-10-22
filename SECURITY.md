# Security Policy

## Supported Versions

This is an academic research project. Only the latest code on the `main`
branch is supported. Older commits and feature branches do not receive fixes.

| Version         | Supported |
| --------------- | --------- |
| `main` (latest) | ✅        |
| anything else   | ❌        |

## Reporting a Vulnerability

**Please do not open a public issue for security problems.**

Report vulnerabilities privately through GitHub:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability** (GitHub private vulnerability reporting).
3. Describe the issue, the affected file or endpoint, and steps to reproduce.

If private reporting is unavailable, contact a member of the
[project team](README.md#team) directly.

You can expect an acknowledgement within **7 days**. We will tell you whether
the report is accepted and, if so, roughly when a fix will land.

## Areas of particular concern

- **API keys and secrets.** `system/.env` holds keys such as `OPENAI_API_KEY`.
  It must never be committed. If you find a key in the history, report it so
  it can be rotated.
- **The local server** (`system/app/server.py`) accepts image uploads from a
  phone on the same network. Upload handling, path traversal and
  unauthenticated access are in scope.
- **The RAG service** (`services/rag`) stores text from captured articles and
  chat sessions. Anything that exposes that stored content is in scope.
- **Personalization data** (`system/layers/l7_personalization/data/`) holds
  interaction logs and preferences. Anything that exposes it is in scope.

## Out of scope

- Vulnerabilities in third-party dependencies with no demonstrated impact on
  this project (please report those upstream).
- Issues that need physical access to the machine running the server.
