---
noteId: "a5051af093cd11f1921cf7e82490a06c"
tags: []

---

# Services

Four processes, started separately, deployed separately, failing separately.

```
frontend/    Vue app                     (existing)
backend/     chat agent + RAG            (existing)  :8000
records/     academic records facade     :8100
identity/    authentication              :8200
```

`records/` and `identity/` are independent projects. They have their own
`requirements.txt`, their own database, their own OpenAPI contract, and no import in
either direction with `backend/` — verified by:

```bash
grep -rn "from backend\|import backend" records/ identity/     # nothing
grep -rn "from records\|from identity"   backend/              # nothing
```

## How a parent question flows

```
parent → frontend → identity   POST /v1/auth/login
                               ← access token  { guardian_id: "G-1", ... }  RS256

parent → frontend → backend    chat turn, Authorization: Bearer <token>
                               backend puts the token on ChatRequestContext.
                               It never decodes it and never learns what it authorises.

                    backend → records  GET /v1/guardians/G-1/students/S-1/grades
                                       X-API-Key:      which system is asking
                                       Authorization:  which parent it asks for

                              records verifies the signature against identity's
                              public key, checks guardian_id claim == path, then
                              checks the guardian↔student link.

                    records → Moodle   grades and attendance, read live
```

The chat backend is the process running a language model on untrusted input. It holds
a token it cannot forge and cannot alter, and it has no ability to change whose records
that token authorises. That is the point of the split.

## Starting everything

Each service is independent, so start them in any order — `records` serves report card
snapshots while Moodle is down, and verifies tokens while `identity` is down.

```bash
# 1. identity  :8200
IDENTITY_ADMIN_KEY=dev-admin-key uvicorn identity.app:app --port 8200

# 2. records   :8100
RECORDS_BOOTSTRAP_ADMIN_KEY=dev-records-admin \
IDENTITY_JWKS_URL=http://localhost:8200/.well-known/jwks.json \
  uvicorn records.app:app --port 8100

# 3. backend   :8000
RECORDS_BASE_URL=http://localhost:8100 RECORDS_API_KEY=<agent key> \
  uvicorn backend.app:app --port 8000
```

`IDENTITY_ISSUER` and `IDENTITY_AUDIENCE` must match across identity and records. They
default to `school-identity` / `school-services` on both sides.

## First-run setup

```bash
# An agent-scoped key for the chat backend. The secret is shown once.
curl -X POST localhost:8100/v1/admin/api-keys \
  -H "X-API-Key: dev-records-admin" \
  -d '{"label":"chat backend","scope":"agent"}'

# A parent login, then the binding that makes it a guardian. Two calls on purpose.
curl -X POST localhost:8200/v1/admin/accounts \
  -H "X-Admin-Key: dev-admin-key" \
  -d '{"username":"0501234567","password":"...","display_name":"Umm Layla"}'

curl -X PUT localhost:8200/v1/admin/accounts/0501234567/guardian-binding \
  -H "X-Admin-Key: dev-admin-key" \
  -d '{"guardian_external_id":"G-1"}'
```

Then link the guardian to a student in records (`POST /v1/admin/guardians/G-1/students`
with `can_view_records: true` — it defaults to false), and bind the Moodle courses.

## Enabling the tool

`get_student_records` is bound by exactly one profile, `school`:

```bash
ACTIVE_PROFILE=school uvicorn backend.app:app --port 8000
```

Every other deployment behaves exactly as before — `base`, `supermew`, `document_kb`
and `ecommerce` do not bind it, and a test asserts they never start to. Every bound
tool ships its schema to the model on every call, so a stray binding would cost an
unrelated deployment tokens per turn and offer the model a capability it cannot serve.

**Binding grants nothing.** It makes the tool callable; whether anything comes back is
decided elsewhere and cannot be reached from a profile:

| Question | Answered by |
| --- | --- |
| Is this session a parent? | identity service — the `guardian_id` claim |
| Which students may they see? | records facade — the guardian link table |
| Does the school even have the data? | Moodle, through `LmsAdapter` |

A signed-in user with no guardian binding gets `NOT_A_PARENT_SESSION` with the tool
fully bound, and so does a session holding a guardian id but no token.

## What still needs building

- `records/lms.py::MoodleAdapter` — a skeleton that raises. See `records/README.md` for
  what to verify on the school's live Moodle first. **This is the critical path**;
  everything else on this list is smaller than it.
- Bulk loaders: terms, course bindings, students, guardian links, parent accounts.
  Only single-record admin routes exist, and a real school is thousands of rows.
- Postgres and deployment config for `records/` and `identity/`; both still default to
  SQLite.
- Report card publication (the write path that freezes a term), per-school grading
  policy, and rate limiting per `(key, guardian)`.
