---
noteId: "a5051af093cd11f1921cf7e82490a06c"
tags: []

---

# Services

Five processes, started separately, deployed separately, failing separately.

```
frontend/    Vue app                     (existing)
backend/     chat agent + RAG            (existing)  :8000
records/     academic records facade     :8100
identity/    authentication              :8200
sis/         student information service :8300

tests/       every pytest suite in the estate — not a process
```

The services hold code. **Every automated check lives in [`tests/`](tests/)**, one
directory per service plus `general/` for the backend and the cross-service journeys, and
`evals/` for retrieval scoring. `pytest` from the repository root still runs all of it;
`python tests/run_regression.py` runs each suite in its own interpreter, which is what the
pre-merge check should use — the service suites configure themselves through environment
variables set at import time, so in one shared process what a suite sees depends on which
suite was collected before it. See [tests/README.md](tests/README.md).

`records/`, `identity/` and `sis/` are independent projects. They have their own
`requirements.txt`, their own database, their own OpenAPI contract, and no import in
either direction with `backend/` — verified by:

```bash
grep -rn "from backend\|import backend" records/ identity/ sis/  # nothing
grep -rn "from records\|from identity\|from sis" backend/        # nothing
```

`sis/` is the school's own registrar-facing system of record: year levels, classes,
subjects, terms, time-bounded class placements, spreadsheet imports, the marks a teacher
stated, and the guardians a child may be contacted through. It is a peer of the gradebook, not a
layer over it — `records/` reads it through the same `LmsAdapter` seam, and `sis/` does
not know `records/` exists.

Guardians live in `sis/` and are **not** the same table as `records/`'s. A student on the
SIS roll need not exist in `records/` at all, which is the intended state rather than
something to reconcile: `records/` is the older facade and its own guardian model is
on its way out. Three tables carry it — `guardians`, `guardian_phones` and
`student_guardians` — and the split is what lets one parent hold two numbers and one child
have any number of adults, each with their own relationship and their own permission to
read academic records.

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

                    records → sis     grades and attendance, read live
                                       GET /v1/students/{number}/grades?term=
                                       X-API-Key: a reader-scoped SIS key
```

Which system of record answers that last hop is `RECORDS_LMS`, and it is the only thing
that changes: `fake` or `sis`. No route, tool schema or parent-facing contract
differs between them.

The chat backend is the process running a language model on untrusted input. It holds
a token it cannot forge and cannot alter, and it has no ability to change whose records
that token authorises. That is the point of the split.

## Starting everything

Each service is independent at *run* time — `records` keeps verifying tokens while
`identity` is down, and says so honestly when `sis/` is. Setup has one ordering constraint:
`sis/` must be up long enough to mint the reader keys the other two now refuse to start
without.

```bash
# 1. identity  :8200
IDENTITY_ADMIN_KEY=dev-admin-key uvicorn identity.app:app --port 8200

# 2. records   :8100
RECORDS_API_KEY=dev-records-agent \
IDENTITY_JWKS_URL=http://localhost:8200/.well-known/jwks.json \
  uvicorn records.app:app --port 8100

# 3. backend   :8000
RECORDS_BASE_URL=http://localhost:8100 RECORDS_API_KEY=dev-records-agent \
  uvicorn backend.app:app --port 8000

# 4. sis       :8300  (only when RECORDS_LMS=sis)
SIS_BOOTSTRAP_REGISTRAR_KEY=dev-sis-registrar uvicorn sis.app:app --port 8300
```

`SIS_DEFAULT_COUNTRY_CODE` (default `+20`, Egypt) is what a guardian's phone number is
normalised against when a registrar types it in national form. It matters more than it
looks: Excel formats a phone column as a number and drops the leading zero, so
`01001234567` reaches the service as `1001234567`, and this setting is what turns it back
into a number that can actually be dialled. A number typed with its own `+` prefix ignores
the setting entirely, so a foreign parent is unaffected at any value, and a bare `20` is
accepted and read as `+20`. Set it wrong and every locally-typed parent number in the
school points at another country.

### SIS authenticates its callers

`sis/` verifies a presented `X-API-Key` against its own `api_keys` table. Two scopes,
compared by **exact equality** — `registrar` does not satisfy a `reader` check and never
implies one — and a key is looked up in the school's own database, so a key minted at one
branch does not exist at another and cannot be made to work there by supplying its
`X-School-Code`.

Two services call it, and each needs its own **`reader`** key:

```bash
# `dev-sis-registrar` is the bootstrap key from the SIS start line above.
curl -X POST localhost:8300/v1/admin/api-keys   -H "X-API-Key: dev-sis-registrar"   -d '{"label":"records adapter","scope":"reader"}'   # -> SIS_API_KEY

curl -X POST localhost:8300/v1/admin/api-keys   -H "X-API-Key: dev-sis-registrar"   -d '{"label":"identity directory","scope":"reader"}' # -> IDENTITY_SIS_API_KEY
```

A registrar key would also read grades, which is the reason not to use one: the processes
that answer parents must not hold the school's write credential. Two separate reader keys
rather than one shared value, so an audit line or a revocation can name a single caller.

Both services **refuse to start** if their base URL is set without a key. That is
deliberate: an unkeyed caller gets a 401 on every request, which downstream reads as "the
school has no such child" and "your number is not registered" — a silent, total failure
dressed as an ordinary answer.

`SIS_BOOTSTRAP_REGISTRAR_KEY` is a full registrar credential for every school, is not in
the `api_keys` table, and cannot be revoked through the API. Unset it once the keys above
exist; `sis/` logs a warning at startup for as long as it is configured.

Course bindings must key on the SIS subject code — details in
[records/README.md](records/README.md#records_lmssis).

`IDENTITY_ISSUER` and `IDENTITY_AUDIENCE` must match across identity and records. They
default to `school-identity` / `school-services` on both sides.

## How a parent signs in

Parents have no password. They prove they hold a number the school already has on file, by
sending one WhatsApp message:

```
browser  -> identity  POST /v1/auth/whatsapp/start
                      <- a wa.me link carrying a nonce, and a poll secret for the browser

parent   -> WhatsApp  taps the link and sends the pre-filled message
                      (WhatsApp never sends it for them)

Meta     -> identity  POST /v1/auth/whatsapp/webhook, signed
identity -> sis       POST /v1/guardians/resolve { phone }
                      <- the guardian's stable public_id, or 404

identity -> WhatsApp  a six-digit code, free: the parent opened the service window
browser  -> identity  POST /v1/auth/whatsapp/verify { poll_secret, code }
                      <- the same token a password login returns, carrying guardian_id
```

Two secrets on purpose. The nonce goes out in a link and comes back over WhatsApp; the poll
secret never leaves the browser. Someone who forwards the link cannot finish the sign-in,
and someone who tricks a parent into sending theirs cannot read the code that results.

It costs nothing because the parent messages first: replies inside the 24-hour customer
service window are not template messages, and Meta does not charge for those. It creates
nobody — a number `sis/` does not hold is refused, because whose parent somebody is stays
the registrar's fact. Setup, and the reason the number must carry its `+`, are in
[identity/README.md](identity/README.md#parent-login-by-whatsapp).

## First-run setup

```bash
# A parent login, then the binding that makes it a guardian. Two calls on purpose.
curl -X POST localhost:8200/v1/admin/accounts \
  -H "X-Admin-Key: dev-admin-key" \
  -d '{"username":"0501234567","password":"...","display_name":"Umm Layla"}'

curl -X PUT localhost:8200/v1/admin/accounts/0501234567/guardian-binding \
  -H "X-Admin-Key: dev-admin-key" \
  -d '{"guardian_external_id":"G-1"}'
```

Then link the guardian to a student **in `sis/`** — upload a guardians sheet, or
`PATCH /v1/students/{student_number}/guardians/{phone}` with `can_view_records: true`.

`records/` needs nothing set up. It has no database, mints no keys and holds no rows: its
credential is `RECORDS_API_KEY` in the environment, the same value the chat backend sends.

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
| Which students may they see? | the SIS — the registrar's own guardian link, checked twice |
| Does the school even have the data? | the SIS, through `LmsAdapter` |

A signed-in user with no guardian binding gets `NOT_A_PARENT_SESSION` with the tool
fully bound, and so does a session holding a guardian id but no token.

## What still needs building

- Bulk loaders: terms, course bindings, students, guardian links, parent accounts.
  Only single-record admin routes exist, and a real school is thousands of rows.
- Postgres and deployment config for `records/` and `identity/`; both still default to
  SQLite.
- Report card publication (the write path that freezes a term), per-school grading
  policy, and rate limiting per `(key, guardian)`.
