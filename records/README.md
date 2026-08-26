---
noteId: "3e5d04b095c711f1a6d4fb6c6accc4db"
tags: []

---

# Academic Records Facade

A small, AI-free service between the school assistant agent and the school's system of
record. It answers "what are my child's grades" for a parent, and nothing else.

It runs on its own port, against its own database, with its own five dependencies.
Nothing here imports `backend/`, and nothing in `backend/` imports this. The only
coupling is the HTTP contract in [openapi.json](openapi.json).

## Why it exists

Three jobs, none of which belongs in the chat backend and none of which Moodle does
well:

1. **Enforcement.** Two credentials on every parent-facing read — a key proving which
   *system* is calling, and a signed token proving *which parent* it asks for — plus a
   refusal for anything it cannot justify.
2. **The school's vocabulary.** Grading policy (letter bands, the pass mark, which of the
   two percentages leads), and the course binding that maps a flat course list onto
   "subject x section x term" for a backend whose titles are whatever a teacher typed.
3. **An audit trail.** Every attempt to read a student record, allowed or denied,
   correlated back to the chat turn that caused it.

**Authorisation is no longer on that list, and that is the important change.** Which
guardian may see which student is the registrar's fact: entered from paperwork, amended by
custody decisions, and audited in `sis/`. This service asks rather than remembers, and
since the guardian handle now travels with every read, `sis/` re-checks the answer before
returning a mark. Two independent refusals from one source of truth, instead of a second
copy that goes stale the first time a court order is applied to the other one.

Grades, attendance, guardians, students and the academic calendar are *not* stored here.
They are read at request time. Copying them is how two systems start disagreeing about a
child.

## The rule everything else follows

> An API key proves **which system** is calling. It never proves **which parent** is
> asking. Both are required before a single grade is returned.

So every parent-facing read carries two independent credentials:

| Header | Proves | Issued by |
| --- | --- | --- |
| `X-API-Key` | which system is calling | this service's admin routes |
| `Authorization: Bearer` | which parent it asks for | the [identity service](../identity/) |

The token's `guardian_id` claim **must match** the `guardian_id` in the URL path. That
equality check is what stops the calling system choosing whose records it reads — it
relays a parent's token and cannot produce a signature for a different one. A fully
compromised chat backend still cannot read a family it holds no token for.

Verification is offline against a public key, so this service holds nothing that could
mint a token, and identity being down does not take records down. It **fails closed**:
with no public key configured every parent-facing read returns 503 rather than falling
back to trusting the path.

The permitted-student set is then resolved server-side from `guardian_students` on
every request — never from anything the caller supplies, and never as a filter applied
to results after the fact.

This matters more than usual because the caller is a language model. No prompt, no
injected instruction inside a parent's chat message, and no clever phrasing reaches
this decision: the LMS is never asked about a student the link check excluded.

## Running it

```bash
pip install -r records/requirements.txt
python -m records.export_openapi              # regenerate the contract
RECORDS_BOOTSTRAP_ADMIN_KEY=... uvicorn records.app:app --port 8100
pytest records/tests -q
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `RECORDS_DATABASE_URL` | `sqlite:///./records.db` | Point at Postgres for real data. |
| `RECORDS_LMS` | `fake` | `fake`, `moodle`, or `sis` (the school's own SIS on :8300). |
| `RECORDS_BOOTSTRAP_ADMIN_KEY` | — | Mints the first admin key, once. |
| `IDENTITY_JWKS_URL` | — | **Required.** e.g. `http://localhost:8200/.well-known/jwks.json`. |
| `IDENTITY_PUBLIC_KEY_PEM` | — | A pinned key instead of JWKS. Takes precedence. |
| `IDENTITY_ISSUER` / `IDENTITY_AUDIENCE` | `school-identity` / `school-services` | Must match the identity service. |
| `MOODLE_BASE_URL` / `MOODLE_TOKEN` | — | Required when `RECORDS_LMS=moodle`. |
| `SIS_BASE_URL` / `SIS_API_KEY` | — | Required when `RECORDS_LMS=sis`. The key must be `reader`-scoped. |
| `SIS_TIMEOUT_SECONDS` | `10` | Per-call ceiling. No retries: a retry budget multiplies it. |

## The contract

Every parent-facing read lives under `/v1/guardians/{guardian_id}/...`. The subject is
part of the path rather than an optional parameter, so a route that reads a record
without naming a guardian has nowhere to put one.

```
GET  /v1/terms
GET  /v1/guardians/{gid}/students
GET  /v1/guardians/{gid}/students/{sid}/grades?term=
GET  /v1/guardians/{gid}/students/{sid}/grades/{course_id}
GET  /v1/guardians/{gid}/students/{sid}/attendance?term=
```

Admin routes (`/v1/admin/...`) mint keys and read the audit. The three that used to manage
guardians and links answer **410**, naming the SIS routes that replaced them — a 404 would
read as "wrong URL" and invite a retry, and accepting the write silently would leave a
registrar believing a parent had been granted access when nobody had. **Scopes do not
nest**: an admin key cannot read a student record, and an agent key cannot grant
itself access. Making admin a superset would mean the school's most widely copied
credential is also the one that reads every record.

Two conventions in every response, both defences against a model reading a record wrong:

- **Nothing is silently absent.** A missing grade carries a `status` explaining why.
  A model handed `null` will narrate a plausible reason; one handed `"excused"` will not.
- **Every payload is stamped** with `as_of`.

On `503 {"code": "lms_unavailable"}` the agent must say records are temporarily
unavailable. Never a remembered or inferred figure.

## Decisions baked into the schema

These are the ones that are cheap now and brutal to retrofit once real data lands.

- **Excused ≠ zero.** An excused assignment leaves the denominator entirely. A missing
  one stays in it as a real zero. This is the most common way a home-built gradebook
  quietly harms a real student — see `test_excused_and_missing_are_not_the_same`.
- **Linking a guardian is not granting access.** `can_view_records` defaults to `False`,
  so a half-finished import leaks nothing.
- **Restricted ≠ deleted.** A guardian barred by a court order stays on file as a
  contact, with the reason attached, because deleting the row loses a fact the school
  needs.
- **Denials are indistinguishable to the caller.** Unknown, unrelated and restricted all
  return the same 404 with the same message; the audit records which actually happened.
  A caller who could tell them apart could enumerate the student body.
- **Course bindings are explicit.** An unbound or unpublished course is invisible to
  parents, so a teacher's sandbox cannot reach a rollup. Unused on the SIS path, which
  reports its own subjects against the school's own codes — and kept for exactly that
  reason, since it is what a *different* system of record would need.
- **The audit is append-only.** No code path updates or deletes a row; admin exposes read
  only. Denials are logged as loudly as successes — a run of them is how probing shows up.

## What is not built yet

**`MoodleAdapter` is a skeleton** — its methods raise. The questions it was blocked on
have now been answered against a real Moodle 5.1.6 with mod_attendance 2026042100; the
full findings and a reproducible local instance live in `~/moodle-dev/` inside the
Ubuntu-22.04 WSL2 distro (moved there off the Windows disk for speed), and the
implementation notes are in the `MoodleAdapter` docstring in [lms.py](lms.py).

Two results change this service's design.

**Grades: read Moodle's computed total, do not re-aggregate.** The web service exposes
no exclusion flag — an excused assignment is indistinguishable from a counted one. But
the course-total row's `percentageformatted` is correct, because Moodle applies
exclusions, weights and drop-lowest itself. Measured on a student with 90/100, an
excluded 10/100 and one ungraded item, the three candidate approaches give **50 %**,
**30 %** and **90 %** — only the last is right.

So [grading.py](grading.py)'s per-course arithmetic should be replaced by reading that
figure. Its term-level rollup stays: Moodle still has no concept of a term. The
`EXCUSED` status survives in the contract because a future SIS may report it, but
Moodle will never populate it.

**Attendance: the core web services are unusable for this.** Reading one child's
attendance returns every classmate's name and status, requires write-capable
permissions, and requires the service account to be enrolled as a teacher in every
course. A `local_` Moodle plugin exposing a read-only per-student endpoint is the
supported path.

Build caching and a hard timeout into that adapter from the first line. These calls are
chatty; a parent asking three questions should not trigger thirty round trips, and a
hung call must raise `LmsUnavailable` rather than hang a chat turn.

Also still open: a `GradingPolicy` loaded per school rather than per process, and rate
limiting per `(key, guardian)`.

Report cards were removed rather than finished. The read route had been broken since the
guardian tables stopped being populated — it looked up `student.id` on an object with no
`id` — and nothing tested it. Freezing a published term is a real requirement, but it is a
write path over marks, and it belongs where the marks are.

## Swapping the LMS

Everything LMS-shaped is behind the `LmsAdapter` protocol in [lms.py](lms.py). Routes
never import a Moodle symbol, never see a web-service function name, never handle a
Moodle error type. Replacing Moodle means writing one class — the blast radius is that
file, and the agent's tool layer does not change.

### `RECORDS_LMS=sis`

[sis_adapter.py](sis_adapter.py) is that claim tested. It reads the school's own Student
Information Service (`:8300`, see [../SERVICES.md](../SERVICES.md)) —
`GET /v1/students/{student_number}/grades?term=` with a `reader`-scoped `X-API-Key` — and
nothing in the routes, the assembler or the tool layer changed to accommodate it.

```bash
RECORDS_LMS=sis SIS_BASE_URL=http://localhost:8300 SIS_API_KEY=<reader key> \
  uvicorn records.app:app --port 8100
```

Four things to know before switching a deployment to it:

- **Bind courses on the subject code.** `records.assembler` matches on the reference the
  system of record reports, which for SIS is the subject code (`MATH`), so a
  `CourseBinding` needs `lms_idnumber` set to that. Unbound subjects are dropped, exactly
  as they are under Moodle — a subject the school has not published is not on a report.
- **Attendance is unavailable.** SIS records grades only, so `get_subject_attendance`
  raises `LmsUnavailable` and the attendance route answers 503. Returning an empty
  register instead would have reported every child as perfectly attending.
- **One figure per subject.** SIS states the mark a teacher wrote; there is no assignment
  ledger, no weighting and no attendance mixed in, so `academic` carries the same figure
  as the course total rather than a second one. A mark stated only as points ("17 out of
  20") arrives with `academic.unavailable: "points_not_percentage"` — this adapter will
  not divide to manufacture a percentage.
- **An unknown student reads as no records, not as an error**, which is the same answer a
  restricted or unlinked student gets. Any other refusal — a revoked key, a wrong base
  URL, a timeout — is `LmsUnavailable` and reaches the parent as "records are temporarily
  unavailable".
