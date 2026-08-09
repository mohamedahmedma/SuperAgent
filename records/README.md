# Academic Records Facade

A small, AI-free service between the school assistant agent and the school's system of
record. It answers "what are my child's grades" for a parent, and nothing else.

It runs on its own port, against its own database, with its own five dependencies.
Nothing here imports `backend/`, and nothing in `backend/` imports this. The only
coupling is the HTTP contract in [openapi.json](openapi.json).

## Why it exists

Three jobs, none of which belongs in the chat backend and none of which Moodle does
well:

1. **Authorisation.** Which guardian may see which student, including the custody
   restrictions that make this a legal question rather than a lookup.
2. **The school's vocabulary.** Terms, subjects and report cards — none of which
   Moodle models — mapped onto Moodle's flat course list.
3. **An audit trail.** Every attempt to read a student record, allowed or denied.

Grades and attendance are *not* stored here. They are read from the LMS at request
time. Copying them is how two systems start disagreeing about a child's transcript.

## The rule everything else follows

> An API key proves **which system** is calling. It never proves **which parent** is
> asking. Both are required before a single grade is returned.

A leaked agent key is therefore worth nothing on its own. The permitted-student set is
resolved server-side from `guardian_students` on every request, from the guardian id in
the URL path — never from anything the caller supplies, and never as a filter applied
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
| `RECORDS_LMS` | `fake` | `fake` or `moodle`. |
| `RECORDS_BOOTSTRAP_ADMIN_KEY` | — | Mints the first admin key, once. |
| `MOODLE_BASE_URL` / `MOODLE_TOKEN` | — | Required when `RECORDS_LMS=moodle`. |

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
GET  /v1/guardians/{gid}/students/{sid}/report-cards/{term_id}
```

Admin routes (`/v1/admin/...`) manage guardians, links, keys and audit reads. **Scopes
do not nest**: an admin key cannot read a student record, and an agent key cannot grant
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
- **Report cards are frozen snapshots, versioned.** A correction creates a new version;
  the old one stays readable. Live recomputation would let a policy change next year
  silently rewrite what last year's report card said.
- **Course bindings are explicit.** An unbound or unpublished Moodle course is invisible
  to parents, so a teacher's sandbox cannot reach a report card.
- **The audit is append-only.** No code path updates or deletes a row; admin exposes read
  only. Denials are logged as loudly as successes — a run of them is how probing shows up.

## What is not built yet

**`MoodleAdapter` is a skeleton** — its methods raise. It is the one piece that must be
written against a live Moodle rather than from documentation. Before writing it, verify
on the school's actual instance:

- Whether `gradereport_user_get_grade_items` is exposed and carries the **exemption
  flag**. If it does not, the excused/missing distinction has to come from
  `mod_assign_get_grades` and the adapter gets considerably chattier.
- That the **`mod_attendance` plugin** is installed with its web-service functions
  exposed, and what its configured status codes actually are — schools customise them,
  so the mapping onto present/absent/late/excused is per-deployment.
- That the token belongs to a **dedicated web-service user** with only those functions
  whitelisted. Not an admin token.

Build caching and a hard timeout into that adapter from the first line. These calls are
chatty; a parent asking three questions should not trigger thirty round trips, and a
hung call must raise `LmsUnavailable` rather than hang a chat turn.

Also still open: report card publication (the write path that freezes a term), a
`GradingPolicy` loaded per school rather than per process, and rate limiting per
`(key, guardian)`.

## Swapping the LMS

Everything LMS-shaped is behind the `LmsAdapter` protocol in [lms.py](lms.py). Routes
never import a Moodle symbol, never see a web-service function name, never handle a
Moodle error type. Replacing Moodle means writing one class — the blast radius is that
file, and the agent's tool layer does not change.
