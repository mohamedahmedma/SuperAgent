# SIS — Student Information Service

The school's record of fact: who is enrolled, in which class, and what the school has stated
about their work and their attendance. A FastAPI service on **:8300** plus a React console it
serves at **/ui**.

It writes school data, and it is the only process in this repo that does. Nothing here averages,
ranks, or infers: a blank grade is `null` and never `0`, an unmarked register entry is a third
state beside present and absent, and a class placement is a dated membership rather than a
column — so a transfer in March leaves October saying what October said.

---

## Running it

### The service

```bash
# from the repository root, with the virtualenv active
python -m uvicorn sis.app:app --port 8300 --reload
```

- API: <http://127.0.0.1:8300/v1>
- Interactive contract: <http://127.0.0.1:8300/docs>
- Console: <http://127.0.0.1:8300/ui/>
- Liveness: <http://127.0.0.1:8300/health>

The database is SQLite at `sis.db` by default. Migrations are Alembic:

```bash
python -m alembic -c sis/alembic.ini upgrade head
```

Back the file up before a migration. `sis/migrations/env.py` turns foreign keys off for the
duration of a SQLite migration and runs `PRAGMA foreign_key_check` afterwards; a failed check
aborts rather than leaving a half-migrated file.

### The console

The console is a **build**. This is the thing to know before anything else:

> `sis/web/` is build output. Nothing in it should ever be edited, and the build wipes it.
> The source is `sis/frontend/src/`.

```bash
cd sis/frontend
npm install          # first time only
npm run build        # writes ../web, which the service already serves
```

Then reload `/ui`. `index.html` is served `no-cache` and the hashed assets are served
`immutable`, so a rebuild is visible on the next refresh without a hard reload.

### Working on the console

`npm run build` after every change is slow. For real work, run the dev server:

```bash
cd sis/frontend
npm run dev          # http://127.0.0.1:5173
```

That serves the app on **:5173** with hot reload and proxies `/v1` and `/health` through to the
Python service on :8300 — so keep uvicorn running in another terminal and open **:5173**, not
`/ui`.

The two are separate: **`npm run dev` does not change what `/ui` serves.** `/ui` shows the last
`npm run build`. If a change is not visible at `/ui`, the build has not been run; if a change is
not visible at :5173, the dev server has not been restarted since a config change.

---

## Adding a school

Schools are separated physically — one database each — so creating one is a database, a
migration and a row, not an `INSERT`. Two ways in, the same code behind both:

```bash
python -m sis.schools provision NCS --name-en "Nasr City" --dry-run   # what it would do
python -m sis.schools provision NCS --name-en "Nasr City"             # do it
python -m sis.schools list                                            # every school and its revision
```

```
POST /v1/admin/schools     {"code": "NCS", "name_en": "Nasr City"}    # registrar scope
```

The connection is **rendered, never typed**. `SIS_DATABASE_URL_TEMPLATE` holds the one
connection the estate shares with `{slug}` where the school's part goes, and provisioning
writes the result into `.env` as `SIS_DATABASE_URL_NCS` alongside an updated `SIS_SCHOOLS`.
Nobody pastes a URL, so no two schools can end up pointed at one database.

The order is database first, configuration last, and it is not arbitrary: a crash in
between leaves a database nothing points at, which the next attempt refuses to overwrite.
The other order leaves `SIS_SCHOOLS` naming a school with no database, and
`sis.tenancy.get_registry` refuses to start the service at all in that state.

The HTTP route is **off unless armed**. It answers 503 until `SIS_ADMIN_DATABASE_URL` is
set, because creating a PostgreSQL database needs a role that can also drop one, and the
process serving parent requests should not hold it. A deployment that prefers a terminal
simply never sets it.

The rules — naming, the refusals, the ordering — are in
[`sis/application/services/estate.py`](application/services/estate.py) and do no I/O;
[`tests/sis/test_estate_provisioning.py`](../tests/sis/test_estate_provisioning.py) drives
them with strings and fakes.

**One caveat.** `SIS_SCHOOLS` is read once per process. A school created through the API
is live in the worker that served the request and not in the others until they restart, so
with more than one worker, provisioning is complete after a rolling restart.

## Testing

Three suites, each catching something the others cannot.

```bash
# 1. The service: domain, repositories, routes, migrations.
python -m pytest tests/sis -q

# 2. The console renders. Mounts the app in jsdom against stubbed responses and walks
#    every screen, failing on a blank screen or anything written to console.error.
cd sis/frontend && npm run smoke
```

The third is inside the first. `tests/sis/test_ui_contract.py` reads the console's source as
text and compares it against `app.openapi()` — it is what catches a screen calling a route
nobody wrote, or sending a body key the service does not read. `tests/sis/test_ui_fixtures.py`
checks the smoke test's stubs against the same document, because a stub that is wrong in the
same way as the screen certifies the bug instead of catching it.

Run the contract suite after touching either side of the boundary. It is fast and it has caught
every shape mismatch made so far.

---

## Layout

```
sis/
  api/            routers and dependency wiring — HTTP only, no rules
  application/    services and ports; the use cases
  domain/         entities, value objects, invariants. No imports from anything above.
  infrastructure/ SQLAlchemy models, repositories, unit of work
  migrations/     Alembic
  frontend/       the console's source — React 18, Vite 5, Bootstrap 5
    src/styles/   tokens.css (the palette and the motion scale), theme.css (maps them onto
                  Bootstrap's --bs-*), base.css, sis.css
  web/            BUILD OUTPUT. Do not edit.
```

The dependency arrow points one way: `api → application → domain`, with `infrastructure`
plugged in at the edge. `domain/` imports nothing from the layers above it, and the test suite
would notice if it did — it lives in [`tests/sis/`](../tests/sis/), with every other suite in the
estate.

### Design tokens

The palette is **Apple's**: white, Apple's greys, black, and systemBlue. The grey ramp draws
from both places Apple publishes one — the apple.com web greys (#F5F5F7 behind a section,
#D2D2D7 for a rule, #86868B and #6E6E73 for secondary text, #1D1D1F for body copy) and the iOS
system greys, which supply the dark end — anchored on **#8E8E93** (`systemGray`, `--grey-500`).

The page is **a light off-white and the panels on it are grey**: a screen is a short stack of
slabs with a hairline and a 14px corner, and a form control on one is the page colour (`--field`)
or it would be the colour of the card behind it.

**The page colour is a setting, not a constant.** `--tint` is the page — #FAF9F7 by default, an
off-white with a little black in it — and the panel, the hover, the well and the two hairlines
are all that colour mixed a measured distance toward a grey. The settings dialog writes one
custom property onto `<html>` and the whole console follows, which is what lets a colour a user
picked look like part of the design instead of a stripe of their colour laid over one. Each
appearance remembers its own; the contract suite pins the derivation so a later edit cannot
quietly hand-set a surface and break the following.

The **settings dialog** (the gear in the header) holds the three preferences a person owns rather
than a school does — appearance, page colour, and which of a child’s two names is shown first.
Its backdrop blurs rather than dims, because a dialog about what a colour does to the page has to
leave the page visible, and it carries a working miniature of the console drawn from the same
tokens the real screens use.

### Arabic, and the two directions

The console's own wording is translated, not only the names it shows. `src/locale/ar.js` is the
whole of it: a flat table keyed by the **English sentence**, so the code still reads as English
prose, a missing entry renders in English rather than as a key, and the file can be handed to a
translator as a list of sentences. `t('Add school')` is the call; `t('{0} class(es)', [n])` is
the call when a value sits inside the sentence, because gluing a number onto a string forces
Arabic into English word order.

Two things stay Latin on purpose: identifiers (a class is `3A`, a year is `2025-2026`, a student
number is Latin digits — a registrar has to match the screen against the paper in their hand) and
the CSV column names an upload has to literally contain.

**RTL is not Bootstrap's here.** Bootstrap's `.ms-*`/`.me-*` are named for start and end but
compile to `margin-left`/`margin-right` in the default build; the logical behaviour lives in
`bootstrap.rtl.css`, which this console does not load. So `ms-auto` pushed the header controls
the same way in both directions, and in Arabic the whole header collapsed into one clump against
the brand. The fix is `.sis-push` / `.sis-pull` (logical, mirror themselves) where the push is
unconditional, and `flex-*-grow-1` on the sibling where it changes at a breakpoint — which is
direction-agnostic already and needs no media query of ours.

**The theme travels on `data-bs-theme` and on nothing else.** Bootstrap’s components watch that
attribute and cannot be taught another, so a second name would be a second source of truth — and
was: `tokens.css` keyed its dark blocks on a `data-theme` nobody set, which broke exactly one
path (choosing Dark explicitly on a light machine) and was covered for everywhere else by the
`prefers-color-scheme` rules. There is a test for it now.

The blue is spent on four things and nothing else: a link, the focus ring, the current item in a
set, and the one button on a form that commits. That last one is #007AFF with a white label,
which measures 4.02:1 — short of the 4.5:1 WCAG AA floor, chosen knowingly, and recorded with
its number in `tokens.css`. Hover darkens to #0062CC (5.80:1), so the label never fades under
the pointer. The contract suite computes these rather than eyeballing them.

Every animation duration goes through `--motion-scale`, so one variable governs all motion and
the reduced-motion rule is a rule rather than a convention.

**Responsive behaviour is Bootstrap's, in the markup.** There is no width or height media query
in any of our stylesheets, and the contract suite fails if one appears. Phones are the primary
target: the narrow case is written first (`col-12 col-md-6`, `d-grid gap-2 d-sm-flex`), tables
hide their secondary columns below a breakpoint via the `hide` key on a column, and the shell's
nav scrolls rather than wrapping.
