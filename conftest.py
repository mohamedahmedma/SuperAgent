"""Session-wide test setup, imported before any test module.

Two variables, both of which have to be set before `backend` is imported: the profile the
tests run under, and whether the run ships traces to LangSmith.

## Why this is at the repo root and not under `tests/`

It sets `ACTIVE_PROFILE`, and the only moment that can work is before anything imports
`backend`. `backend/env.py` loads the project's `.env` at import, `.env` names the
DEPLOYMENT's profile, and `load_dotenv(override=False)` means whatever is already in the
environment wins. Once `backend` has been imported the variable is too late to matter.

pytest imports the rootdir conftest before collecting anything, which is early enough.

Since every suite moved under `tests/`, a `tests/conftest.py` would in fact also load
early enough today. It stays here anyway: the rootdir conftest is the earliest hook pytest
offers and the only one whose timing does not depend on where the suites happen to sit, so
it keeps working through whatever the next reorganisation is. Moving it buys nothing and
re-opens a failure whose symptom is a passing test file.

## Why it matters

The backend tests that do not name a profile assert against the default one, `school`.
Pinning it here keeps a developer's `.env` from choosing a different one underneath them.

It is an ORDERING failure when it goes wrong, which is what makes it worth a comment this
long. Every one of those tests passes in isolation: `ProfileTestCase` clears the profile
cache in its teardown, so only the files that run AFTER `test_domain_profiles.py` reload
the profile from the environment and see the deployment's.

The shell still wins — `ACTIVE_PROFILE=base pytest tests/` does what it says — because
what the shell set is captured here, before `.env` can be read. A test that needs a
particular profile should name it (`load_profile("base")`) rather than depend on which
one happens to be ambient.

## Why tracing is off

A test run is not a conversation worth recording, and `.env` carries the deployment's
LangSmith settings, so an unguarded run posts every fake turn to the project's traces and
waits on the network to do it.

It is not only noise. Measured on 2026-09-13 over the whole of `tests/general` in a fixed
order: 11 failures with tracing on, 5 with it off. The six that go are all in
`test_forced_tool_middleware.TheStreamedPath`, where `langsmith.run_helpers.async_wrapper`
drains a fake model's one-message iterator a second time and raises `StopIteration`. That
file passes in isolation either way, so only a full run shows it — which is exactly the
kind of failure a developer attributes to their own change.

Same rule as the profile: the shell wins, so `LANGSMITH_TRACING=true pytest tests/` still
records a run when someone is deliberately debugging one.
"""
import os

_FROM_SHELL = (os.environ.get("ACTIVE_PROFILE") or "").strip()
os.environ["ACTIVE_PROFILE"] = _FROM_SHELL or "school"

#: Both spellings: `LANGCHAIN_TRACING_V2` is the older one, and langsmith still reads it.
for _tracing in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
    if not (os.environ.get(_tracing) or "").strip():
        os.environ[_tracing] = "false"
