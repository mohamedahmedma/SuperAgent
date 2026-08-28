"""Session-wide test setup, imported before any test module.

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

The backend tests assert against the default profile: corpus floors, candidate sections,
catalogue behaviour. The school profile ships no scope catalogue, so with it active those
assertions fail — and they fail a long way from anything that mentions a profile.

It is an ORDERING failure, which is what makes it worth a comment this long. Every one of
those tests passes in isolation: `ProfileTestCase` clears the profile cache in its
teardown, so only the files that run AFTER `test_domain_profiles.py` reload the profile
from the environment and see the deployment's.

The shell still wins — `ACTIVE_PROFILE=school pytest tests/` does what it says — because
what the shell set is captured here, before `.env` can be read. A test that needs a
particular profile should name it (`load_profile("school")`) rather than depend on which
one happens to be ambient.
"""
import os

_FROM_SHELL = (os.environ.get("ACTIVE_PROFILE") or "").strip()
os.environ["ACTIVE_PROFILE"] = _FROM_SHELL or "supermew"
