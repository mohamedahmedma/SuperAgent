# The test suite

Every automated check in the estate lives here. The services hold code; this holds the
proof that the code still does what it claims.

```
tests/
  general/      the chat backend: RAG, retrieval, prompts, the agent loop, and the
                cross-service journeys that exercise the whole estate end to end
  identity/     the authentication service
  records/      the academic records facade
  schoolauth/   the shared token verifier
  sis/          the school's system of record
  evals/        retrieval quality — a score, not a verdict. Not run by pytest.

  run_regression.py   the pre-merge check: every suite, one process each
  __init__.py         makes tests/ a package, so tests/identity/ is named
                      `tests.identity` and cannot shadow the real `identity`
```

## Running things

```bash
pytest                          # everything, one process, from the repository root
pytest tests/sis -q             # one suite, while you are working on it
pytest tests/general/test_child_context.py -q

python tests/run_regression.py          # the build gate: each suite isolated
python tests/run_regression.py --eval   # and score retrieval afterwards
```

`pytest` from the root still collects all of it, exactly as before. The runner exists for
the different question — *is this build good?* — and the difference between them is worth
understanding before you trust either.

## Why the runner uses one process per suite

Every service suite configures itself by **setting environment variables at import time**,
before the service it tests is first imported. `sis/conftest.py` binds `SIS_DATABASE_URL`
to a temporary file; `identity/conftest.py` points the signing key somewhere disposable;
`records/conftest.py` fixes the issuer and audience. That is the only moment those values
can be set, because the modules under test read them once and cache them.

In a single process those assignments are global and permanent, so what any given suite
sees depends on which suite was collected before it. The long comment at the top of the
repository root's `conftest.py` documents one instance of that hazard, and it is long
because the failure is invisible: every affected test passes when run alone.

One interpreter per suite removes the class of problem rather than the instance. A
regression run should fail because the code broke — never because of collection order.

It costs real time. Each process re-imports the full dependency tree, and collection alone
runs to tens of seconds. So: the runner before you merge, plain `pytest` while you work.

## Two things that deliberately did not move here

**`conftest.py` stays at the repository root.** It sets `ACTIVE_PROFILE`, and the only
moment that can work is before anything imports `backend`. pytest imports the rootdir
conftest before collecting anything; a conftest under `tests/` is loaded too late to
matter. Its own docstring explains this at length — read it before moving it.

**Frontend tests stay with their frontends.** `frontend/` runs `vitest`, and
`sis/frontend/tests/` holds the fixtures its console is checked against. Those belong to a
JavaScript build that knows nothing about pytest. `tests/sis/test_ui_contract.py` and
`tests/sis/test_ui_fixtures.py` read across into them, which is the seam working as
intended: the Python side asserts the console cannot call a route nobody wrote.

## evals is not a test

`evals/retrieval_eval.py` scores retrieval against a **live indexed corpus** — it needs
Milvus up and a key in the environment, and it reports a percentage rather than pass or
fail. Nothing collects it, because a retrieval score that dropped four points is a signal
to investigate, not a build to block.

Run it deliberately:

```bash
python tests/evals/retrieval_eval.py
python tests/evals/retrieval_eval.py --rerank
python tests/run_regression.py --eval-only
```
