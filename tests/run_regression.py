"""Run every suite in the estate, one isolated process each, and report what broke.

## Why a runner rather than `pytest`

`pytest` from the repository root still works and still collects everything. This exists
for the other question — "is the build good?" — where the answer has to survive the way
these suites are written.

Every service suite configures itself by **setting environment variables at import time**,
before the service it tests is first imported: `tests/sis/conftest.py` binds
`SIS_DATABASE_URL` to a temporary file, `tests/identity/conftest.py` points the signing key
somewhere disposable, `tests/records/conftest.py` fixes the issuer and audience. That is
the only moment those values can be set, because the modules under test read them once and
cache them.

In a single process those assignments are global and permanent, so what a suite sees
depends on which suite was collected before it. The long comment at the top of the
repository's root `conftest.py` is about exactly one instance of that hazard, and it exists
because the failure it describes is invisible: every affected test passes on its own.

Running each suite in its own interpreter removes the whole class of problem rather than
the one instance. A regression run should fail because the code broke, never because of
collection order.

The cost is real — each process re-imports the dependency tree, and collection alone is
tens of seconds — so this is the pre-merge check, not the inner loop. Use
`pytest tests/general -q` while you are working.

## Usage

    python tests/run_regression.py                  # every suite
    python tests/run_regression.py sis records      # only these
    python tests/run_regression.py --eval           # suites, then the retrieval eval
    python tests/run_regression.py --eval-only      # just the retrieval eval
    python tests/run_regression.py --single-process # one pytest run, the fast/unsafe way
    python tests/run_regression.py --list
    python tests/run_regression.py sis -- -k guardian -x   # after --, pytest's own flags

Exit status is 0 only if every selected suite passed. Anything else is a build that
should not ship.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS = REPO_ROOT / "tests"

#: Suite name -> directory. Ordered cheapest-first, so a broken build usually reports in
#: seconds rather than after the slowest suite has finished.
SUITES: dict[str, Path] = {
    "schoolauth": TESTS / "schoolauth",
    "records": TESTS / "records",
    "identity": TESTS / "identity",
    "sis": TESTS / "sis",
    "general": TESTS / "general",
}

#: Not a pytest suite. It scores retrieval against a live indexed corpus, so it needs
#: Milvus up and a key in the environment, and it reports a number rather than a verdict.
#: Opt in with --eval; it is never part of the default run.
EVAL = TESTS / "evals" / "retrieval_eval.py"


def _run(name: str, args: list[str]) -> tuple[str, int, float]:
    print(f"\n{'=' * 70}\n== {name}\n{'=' * 70}", flush=True)
    started = time.monotonic()
    completed = subprocess.run(args, cwd=REPO_ROOT)
    return name, completed.returncode, time.monotonic() - started


def _report(results: list[tuple[str, int, float]]) -> int:
    print(f"\n{'=' * 70}\n== regression summary\n{'=' * 70}")
    width = max(len(name) for name, _, _ in results)
    for name, code, seconds in results:
        verdict = "PASS" if code == 0 else f"FAIL ({code})"
        print(f"  {name:<{width}}  {verdict:<10} {seconds:6.1f}s")

    failed = [name for name, code, _ in results if code != 0]
    total = sum(seconds for _, _, seconds in results)
    print(f"\n  {len(results) - len(failed)}/{len(results)} passed in {total:.1f}s")
    if failed:
        print(f"  failed: {', '.join(failed)}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the estate's suites as a cross-service regression.",
    )
    parser.add_argument("suites", nargs="*", help=f"subset of: {', '.join(SUITES)}")
    parser.add_argument("--eval", action="store_true", help="also run the retrieval eval")
    parser.add_argument("--eval-only", action="store_true", help="run only the retrieval eval")
    parser.add_argument(
        "--single-process",
        action="store_true",
        help="one pytest run over everything. Faster, and vulnerable to the cross-suite "
        "environment leakage this runner exists to prevent.",
    )
    parser.add_argument("--list", action="store_true", help="list the suites and exit")

    # Split on `--` by hand. argparse cannot do this here: a `nargs="*"` positional
    # swallows every remaining token, so a REMAINDER argument after it never sees one and
    # `run_regression.py sis -- -k foo` fails claiming `-k` is an unknown suite.
    argv = sys.argv[1:]
    if "--" in argv:
        cut = argv.index("--")
        argv, passthrough = argv[:cut], argv[cut + 1:]
    else:
        passthrough = []
    options = parser.parse_args(argv)

    if options.list:
        for name, path in SUITES.items():
            print(f"  {name:<12} {path.relative_to(REPO_ROOT).as_posix()}")
        print(f"  {'evals':<12} {EVAL.relative_to(REPO_ROOT).as_posix()}  (--eval)")
        return 0

    if options.eval_only:
        return _report([_run("evals", [sys.executable, str(EVAL), *passthrough])])

    unknown = [name for name in options.suites if name not in SUITES]
    if unknown:
        parser.error(f"unknown suite(s): {', '.join(unknown)}. Choose from: {', '.join(SUITES)}")

    selected = {name: SUITES[name] for name in (options.suites or SUITES)}

    if options.single_process:
        paths = [str(path) for path in selected.values()]
        results = [_run("all suites", [sys.executable, "-m", "pytest", *paths, "-q", *passthrough])]
    else:
        results = [
            _run(name, [sys.executable, "-m", "pytest", str(path), "-q", *passthrough])
            for name, path in selected.items()
        ]

    if options.eval:
        results.append(_run("evals", [sys.executable, str(EVAL), *passthrough]))

    return _report(results)


if __name__ == "__main__":
    raise SystemExit(main())
