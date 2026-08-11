"""Regression suite for local_schoolapi, against a live Moodle.

    python regression_suite.py --token <TOKEN> [--url http://localhost:8081]

Two kinds of check, and the first is the reason this suite is worth trusting:

DIFFERENTIAL — Moodle computes the same figures independently, through completely
    different code: `gradereport_user_get_grade_items` renders the user report via
    Moodle's own grade tree and formatter, where our plugin reads two aggregate
    queries. If the two disagree, our number is wrong by definition. This catches
    divergence nobody thought to assert, which is exactly the class of bug a
    hand-written expectation misses.

ASSERTED — cases where Moodle has no answer, or gives two. "No register taken" is the
    clearest: mod_attendance's report shows 0.0% while its gradebook push writes null.
    There is no oracle, so the expectation is hand-computed and stated in the seed
    script next to the setup that produces it.

Every scenario is one student whose idnumber names it, so a failure names its own cause.

NOTE ON COST: a web service call against a Moodle on a Windows bind mount takes ~48s.
The suite therefore makes ONE call per student per endpoint and does all comparison in
memory; it does not re-query to check a detail.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import requests

TIMEOUT = 300


class MoodleError(RuntimeError):
    """A web service call that Moodle refused."""


@dataclass
class Result:
    name: str
    passed: bool
    detail: str = ""
    kind: str = "asserted"          # asserted | differential
    seconds: float = 0.0


@dataclass
class Report:
    results: list[Result] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "", kind: str = "asserted",
            seconds: float = 0.0) -> None:
        self.results.append(Result(name, passed, detail, kind, seconds))
        mark = "PASS" if passed else "FAIL"
        suffix = f"  {detail}" if detail else ""
        print(f"  [{mark}] {name}{suffix}")

    @property
    def failed(self) -> list[Result]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> int:
        total = len(self.results)
        failed = self.failed
        differential = [r for r in self.results if r.kind == "differential"]

        print("\n" + "=" * 72)
        print(f"{total - len(failed)}/{total} passed "
              f"({len(differential)} differential against Moodle's own computation)")

        if failed:
            print(f"\n{len(failed)} FAILED:")
            for r in failed:
                print(f"  - {r.name}: {r.detail}")
        print("=" * 72)
        return 1 if failed else 0


class Client:
    """Moodle REST, with the failure mode Moodle actually uses.

    Moodle signals errors with HTTP 200 and an `exception` key, so a naive
    `raise_for_status()` treats every refusal as success.
    """

    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.token = token
        self.calls = 0
        self.total_seconds = 0.0
        self.transport_failures = 0

    def call(self, function: str, **params) -> Any:
        """One REST call, with transport failures normalised to MoodleError.

        Every transport problem — a read timeout, a dropped connection — is converted
        rather than allowed to propagate. A suite that dies on the first slow response
        throws away every result it had already collected, which on a ~48s-per-call
        instance means losing forty minutes of work to one unlucky request. Recording
        it as a failed check and carrying on is strictly more useful.

        One retry, because on this filesystem a slow response is usually a cold cache
        rather than a real fault, and the second attempt is nearly always fast.
        """
        payload = {
            "wstoken": self.token,
            "wsfunction": function,
            "moodlewsrestformat": "json",
            **params,
        }

        last_error: Exception | None = None
        for attempt in (1, 2):
            started = time.monotonic()
            try:
                response = requests.post(
                    f"{self.url}/webservice/rest/server.php", data=payload, timeout=TIMEOUT
                )
                response.raise_for_status()
                body = response.json()
            except requests.RequestException as exc:
                last_error = exc
                self.calls += 1
                self.total_seconds += time.monotonic() - started
                self.transport_failures += 1
                continue
            except ValueError as exc:
                # HTTP 200 carrying something that is not JSON — Moodle does this when
                # it redirects (a host that is not $CFG->wwwroot returns an HTML page).
                last_error = exc
                self.calls += 1
                self.total_seconds += time.monotonic() - started
                raise MoodleError(f"{function}: non-JSON response ({exc})") from exc

            self.calls += 1
            self.total_seconds += time.monotonic() - started

            if isinstance(body, dict) and "exception" in body:
                raise MoodleError(f"{function}: {body.get('errorcode')} — {body.get('message')}")
            return body

        raise MoodleError(f"{function}: transport failure after 2 attempts — {last_error}")

    # -- our plugin ---------------------------------------------------------

    def grades(self, idnumber: str, term: str = "") -> dict:
        return self.call(
            "local_schoolapi_get_student_grades",
            studentidnumber=idnumber, term=term,
        )

    def attendance(self, idnumber: str, term: str = "") -> dict:
        return self.call(
            "local_schoolapi_get_student_attendance",
            studentidnumber=idnumber, term=term,
        )

    # -- the oracle ---------------------------------------------------------

    def moodle_course_percentage(self, courseid: int, userid: int) -> float | None:
        """Moodle's own percentage for a course, from the user report.

        Reads `percentageformatted` on the row where itemtype == 'course'. That string
        is what a teacher sees, produced by `grade_format_gradevalue` after the report
        substitutes the per-student aggregation bounds — a completely different path
        from our two SQL aggregates.
        """
        data = self.call(
            "gradereport_user_get_grade_items", courseid=courseid, userid=userid
        )
        usergrades = data.get("usergrades") or []
        if not usergrades:
            return None

        for item in usergrades[0].get("gradeitems") or []:
            if item.get("itemtype") != "course":
                continue
            raw = (item.get("percentageformatted") or "").replace("%", "").strip()
            if raw in ("", "-"):
                return None
            try:
                return round(float(raw), 2)
            except ValueError:
                return None
        return None


def close_enough(ours: float | None, theirs: float | None, tolerance: float = 0.05) -> bool:
    """Compare two percentages.

    A tolerance rather than equality because Moodle formats its percentage through
    `format_float` for display and we compute ours in SQL — the two can differ in the
    last decimal place without either being wrong. Anything larger is a real divergence.
    """
    if ours is None and theirs is None:
        return True
    if ours is None or theirs is None:
        return False
    return abs(ours - theirs) <= tolerance


def subject_by_course(payload: dict, idnumber_prefix: str = "") -> list[dict]:
    subjects = payload.get("subjects") or []
    if not idnumber_prefix:
        return subjects
    return [s for s in subjects if str(s.get("idnumber", "")).startswith(idnumber_prefix)]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_asserted(report: Report, client: Client, idnumber: str, spec: dict, term: str) -> dict | None:
    """Hand-computed expectations from the seed manifest."""
    scenario_term = spec.get("term", term)
    try:
        payload = client.grades(idnumber, scenario_term)
    except MoodleError as exc:
        report.add(f"{idnumber} :: grades call", False, str(exc)[:120])
        return None

    # found / not found
    if "found" in spec:
        ok = payload.get("found") is spec["found"]
        report.add(f"{idnumber} :: found == {spec['found']}", ok,
                   "" if ok else f"got found={payload.get('found')}")

    subjects = payload.get("subjects") or []

    if "subject_count" in spec:
        ok = len(subjects) == spec["subject_count"]
        report.add(f"{idnumber} :: {spec['subject_count']} subject(s)", ok,
                   "" if ok else f"got {len(subjects)}: "
                                 f"{[s.get('idnumber') for s in subjects]}")

    if "grades_pct" in spec:
        expected = spec["grades_pct"]
        actual = subjects[0].get("percentage") if subjects else None
        ok = close_enough(actual, expected)
        report.add(f"{idnumber} :: grade == {expected}", ok,
                   "" if ok else f"got {actual}  ({spec.get('note', '')})")

    if "pendingcount" in spec and subjects:
        ok = subjects[0].get("pendingcount") == spec["pendingcount"]
        report.add(f"{idnumber} :: pendingcount == {spec['pendingcount']}", ok,
                   "" if ok else f"got {subjects[0].get('pendingcount')}")

    if "academic_pct" in spec:
        expected = spec["academic_pct"]
        academic = (subjects[0].get("academic") or {}) if subjects else {}
        actual = academic.get("percentage")
        ok = close_enough(actual, expected)
        report.add(f"{idnumber} :: academic == {expected}", ok,
                   "" if ok else f"got {actual} unavailable={academic.get('unavailable')!r}")

        # The point of having two figures at all: where attendance is graded they must
        # NOT be the same number. If they coincide, either the academic subtotal is
        # silently falling back to the course total, or the scenario failed to seed.
        overall = subjects[0].get("percentage") if subjects else None
        if actual is not None and overall is not None:
            report.add(f"{idnumber} :: academic differs from course total",
                       not close_enough(actual, overall),
                       f"both {actual} — attendance is not affecting the total")

    return payload


def check_attendance(report: Report, client: Client, idnumber: str, spec: dict, term: str) -> None:
    if not any(k in spec for k in ("attendance_pct", "taken")):
        return

    scenario_term = spec.get("term", term)
    try:
        payload = client.attendance(idnumber, scenario_term)
    except MoodleError as exc:
        report.add(f"{idnumber} :: attendance call", False, str(exc)[:120])
        return

    subjects = payload.get("subjects") or []

    if "attendance_pct" in spec:
        expected = spec["attendance_pct"]
        actual = subjects[0].get("percentage") if subjects else None
        ok = close_enough(actual, expected)
        report.add(f"{idnumber} :: attendance == {expected}", ok,
                   "" if ok else f"got {actual}  ({spec.get('note', '')})")

    if "taken" in spec:
        expected = spec["taken"]
        actual = subjects[0].get("takensessions") if subjects else 0
        ok = actual == expected
        report.add(f"{idnumber} :: takensessions == {expected}", ok,
                   "" if ok else f"got {actual}")


def check_differential(report: Report, client: Client, idnumber: str, payload: dict,
                       userid: int) -> None:
    """Our grade percentage must equal Moodle's, per course.

    The strongest check in the suite: two independent implementations over the same
    data. Ours reads two SQL aggregates; Moodle's renders the user report through its
    grade tree and formatter. Anything that makes them disagree is a bug in ours,
    whether or not anybody predicted it.

    `userid` comes from the seed manifest rather than from our own payload, because the
    plugin deliberately never returns Moodle user ids — and resolving one over the wire
    would cost a ~48s call per student.
    """
    for subject in payload.get("subjects") or []:
        courseid = int(subject["courseid"])
        ours = subject.get("percentage")

        try:
            theirs = client.moodle_course_percentage(courseid, userid)
        except MoodleError as exc:
            report.add(f"{idnumber} :: differential {subject.get('idnumber')}", False,
                       f"oracle unavailable: {str(exc)[:90]}", kind="differential")
            continue

        ok = close_enough(ours, theirs)
        report.add(
            f"{idnumber} :: differential {subject.get('idnumber')}",
            ok,
            f"ours={ours} moodle={theirs}" if not ok else f"both {ours}",
            kind="differential",
        )


def check_isolation(report: Report, client: Client, idnumbers: list[str], term: str) -> None:
    """No response may mention a student other than the one asked for.

    The guarantee core Moodle could not give: `mod_attendance_get_session` returns every
    classmate. Asserted by serialising the whole payload and searching it for any OTHER
    seeded student's idnumber.
    """
    if len(idnumbers) < 2:
        return

    target = idnumbers[0]
    others = set(idnumbers[1:])

    for endpoint, fetch in (("grades", client.grades), ("attendance", client.attendance)):
        try:
            payload = fetch(target, term)
        except MoodleError as exc:
            report.add(f"isolation :: {endpoint}", False, str(exc)[:120])
            continue

        blob = json.dumps(payload)
        leaked = sorted(o for o in others if o and o in blob)
        report.add(
            f"isolation :: {endpoint} mentions no other student",
            not leaked,
            "" if not leaked else f"LEAKED: {leaked[:5]}",
        )


def check_injection(report: Report, client: Client, term: str) -> None:
    """Hostile input must be handled as data, never as a pattern or as SQL."""
    cases = [
        ("%", "a bare LIKE wildcard must not match every student"),
        ("_", "a single-char wildcard must not match a real idnumber"),
        ("' OR '1'='1", "quote injection must be inert"),
        ("../../etc/passwd", "path traversal is just an unknown idnumber"),
        ("", "an empty idnumber is not a student"),
    ]

    for value, why in cases:
        try:
            payload = client.grades(value, term)
        except MoodleError as exc:
            # A refusal is acceptable; a 500 or a SQL error is not.
            report.add(f"injection :: {value!r}", "dml" not in str(exc).lower(),
                       f"{why} — refused: {str(exc)[:70]}")
            continue

        # The only safe answer is "no such student".
        ok = payload.get("found") is False and not (payload.get("subjects") or [])
        report.add(f"injection :: {value!r}", ok,
                   why if ok else f"RETURNED DATA: found={payload.get('found')} "
                                  f"subjects={len(payload.get('subjects') or [])}")

    # Whitespace is REJECTED, not silently trimmed — and that is the better behaviour.
    #
    # Moodle's validate_param() compares a value against clean_param() of itself and
    # throws when they differ, so PARAM_RAW_TRIMMED *validates* that input is already
    # trimmed rather than trimming it. The alternative, PARAM_RAW, would pass "  X  "
    # through to the lookup, match nothing, and return found=false — a padded idnumber
    # would then be indistinguishable from a student who does not exist. An explicit
    # error beats a silent no-match, because the caller can fix the former.
    try:
        padded = client.grades("  GRD-BASELINE  ", term)
        report.add(
            "injection :: padded idnumber is rejected, not silently unmatched",
            False,
            f"expected a refusal; got found={padded.get('found')}",
        )
    except MoodleError as exc:
        rejected = "invalidparameter" in str(exc).lower()
        report.add(
            "injection :: padded idnumber is rejected, not silently unmatched",
            rejected,
            "" if rejected else f"refused for the wrong reason: {str(exc)[:90]}",
        )


def check_term_filter(report: Report, client: Client, term: str) -> None:
    """A term prefix must bound the result, and a wildcard in it must not widen it."""
    try:
        everything = client.grades("GRD-TWO-COURSES", "")
        thisterm = client.grades("GRD-TWO-COURSES", term)
        wildcard = client.grades("GRD-TWO-COURSES", "2026-T1-%")
    except MoodleError as exc:
        report.add("term filter", False, str(exc)[:120])
        return

    report.add("term :: empty prefix returns everything",
               len(everything.get("subjects") or []) >= len(thisterm.get("subjects") or []))

    report.add("term :: prefix narrows to that term",
               all(str(s.get("idnumber", "")).startswith(term)
                   for s in thisterm.get("subjects") or []))

    # '%' is escaped, so it is a literal — nothing has an idnumber containing it.
    report.add("term :: a wildcard in the prefix is escaped, not honoured",
               len(wildcard.get("subjects") or []) == 0,
               f"got {len(wildcard.get('subjects') or [])} subjects")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8081")
    parser.add_argument("--token", required=True)
    parser.add_argument("--manifest", default="", help="path to schoolapi_scenarios.json")
    parser.add_argument("--only", default="", help="run scenarios whose id contains this")
    parser.add_argument("--skip-differential", action="store_true")
    args = parser.parse_args()

    client = Client(args.url, args.token)
    report = Report()

    if not args.manifest:
        print("Pass --manifest pointing at the JSON the seed script wrote "
              "(/var/www/moodledata/schoolapi_scenarios.json inside the container).")
        return 2

    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)

    term = manifest.get("term", "2026-T1-")
    scenarios = manifest.get("scenarios") or {}
    if args.only:
        scenarios = {k: v for k, v in scenarios.items() if args.only in k}

    print(f"{len(scenarios)} scenarios, term prefix {term!r}\n")

    started = time.monotonic()

    print("-- asserted expectations " + "-" * 47)
    payloads: dict[str, dict] = {}
    for idnumber, spec in scenarios.items():
        payload = check_asserted(report, client, idnumber, spec, term)
        if payload:
            payloads[idnumber] = payload
        check_attendance(report, client, idnumber, spec, term)

    if not args.skip_differential:
        print("\n-- differential against Moodle " + "-" * 42)
        for idnumber, spec in scenarios.items():
            if not spec.get("differential"):
                continue
            payload = payloads.get(idnumber)
            userid = int(spec.get("userid") or 0)
            if payload and userid:
                check_differential(report, client, idnumber, payload, userid)

    print("\n-- isolation " + "-" * 59)
    check_isolation(report, client, list(scenarios.keys()), term)

    print("\n-- injection " + "-" * 59)
    check_injection(report, client, term)

    print("\n-- term filtering " + "-" * 54)
    check_term_filter(report, client, term)

    wall = time.monotonic() - started
    print(f"\n{client.calls} web service calls, "
          f"{client.total_seconds:.0f}s in calls, {wall:.0f}s wall")
    if client.transport_failures:
        # Reported separately from assertion failures: a timeout says something about
        # this machine, not about the plugin, and conflating the two would make a slow
        # afternoon look like a regression.
        print(f"{client.transport_failures} transport failure(s) retried")

    return report.summary()


if __name__ == "__main__":
    sys.exit(main())
