"""The UI is pinned to routes that actually exist.

This suite exists because of a specific, expensive failure: the structure page posted to
`POST /v1/academic-years` for as long as that route did not exist, so a registrar could
create nothing and generation refused every request with "no academic year is on file".
Nothing caught it, because the browser code was the one part of the service with no tests
and the Python suite was green throughout.

So these are contract tests, not behaviour tests. They read the shipped HTML and JS as text
and compare it against `app.openapi()`. They cannot tell you a page is usable; they can tell
you it is calling something that is not there, which is the failure that actually happened.

Comments are stripped before any pattern check. Several of these files legitimately *discuss*
the forbidden idioms in a comment explaining why they are forbidden, and a test that flagged
its own rationale would be deleted within a week.
"""
import re
from pathlib import Path

import pytest

from sis.app import app

WEB = Path(__file__).resolve().parents[1] / "web"

#: Base path `api.js` prepends to every request; page code writes `/structure/classes`.
API_BASE = "/v1"


def _sources() -> list[tuple[Path, str]]:
    files = sorted(list(WEB.glob("*.html")) + list(WEB.glob("*.js")))
    assert files, f"no UI sources found under {WEB}"
    return [(f, f.read_text(encoding="utf-8")) for f in files]


def _strip_comments(text: str) -> str:
    """Blank out HTML and JS comments, preserving line numbers for error messages."""
    def blank(m: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", m.group(0))

    text = re.sub(r"<!--.*?-->", blank, text, flags=re.S)
    text = re.sub(r"/\*.*?\*/", blank, text, flags=re.S)
    text = re.sub(r"(?m)//[^\n]*", blank, text)
    return text


def _real_routes() -> set[tuple[str, ...]]:
    """Every declared route as a tuple of path segments, with `{param}` normalised to `{}`."""
    routes = set()
    for path in app.openapi()["paths"]:
        routes.add(tuple("{}" if s.startswith("{") else s for s in path.strip("/").split("/")))
    return routes


def _join_concatenations(text: str) -> str:
    """Rewrite `'/students/' + id + '/grades'` into the single literal `'/students/{}/grades'`.

    Without this the extractor sees the fragment `'/students/'` and reports it as a call to
    a route nobody serves — a false alarm that would teach the next reader to distrust this
    file, which is worse than having no test at all.
    """
    # `' + expr + '` between two literal halves, applied repeatedly for chained joins.
    inner = re.compile(r"""(['"])\s*\+\s*[^+'"]+?\s*\+\s*\1""")
    for _ in range(6):
        text, n = inner.subn("{}", text)
        if not n:
            break
    # A literal that ends the string but is still joined to a trailing expression.
    text = re.sub(r"""(['"])(/[^'"]*?)\1\s*\+\s*[A-Za-z_$][\w$.()]*""", r"\1\2{}\1", text)
    return text


def _normalise(raw: str) -> tuple[str, ...]:
    """A UI path literal as comparable segments.

    `${batchId}`, `{id}` and a concatenated variable all collapse to `{}` — the UI's way of
    naming a path parameter is not the contract, the shape is.
    """
    p = raw.split("?")[0]
    p = re.sub(r"\$\{[^}]*\}", "{}", p)
    p = re.sub(r"\{[^}]*\}", "{}", p)
    if not p.startswith(API_BASE + "/"):
        p = API_BASE + "/" + p.lstrip("/")
    return tuple("{}" if ("{" in s or "$" in s) else s for s in p.strip("/").split("/"))


#: A quoted absolute path that looks like an API call rather than a page or asset.
_PATH_LITERAL = re.compile(r"""['"`](/(?:v1/)?(?:academic-years|structure|terms|subjects|imports|classes|students|admin)[^'"`\s]*)['"`]""")


def test_every_ui_path_matches_a_real_route() -> None:
    """The regression that started this file: a page calling a route nobody wrote."""
    real = _real_routes()
    bad: list[str] = []
    for path, text in _sources():
        body = _join_concatenations(_strip_comments(text))
        for m in _PATH_LITERAL.finditer(body):
            raw = m.group(1)
            if _normalise(raw) not in real:
                line = body[: m.start()].count("\n") + 1
                bad.append(f"{path.name}:{line} calls {raw!r}, which no route serves")
    assert not bad, "UI calls endpoints that do not exist:\n  " + "\n  ".join(bad)


def test_only_api_js_talks_http() -> None:
    """One client, in api.js.

    Three pages once reimplemented key storage against `localStorage` while `api.js`
    correctly used `sessionStorage`, and the structure page carried a whole second client.
    Duplication is how the pages drifted apart from each other and from the service.
    """
    offenders: list[str] = []
    for path, text in _sources():
        if path.name == "api.js":
            continue
        body = _strip_comments(text)
        for pattern, why in (
            (r"\bfetch\s*\(", "calls fetch() directly"),
            (r"XMLHttpRequest", "uses XMLHttpRequest"),
            (r"['\"]X-API-Key['\"]", "builds its own X-API-Key header"),
        ):
            for m in re.finditer(pattern, body):
                offenders.append(f"{path.name}:{body[: m.start()].count(chr(10)) + 1} {why}")
    assert not offenders, "pages must go through SIS in api.js:\n  " + "\n  ".join(offenders)


def test_the_api_key_is_never_persisted_or_put_in_a_url() -> None:
    """A registrar key must not outlive the tab or leak into a URL.

    `localStorage` is allowed for batch-id history (ids, not credentials), so this checks
    what is being stored rather than banning the API outright.
    """
    offenders: list[str] = []
    for path, text in _sources():
        body = _strip_comments(text)
        for m in re.finditer(r"localStorage\.(?:get|set|remove)Item\(\s*([^,)]+)", body):
            key = m.group(1)
            if re.search(r"api[_.]?key|sis\.api", key, re.I):
                offenders.append(f"{path.name}:{body[: m.start()].count(chr(10)) + 1} stores the key in localStorage")
        for m in re.finditer(r"[?&](?:api_?key|key)=", body, re.I):
            offenders.append(f"{path.name}:{body[: m.start()].count(chr(10)) + 1} puts a key in a query string")
    assert not offenders, "\n  " + "\n  ".join(offenders)


def test_no_external_network_references() -> None:
    """Schools may be offline or filtered; a CDN stylesheet that fails to load is a dead page."""
    offenders: list[str] = []
    for path, text in _sources():
        body = _strip_comments(text)
        for m in re.finditer(r"https?://([A-Za-z0-9.-]+)", body):
            host = m.group(1)
            if host in {"localhost", "127.0.0.1"} or host.endswith(".w3.org"):
                continue
            offenders.append(f"{path.name}:{body[: m.start()].count(chr(10)) + 1} references {host}")
    assert not offenders, "UI must be fully self-hosted:\n  " + "\n  ".join(offenders)


@pytest.mark.parametrize(
    "pattern,why",
    [
        (r"\.percentage\s*\|\|", "`percentage || x` reports a real 0 as missing"),
        (r"\.percentage\s*\?\?\s*0", "`percentage ?? 0` invents a grade of 0"),
        (r"if\s*\([^)]*\.percentage\s*\)", "`if (percentage)` treats a real 0 as ungraded"),
        (r"\.score\s*\|\|", "`score || x` reports a real 0 as missing"),
    ],
)
def test_a_real_zero_is_never_confused_with_no_grade(pattern: str, why: str) -> None:
    """Zero is a mark a child can earn; null means nobody has marked the work yet.

    Every idiom below silently merges the two. The visible consequence is a parent being
    told their child scored 0% in a subject that has not been graded, or a genuine zero
    disappearing from a report. Branch on `is_graded`, never on the number.
    """
    offenders: list[str] = []
    for path, text in _sources():
        body = _strip_comments(text)
        for m in re.finditer(pattern, body):
            offenders.append(f"{path.name}:{body[: m.start()].count(chr(10)) + 1}")
    assert not offenders, f"{why} — found at:\n  " + "\n  ".join(offenders)


def test_ui_is_served_by_the_app() -> None:
    """The pages are reachable, so a rename cannot silently unmount the console."""
    mounted = [r for r in app.routes if getattr(r, "path", "").startswith("/ui")]
    assert mounted, "no /ui mount is registered on the app"
    for name in ("index.html", "structure.html", "roster.html", "grades.html", "imports.html", "api.js"):
        assert (WEB / name).is_file(), f"{name} is missing from {WEB}"
