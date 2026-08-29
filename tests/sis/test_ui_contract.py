"""The console is pinned to routes that actually exist, and to the design it was asked for.

This suite exists because of a specific, expensive failure: the structure page posted to
`POST /v1/academic-years` for as long as that route did not exist, so a registrar could
create nothing and generation refused every request with "no academic year is on file".
Nothing caught it, because the browser code was the one part of the service with no tests
and the Python suite was green throughout.

So these are contract tests, not behaviour tests. They read the console's **source** as text
and compare it against `app.openapi()`, and they read the **build output** to check what is
actually served. They cannot tell you a screen is usable; they can tell you it is calling
something that is not there, which is the failure that actually happened.

Two trees, and the split matters:

`sis/frontend/src` is what a person edits — React components, the one HTTP client, the four
stylesheets. Every house rule below is checked here, because here is where a rule can be
broken on purpose.

`sis/web` is what Vite writes and FastAPI serves. Only two things are asked of it: that it
exists and is mounted, and that it contains the screens the source defines — a stale build
missing a whole view is the deployment version of the bug this file was written for.

Comments are stripped before any pattern check. Several of these files legitimately *discuss*
the forbidden idioms in a comment explaining why they are forbidden, and a test that flagged
its own rationale would be deleted within a week.
"""
import json
import re
from pathlib import Path

import pytest

from sis.app import app

_SIS = Path(__file__).resolve().parents[2] / "sis"

#: What a person edits.
SRC = _SIS / "frontend" / "src"

#: What Vite writes and the app serves.
WEB = _SIS / "web"

#: Base path `api.js` prepends to every request; screen code writes `/structure/classes`.
API_BASE = "/v1"

#: The one file allowed to touch the network.
CLIENT = "api.js"

#: Vite's output directory for content-hashed files.
ASSETS = "assets"


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _sources() -> list[tuple[str, str]]:
    """Every first-party source file in the console, as (relative path, text)."""
    files = sorted(
        path
        for pattern in ("*.js", "*.jsx", "*.css", "*.html")
        for path in SRC.rglob(pattern)
    )
    assert files, f"no console sources found under {SRC}"
    return [(_relative(f, SRC), f.read_text(encoding="utf-8")) for f in files]


def _built() -> list[tuple[str, str]]:
    """Every text file in the build output, as (relative path, text)."""
    files = sorted(
        path
        for pattern in ("*.html", "*.js", "*.css")
        for path in WEB.rglob(pattern)
    )
    assert files, (
        f"nothing is built into {WEB}. Run `npm run build` in sis/frontend — the app serves "
        "this directory, and an empty one is a console that 404s."
    )
    return [(_relative(f, WEB), f.read_text(encoding="utf-8")) for f in files]


def _css(name: str) -> str:
    return _strip_comments((SRC / "styles" / name).read_text(encoding="utf-8"))


def _strip_comments(text: str) -> str:
    """Blank out HTML, JS and CSS comments, preserving line numbers for error messages."""
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

    `${batchId}`, `{id}` and a concatenated variable all collapse to `{}` — the way the UI
    names a path parameter is not the contract, the shape is.
    """
    p = raw.split("?")[0]
    p = re.sub(r"\$\{[^}]*\}", "{}", p)
    p = re.sub(r"\{[^}]*\}", "{}", p)
    if not p.startswith(API_BASE + "/"):
        p = API_BASE + "/" + p.lstrip("/")
    return tuple("{}" if ("{" in s or "$" in s) else s for s in p.strip("/").split("/"))


#: A quoted absolute path that looks like an API call rather than a page or asset.
_PATH_LITERAL = re.compile(
    r"""['"`](/(?:v1/)?(?:academic-years|structure|terms|subjects|imports|classes|students|guardians|schools|admin)[^'"`\s]*)['"`]"""
)


def test_every_ui_path_matches_a_real_route() -> None:
    """The regression that started this file: a screen calling a route nobody wrote."""
    real = _real_routes()
    bad: list[str] = []
    for path, text in _sources():
        body = _join_concatenations(_strip_comments(text))
        for m in _PATH_LITERAL.finditer(body):
            raw = m.group(1)
            if _normalise(raw) not in real:
                line = body[: m.start()].count("\n") + 1
                bad.append(f"{path}:{line} calls {raw!r}, which no route serves")
    assert not bad, "UI calls endpoints that do not exist:\n  " + "\n  ".join(bad)


def _client_calls() -> dict[str, tuple[str, tuple[str, ...]]]:
    """Every method on the client, as `name -> (verb, path segments)`.

    Read out of `api.js` rather than hard-coded, because the whole point is to compare two
    things that can drift: what a screen sends, and what the route it reaches declares.
    """
    text = _join_concatenations(_strip_comments((SRC / CLIENT).read_text(encoding="utf-8")))
    entries = list(re.finditer(r"(?m)^  ([A-Za-z][\w]*): function\s*\(", text))
    calls: dict[str, tuple[str, tuple[str, ...]]] = {}
    for index, entry in enumerate(entries):
        name = entry.group(1)
        body = text[entry.end() : entries[index + 1].start() if index + 1 < len(entries) else len(text)]
        call = re.search(r"return\s+(get|post|postForm|request)\(", body)
        if not call:
            continue
        helper = call.group(1)
        literal = re.search(r"""['"]([^'"]+)['"]""", body[call.end() :])
        if not literal:
            continue
        verb = {"get": "get", "post": "post", "postForm": "post"}.get(helper)
        if verb is None:
            method = re.search(r"""method:\s*['"]([A-Z]+)['"]""", body)
            verb = method.group(1).lower() if method else "get"
        calls[name] = (verb, _normalise(literal.group(1)))
    assert calls, f"no client methods parsed out of {CLIENT}"
    return calls


def _body_properties() -> dict[str, set[str] | None]:
    """For each client method, the keys its route declares — or None when there is no schema.

    None covers a GET, a multipart upload and a POST with no body: three cases where "every
    key must be declared" has nothing to compare against, and guessing would produce a test
    that fails on correct code.
    """
    spec = app.openapi()
    schemas = spec["components"]["schemas"]

    def resolve(ref: str) -> dict:
        return schemas[ref.rsplit("/", 1)[-1]]

    by_shape = {}
    for path, operations in spec["paths"].items():
        shape = tuple("{}" if s.startswith("{") else s for s in path.strip("/").split("/"))
        for verb, operation in operations.items():
            by_shape[(verb, shape)] = operation

    properties: dict[str, set[str] | None] = {}
    for name, (verb, shape) in _client_calls().items():
        operation = by_shape.get((verb, shape))
        if operation is None:
            properties[name] = None
            continue
        content = (operation.get("requestBody") or {}).get("content") or {}
        schema = (content.get("application/json") or {}).get("schema")
        if not schema:
            properties[name] = None
            continue
        if "$ref" in schema:
            schema = resolve(schema["$ref"])
        declared = set((schema.get("properties") or {}).keys())
        properties[name] = declared or None
    return properties


def _object_keys(text: str) -> list[tuple[int, str]]:
    """Top-level keys of every object literal in `text`, as (offset, key).

    Depth-counted, so a nested object contributes its keys at its own depth and an array of
    entries does not leak its element keys into the outer check.

    A key is only accepted where a key can legally begin — straight after `{` or `,`. Without
    that rule the second arm of a ternary reads as one: `capacity: x === '' ? null : 0` offers
    up `null` as a field name, and the test reports a bug in correct code, which is the fastest
    way to get a test deleted.
    """
    keys: list[tuple[int, str]] = []
    depth = 0
    opens_a_key = False
    for m in re.finditer(r"[{},]|\b([A-Za-z_]\w*)\s*:", text):
        token = m.group(0)
        if token == "{":
            depth += 1
            opens_a_key = True
        elif token == "}":
            depth -= 1
            opens_a_key = False
        elif token == ",":
            opens_a_key = True
        else:
            if depth == 1 and opens_a_key:
                keys.append((m.start(), m.group(1)))
            opens_a_key = False
    return keys


def _call_arguments(text: str, open_paren: int) -> str:
    """The text between `(` at `open_paren` and its matching `)`."""
    depth = 0
    for index in range(open_paren, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren + 1 : index]
    return ""


def test_every_request_body_key_is_one_the_route_declares() -> None:
    """A body key the service ignores is a form that silently saves nothing.

    This is the sibling of the route check and catches the same class of bug one level down.
    A screen posting `{name_en: ...}` to a schema that reads `full_name_en` gets a 200 and
    stores an empty name; posting `{class_section_code: ...}` to one that reads `class_code`
    gets a 422 if the field is required and a silent no-op if it is not. Both happened during
    the React rewrite, in code that looked right and reviewed clean — `name_en` **is** the
    correct field for a school, a term and a class section, and is wrong for exactly the two
    things that have a person's name.

    Which is why the expectation is read out of the OpenAPI schema per route rather than
    written down here: a list of banned words would have to encode "correct for a term, wrong
    for a child", and would be wrong again the next time a field is added.
    """
    expected = _body_properties()
    offenders: list[str] = []

    for path, text in _sources():
        if not path.endswith((".js", ".jsx")) or path == CLIENT:
            continue
        body = _strip_comments(text)
        for call in re.finditer(r"\bapi\.([A-Za-z]\w*)\(", body):
            declared = expected.get(call.group(1))
            if declared is None:
                continue
            arguments = _call_arguments(body, call.end() - 1)
            for offset, key in _object_keys(arguments):
                if key in declared:
                    continue
                line = body[: call.end() + offset].count("\n") + 1
                nearest = sorted(
                    (name for name in declared if key in name or name in key), key=len
                )
                hint = f"; did you mean {nearest[0]!r}?" if nearest else ""
                offenders.append(
                    f"{path}:{line} api.{call.group(1)}() sends {key!r}, which its route "
                    f"does not declare{hint}"
                )

    assert not offenders, (
        "a request body key the service does not read is a form that saves nothing:\n  "
        + "\n  ".join(sorted(set(offenders)))
    )


def test_only_the_client_talks_http() -> None:
    """One client, in api.js.

    Three pages once reimplemented key storage against `localStorage` while the client
    correctly used `sessionStorage`, and the structure page carried a whole second client.
    Duplication is how the screens drifted apart from each other and from the service.
    """
    offenders: list[str] = []
    for path, text in _sources():
        if path == CLIENT:
            continue
        body = _strip_comments(text)
        for pattern, why in (
            (r"\bfetch\s*\(", "calls fetch() directly"),
            (r"XMLHttpRequest", "uses XMLHttpRequest"),
            (r"['\"]X-API-Key['\"]", "builds its own X-API-Key header"),
        ):
            for m in re.finditer(pattern, body):
                offenders.append(f"{path}:{body[: m.start()].count(chr(10)) + 1} {why}")
    assert not offenders, f"screens must go through {CLIENT}:\n  " + "\n  ".join(offenders)


def test_nothing_assembles_markup_by_hand() -> None:
    """No `innerHTML`, anywhere.

    The build this replaced rendered every table by concatenating strings into `innerHTML`
    and escaping by hand. Names arrive from uploaded spreadsheets, so anybody who could hand
    a registrar an .xlsx could put a script tag in the `full_name` column, and safety
    depended on nobody ever forgetting the escape at any one of dozens of interpolation
    sites. JSX escapes text by construction, and `dangerouslySetInnerHTML` is the one door
    left — which is why it is named in the list.
    """
    offenders: list[str] = []
    for path, text in _sources():
        body = _strip_comments(text)
        for pattern in (
            r"\.innerHTML",
            r"\.outerHTML",
            r"insertAdjacentHTML",
            r"dangerouslySetInnerHTML",
        ):
            for m in re.finditer(pattern, body):
                offenders.append(f"{path}:{body[: m.start()].count(chr(10)) + 1}")
    assert not offenders, (
        "markup must be built through JSX, never by assigning HTML text:\n  "
        + "\n  ".join(offenders)
    )


def test_the_api_key_is_never_persisted_or_put_in_a_url() -> None:
    """A credential must not outlive the tab or leak into a URL.

    `localStorage` is allowed for batch-id history and for display preferences (ids, a
    colour scheme, a table-versus-tabs choice), so this checks what is being stored rather
    than banning the API outright.
    """
    offenders: list[str] = []
    for path, text in _sources():
        body = _strip_comments(text)
        for m in re.finditer(r"localStorage\.(?:get|set|remove)Item\(\s*([^,)]+)", body):
            key = m.group(1)
            if re.search(r"api[_.]?key|sis\.api", key, re.I):
                offenders.append(
                    f"{path}:{body[: m.start()].count(chr(10)) + 1} stores the key in localStorage"
                )
        for m in re.finditer(r"[?&](?:api_?key|key)=", body, re.I):
            offenders.append(
                f"{path}:{body[: m.start()].count(chr(10)) + 1} puts a key in a query string"
            )
    assert not offenders, "\n  " + "\n  ".join(offenders)


def test_no_external_network_references() -> None:
    """Schools may be offline or filtered; a CDN stylesheet that fails to load is a dead page.

    The **build output** is what this checks, not just the source, because that is where the
    dependency would appear. Bootstrap comes from npm and is bundled into the CSS Vite emits;
    if it were ever swapped for a `<link>` to a CDN, or a font import were added to a
    stylesheet, this is the test that would see it.
    """
    offenders: list[str] = []
    for path, text in _built() + _sources():
        body = _strip_comments(text)
        for m in re.finditer(r"https?://([A-Za-z0-9.-]+)", body):
            host = m.group(1)
            if host in {"localhost", "127.0.0.1"} or host.endswith(".w3.org"):
                continue
            offenders.append(f"{path}:{body[: m.start()].count(chr(10)) + 1} references {host}")
    assert not offenders, "UI must be fully self-hosted:\n  " + "\n  ".join(sorted(set(offenders)))


@pytest.mark.parametrize(
    "pattern,why",
    [
        (r"\.percentage\s*\|\|", "`percentage || x` reports a real 0 as missing"),
        (r"\.percentage\s*\?\?\s*0", "`percentage ?? 0` invents a grade of 0"),
        (r"if\s*\([^)]*\.percentage\s*\)", "`if (percentage)` treats a real 0 as ungraded"),
        (r"\.score\s*\|\|", "`score || x` reports a real 0 as missing"),
        (r"\.points\s*\|\|", "`points || x` reports a real 0 as missing"),
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
            offenders.append(f"{path}:{body[: m.start()].count(chr(10)) + 1}")
    assert not offenders, f"{why} — found at:\n  " + "\n  ".join(offenders)


def test_an_unmarked_register_entry_is_never_read_as_present() -> None:
    """The attendance analogue of the zero rule, and the more dangerous of the two.

    A child nobody marked comes back with `state: null`, which is a third value beside
    present and absent. `state || 'present'` and `state ?? 'present'` both turn "nobody
    looked" into "she was here" — and unlike a wrong grade, nothing downstream contradicts
    it: the register simply reports a full house every morning.
    """
    offenders: list[str] = []
    for path, text in _sources():
        body = _strip_comments(text)
        for pattern in (
            r"state\s*(?:\|\||\?\?)\s*['\"]present['\"]",
            r"state\s*(?:\|\||\?\?)\s*['\"]absent['\"]",
        ):
            for m in re.finditer(pattern, body):
                offenders.append(f"{path}:{body[: m.start()].count(chr(10)) + 1}")
    assert not offenders, (
        "an unmarked child must stay unmarked; defaulting her to a state states a fact the "
        "school did not:\n  " + "\n  ".join(offenders)
    )


def test_the_console_is_built_and_mounted() -> None:
    """The console is reachable and complete, so a rename cannot silently unmount it."""
    mounted = [r for r in app.routes if getattr(r, "path", "").startswith("/ui")]
    assert mounted, "no /ui mount is registered on the app"

    index = WEB / "index.html"
    assert index.is_file(), f"{index} is missing — run `npm run build` in sis/frontend"

    html = index.read_text(encoding="utf-8")
    referenced = re.findall(r"""(?:src|href)=["']\.?/?((?:assets/)[^"']+)["']""", html)
    assert referenced, (
        "index.html references no built asset. A Vite build always emits at least the entry "
        "module; an index.html without one is a hand-edited or half-copied deployment."
    )
    for name in referenced:
        assert (WEB / name).is_file(), f"index.html loads {name}, which is not in {WEB}"

    js = [path for path in (WEB / ASSETS).glob("*.js")]
    assert js, f"no JavaScript in {WEB / ASSETS}"


def test_every_screen_reaches_the_build() -> None:
    """Each view in the source appears in the bundle.

    A view that is written, imported and never built is a tab that renders nothing — the
    deployment-level version of the bug this file exists for. Checked by looking for a
    string only that screen contains, because module names do not survive minification.

    A failure here almost always means the build is stale rather than that a screen is
    broken: `npm run build`, then read it again.
    """
    fingerprints = {
        "School.jsx": "Schools",
        "Level.jsx": "Classes on this rung",
        "Year.jsx": "Add term",
        "Klass.jsx": "Place an existing child",
        "Student.jsx": "Insights",
        "Roster.jsx": "roster",
        "Guardians.jsx": "Guardians",
        "Marks.jsx": "Marks",
        "Batches.jsx": "Batches",
        "AttendancePanel.jsx": "not yet marked",
    }
    bundle = "\n".join(
        text for path, text in _built() if path.endswith(".js") and not path.endswith(".map")
    )
    missing = [
        f"{name} (looked for {needle!r})"
        for name, needle in fingerprints.items()
        if needle not in bundle
    ]
    assert not missing, (
        "these screens are in the source but not in the build — it is stale:\n  "
        + "\n  ".join(missing)
    )


def test_the_entry_point_is_revalidated_and_the_hashed_assets_are_not() -> None:
    """Two cache policies, and getting them the wrong way round is not symmetric.

    A regression test for a bug with no symptom except silence: Starlette's `StaticFiles`
    sends `ETag` and `Last-Modified` and no `Cache-Control`, which lets a browser guess how
    long a file stays fresh and serve it from cache **without asking the server**. A
    stylesheet was edited, the page was refreshed, and the old design came back — because
    the request never left the browser.

    The fix is not one header for everything. `index.html` gets `no-cache` — meaning
    "revalidate before use", not "do not store", so a 304 still costs an empty body — and it
    is enough, because it names the hashed assets and a new build renames them. Those hashed
    files get a year and `immutable`, because a URL that can only ever mean one set of bytes
    is free to keep.
    """
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        entry = client.get("/ui/index.html")
        assert entry.status_code == 200, entry.text
        directive = entry.headers.get("cache-control", "")
        assert "no-cache" in directive, (
            "index.html must be served `no-cache`; without it a browser keeps serving the "
            f"old entry point and never learns the new asset names. Got {directive!r}"
        )

        # The other half: revalidation still short-circuits, so a reload costs a header
        # exchange rather than the file.
        again = client.get(
            "/ui/index.html", headers={"If-None-Match": entry.headers["etag"]}
        )
        assert again.status_code == 304, "`no-cache` is not `no-store`; 304 must still work"
        assert not again.content

        hashed = re.findall(r"""(?:src|href)=["']\.?/?(assets/[^"']+)["']""", entry.text)
        assert hashed, "index.html references no hashed asset"
        for name in hashed:
            response = client.get(f"/ui/{name}")
            assert response.status_code == 200, name
            directive = response.headers.get("cache-control", "")
            assert "immutable" in directive and "max-age=31536000" in directive, (
                f"{name} carries a content hash, so its URL can never mean anything else and "
                f"it should be cached for a year. Got {directive!r}"
            )


def test_the_stylesheet_order_is_declared_once() -> None:
    """The four stylesheets are imported in the one order that works, from main.jsx.

    Bootstrap first so everything after it can override; tokens next because theme.css reads
    them; theme.css before base and sis because it is the file that re-points `--bs-*`. The
    old build expressed this as a list of `<link>` tags with a comment asking the reader not
    to reorder them, which is not a constraint a build can check. This is.
    """
    main = _strip_comments((SRC / "main.jsx").read_text(encoding="utf-8"))
    imports = re.findall(r"""import\s+['"]([^'"]+\.css)['"]""", main)
    order = [name.rsplit("/", 1)[-1] for name in imports]

    assert order[0] == "bootstrap.min.css", (
        f"Bootstrap must be imported first so the theme can override it; got {order}"
    )
    for earlier, later, why in (
        ("tokens.css", "theme.css", "theme.css maps the tokens onto --bs-*"),
        ("theme.css", "base.css", "base.css assumes the --bs-* variables are re-pointed"),
        ("base.css", "sis.css", "sis.css layers on top of the document rules"),
    ):
        assert earlier in order and later in order, f"{earlier} or {later} is not imported"
        assert order.index(earlier) < order.index(later), why


def test_the_token_layer_names_no_component() -> None:
    """`tokens.css` is the file another project copies to inherit this design language.

    That only works while it stays a list of custom properties: the moment it styles a
    `.card`, taking it means taking this console's markup with it. The other half of the
    check is that the layers above it hard-code no colour — one hex literal outside the token
    file is one thing the dark theme cannot reach.
    """
    leaked = re.findall(r"(?m)^\s*\.[a-z][\w-]*\s*[,{]", _css("tokens.css"))
    assert not leaked, f"tokens.css must declare variables, not style components: {leaked}"

    for name in ("theme.css", "base.css", "sis.css"):
        hard_coded = re.findall(r":\s*(#[0-9a-fA-F]{3,8})\b", _css(name))
        assert not hard_coded, (
            f"{name} must take colour from a token so the dark theme reaches it; "
            f"found {sorted(set(hard_coded))}"
        )


def test_the_footer_is_pinned_to_the_bottom_of_the_window() -> None:
    """The shell fills the viewport and the footer takes up the slack.

    This is a regression test for a bug that shipped: the shell used `min-height: 100%`,
    which resolves against `#app` — the mount point, which has no height — so the percentage
    resolved against nothing and the column only grew to fit its content. The footer then sat
    directly under the last card in the middle of a tall window.

    Both halves are asserted because either one alone leaves the bug. Filling the viewport
    without `margin-top: auto` puts the footer at the top of the empty space; the margin
    without a filled viewport has no space to consume.
    """
    css = _css("sis.css")

    shell = re.search(r"\.sis-app\s*\{([^}]*)\}", css)
    assert shell, ".sis-app rule is missing"
    assert re.search(r"min-height:\s*100dvh", shell.group(1)), (
        "`.sis-app` must fill the viewport with `min-height: 100dvh`. A percentage resolves "
        "against the mount point, which has no height, and the footer floats mid-page."
    )

    footer = re.search(r"\.sis-footer\s*\{([^}]*)\}", css)
    assert footer, ".sis-footer rule is missing"
    assert re.search(r"margin-top:\s*auto", footer.group(1)), (
        "`.sis-footer` must claim the leftover space with `margin-top: auto`; a fixed margin "
        "pushes it away from the content instead of down to the window."
    )


def test_responsive_behaviour_is_bootstraps_and_not_hand_rolled() -> None:
    """No breakpoint of our own. Every one comes from Bootstrap, in the markup.

    This is the constraint the Bootstrap migration was for. The console is used mostly from
    phones, and the previous pass tried to meet that with `clamp()` control tokens and a
    handful of custom `@media` blocks — two systems of breakpoints that disagreed with each
    other, so a card would reflow at 768px while the toolbar inside it reflowed at 720px.

    The two `@media` rules that remain are allowed by name, and neither is a breakpoint:
    `prefers-color-scheme` and `prefers-reduced-motion` are user settings, and `print` is a
    different medium rather than a narrower one. A width or height query is the failure.
    """
    offenders: list[str] = []
    for name in ("tokens.css", "theme.css", "base.css", "sis.css"):
        body = _css(name)
        for m in re.finditer(r"@media([^{]*)\{", body):
            query = m.group(1).strip()
            if re.search(r"\b(?:min|max)-(?:width|height)\b|\bwidth\b|\bheight\b", query):
                offenders.append(f"{name}:{body[: m.start()].count(chr(10)) + 1} @media {query}")
    assert not offenders, (
        "responsive behaviour belongs to Bootstrap's grid and utilities, in the markup — a "
        "second set of breakpoints in CSS is a second set that disagrees:\n  "
        + "\n  ".join(offenders)
    )


def test_the_phone_is_the_default_layout() -> None:
    """Every grid column and button row states its narrow case first.

    Mobile-first is not a style here, it is the thing that was asked for: `col-md-6` with no
    `col-12` beside it is a column that is full width by accident rather than by decision,
    and `d-flex` with no `d-grid` under it is a row of buttons that overflows a 360px screen
    instead of stacking.

    Checked over the whole component tree, because one screen written desktop-first is one
    screen a form teacher cannot use in a corridor.
    """
    offenders: list[str] = []
    for path, text in _sources():
        if not path.endswith(".jsx"):
            continue
        body = _strip_comments(text)
        for m in re.finditer(r'className=(?:"([^"]*)"|\{[^}]*?"([^"]*)")', body):
            classes = (m.group(1) or m.group(2) or "").split()
            line = body[: m.start()].count("\n") + 1
            breakpointed = [c for c in classes if re.match(r"col-(?:sm|md|lg|xl)-\d+$", c)]
            if breakpointed and not any(re.match(r"col-\d+$", c) for c in classes):
                offenders.append(
                    f"{path}:{line} has {breakpointed} with no col-N base — the phone case "
                    "is unstated"
                )
    assert not offenders, "the narrow case must be written first:\n  " + "\n  ".join(offenders)


def test_the_accent_is_not_spent_on_ordinary_buttons() -> None:
    """The blue reaches a control through `--action` and through nothing else.

    The rule changed shape when the palette did, so it is worth stating precisely. The accent
    used to be barred from every button; it now draws exactly one of them — the button that
    commits a form — because the panels went grey and a grey fill on a grey panel has no shape.

    What is still barred is the accent reaching a control by any other path. `--action` is the
    single seam, so the design owner can move the committing button to a different colour by
    editing one token; a `.btn-outline-secondary` that names `--accent-600` directly is a blue
    that no longer answers to that token, and by the fourth blue rectangle on the screen the
    reader has stopped seeing any of them.

    Bootstrap's `.btn-primary` hard-codes `#0d6efd` rather than reading `--bs-primary`, so
    overriding the palette alone leaves every button that blue — which is why the theme
    re-points the button's own `--bs-btn-*` variables, and why this test reads that rule
    specifically.
    """
    css = _css("theme.css")

    primary = re.search(r"\.btn-primary\s*\{([^}]*)\}", css)
    assert primary, ".btn-primary rule is missing from theme.css"
    body = primary.group(1)
    assert "var(--action)" in body, (
        "the primary button must be drawn through --action, which is the one token the "
        "committing colour is allowed to come from"
    )
    assert "accent" not in body, (
        "the primary button names the accent directly; it must go through --action, or moving "
        "the action colour stops moving this button"
    )
    assert re.search(r"--bs-btn-bg\s*:", body), (
        "Bootstrap hard-codes .btn-primary's background, so the theme must re-point "
        "`--bs-btn-bg`; setting `--bs-primary` alone leaves the button blue"
    )

    # The two ordinary buttons. Neither may name the accent at all: one is the button for
    # everything that is not a commit, the other is the per-row action in a table, and a
    # column of twenty blue buttons is the exact failure this rule exists for.
    for name in (".btn-outline-secondary", ".btn-quiet"):
        rule = re.search(rf"\{name}\s*\{{([^}}]*)\}}", css)
        assert rule, f"{name} rule is missing from theme.css"
        assert "accent" not in rule.group(1) and "--action" not in rule.group(1), (
            f"{name} is drawn in the accent; the blue belongs to the committing button, links, "
            "focus and the current item, and nothing else"
        )

    assert "accent" in css, (
        "the accent has disappeared from the theme entirely — it is meant to be held back, "
        "not abandoned"
    )


#: The measured contrast of the action pair, and the floor this test holds it to.
#:
#: 4.5:1 is the WCAG AA floor for text of this size, and this pair does not meet it. White on
#: #007AFF measures 4.02:1. That is a deliberate choice and it is also a large improvement on
#: what it replaced: the action colour used to be #8E8E8E carrying a #DADADC label, which
#: measured 2.35:1 and was itself a knowingly-accepted failure.
#:
#: Why the accent is spent here at all, having been kept off controls everywhere else: the
#: panels are grey now, and a grey button on a grey panel has no shape. systemBlue with a white
#: label is the platform's prominent button and is what the design owner asked for.
#:
#: What the floor still buys: the pair cannot get worse without failing. Lightening the fill,
#: greying the label, or swapping either for a neighbour in the ramp all land under 4.0 and are
#: caught. The only thing not asserted is the AA gap that was accepted on purpose.
ACTION_CONTRAST_FLOOR = 4.0

#: What AA would require, kept here so the gap is a number in the file rather than a memory.
AA_FLOOR = 4.5


def test_the_action_colour_is_systemblue_with_its_stated_label() -> None:
    """`--action` is #007AFF, its label is white, and the pair has not drifted.

    Three assertions, because each alone permits a wrong answer. The fill is pinned to the blue
    it was specified as — an earlier pass made it the anchor grey, and before that near-black,
    both of which are different designs. The label is pinned to white, so neither half can move
    while the other stays put. And the ratio between them is measured rather than trusted, so a
    later edit cannot quietly make the label fainter than what was accepted.

    This test asserts a floor rather than WCAG AA on this one pair; see
    `ACTION_CONTRAST_FLOOR` for why, and for what it asserts instead.
    """
    tokens = _css("tokens.css")

    ramp = dict(re.findall(r"(--(?:grey|accent)-\d+|--black)\s*:\s*(#[0-9a-fA-F]{6})", tokens))

    def resolve(name: str) -> str:
        """Follow one `var(--…)` hop to a literal, which is all these tokens ever use."""
        declaration = re.search(rf"{name}\s*:\s*([^;]+);", tokens)
        assert declaration, f"{name} is not defined"
        value = declaration.group(1).strip()
        reference = re.fullmatch(r"var\((--[\w-]+)\)", value)
        if reference:
            resolved = ramp.get(reference.group(1))
            assert resolved, f"{name} points at {reference.group(1)}, which is not in the ramp"
            return resolved.lower()
        return value.lower()

    assert resolve("--action") == "#007aff", (
        f"--action is {resolve('--action')}; the action colour is specified as #007AFF, Apple's "
        "systemBlue. A grey fill is a different design decision and was not the one asked for."
    )
    assert resolve("--action-ink") == "#ffffff", (
        f"--action-ink is {resolve('--action-ink')}; the label on the action colour is "
        "specified as white. Pinned because it is the half of this pair that has now moved "
        "twice, and an unpinned half drifts."
    )

    def luminance(colour: str) -> float:
        def channel(part: str) -> float:
            value = int(part, 16) / 255
            return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

        red, green, blue = channel(colour[1:3]), channel(colour[3:5]), channel(colour[5:7])
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue

    def contrast(one: str, other: str) -> float:
        first, second = luminance(one), luminance(other)
        return (max(first, second) + 0.05) / (min(first, second) + 0.05)

    ratio = contrast(resolve("--action"), resolve("--action-ink"))
    assert ratio >= ACTION_CONTRAST_FLOOR, (
        f"the label on the action colour measures {ratio:.2f}:1, under the "
        f"{ACTION_CONTRAST_FLOOR}:1 this pair was accepted at. It is already below the "
        f"{AA_FLOOR}:1 WCAG AA floor by a deliberate decision, and making it fainter still is "
        "not covered by that decision."
    )


def test_the_action_hover_darkens_so_a_light_label_survives_it() -> None:
    """Hover must not be the state in which the button's own text disappears.

    The direction is a consequence of the label rather than a taste. While the label was
    near-black, `--action-hover` lightened, because darkening a mid tone closes the gap to a
    dark label. The label is white now, so the arithmetic inverts and hover has to darken
    instead. The failure this guards against is precise and nasty: the text fades at the exact
    moment the pointer is on it, which is the one moment nobody screenshots.

    Asserted as "hover is no worse than resting" rather than as a particular shade, so the
    colour stays a design choice while the direction stays a rule.
    """
    tokens = _css("tokens.css")
    # Both ramps: the action trio points into the accent now, and a dict of greys alone would
    # raise a KeyError rather than fail with something a reader can act on.
    ramp = dict(
        re.findall(r"(--(?:grey|accent)-\d+|--black)\s*:\s*(#[0-9a-fA-F]{6})", tokens)
    )

    def luminance(colour: str) -> float:
        def channel(part: str) -> float:
            value = int(part, 16) / 255
            return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

        return (
            0.2126 * channel(colour[1:3])
            + 0.7152 * channel(colour[3:5])
            + 0.0722 * channel(colour[5:7])
        )

    def contrast(one: str, other: str) -> float:
        first, second = luminance(one), luminance(other)
        return (max(first, second) + 0.05) / (min(first, second) + 0.05)

    # Every theme block declares the trio, and each is checked on its own: the toggle's copy of
    # the dark theme is written out separately from the media query's, and a fix applied to one
    # and not the other is exactly the bug that duplication invites.
    blocks = re.findall(
        r"--action:\s*var\((--[\w-]+)\);\s*--action-hover:\s*var\((--[\w-]+)\);\s*"
        r"--action-ink:\s*var\((--[\w-]+)\);",
        tokens,
    )
    assert len(blocks) == 3, (
        f"expected the action trio in all three theme blocks, found {len(blocks)}. The light "
        "theme, the `prefers-color-scheme` block and the `[data-theme=dark]` block each declare "
        "it, and a block missing from this list is a theme nobody is checking."
    )

    for action, hover, ink in blocks:
        resting = contrast(ramp[action], ramp[ink])
        hovered = contrast(ramp[hover], ramp[ink])
        assert hovered >= resting, (
            f"hovering drops the label from {resting:.2f}:1 to {hovered:.2f}:1. The hover fill "
            f"({hover} = {ramp[hover]}) moves toward the label ({ink} = {ramp[ink]}) instead of "
            "away from it, so the button's own text fades as the pointer lands on it."
        )


def test_the_grey_ramp_is_apples_and_anchored_on_the_stated_grey() -> None:
    """#8E8E93 is `--grey-500`, and every step carries Apple's cool cast and no other.

    This replaced a test that asserted the opposite. The ramp used to be strictly neutral —
    channels within two points of each other — on the argument that a tinted grey reads as a
    colour beside white. The design owner asked for Apple's system greys instead, and those are
    tinted by construction: systemGray is #8E8E93, five points of blue over red.

    So the property worth pinning is not "no cast" but "one cast, applied consistently". A step
    that drifts warm, or carries a cast several times deeper than its neighbours, is the one
    that will look wrong in the ramp — and that is what an accidental edit produces.

    Three checks, and each catches a different accident:

      red == green          the cast lives in the blue channel alone, as Apple's does
      blue >= red           no step goes warm
      blue - red <= 6       a cast, not a colour
    """
    tokens = _css("tokens.css")

    anchor = re.search(r"--grey-500\s*:\s*(#[0-9a-fA-F]{6})", tokens)
    assert anchor, "--grey-500 is not defined"
    assert anchor.group(1).lower() == "#8e8e93", (
        f"--grey-500 is {anchor.group(1)}; the ramp is specified to be anchored on #8E8E93, "
        "Apple's systemGray"
    )

    wrong: list[str] = []
    for name, value in re.findall(r"(--grey-\d+|--black)\s*:\s*(#[0-9a-fA-F]{6})", tokens):
        red = int(value[1:3], 16)
        green = int(value[3:5], 16)
        blue = int(value[5:7], 16)
        if red != green:
            wrong.append(f"{name} {value} has a red/green split; the cast belongs to blue alone")
        elif blue < red:
            wrong.append(f"{name} {value} is warm; every step in this ramp leans cool")
        elif blue - red > 6:
            wrong.append(f"{name} {value} is tinted {blue - red} points, deeper than a cast")
    assert not wrong, (
        "the ramp is Apple's system greys; these steps are not:\n  "
        + "\n  ".join(wrong)
    )


def test_the_page_colour_is_one_token_and_every_surface_follows_it() -> None:
    """`--tint` is the page, and the six surfaces on it are mixed from it. Nothing is hand-set.

    This is what makes the page colour a setting rather than a constant. The dialog writes one
    custom property onto `<html>`; if a surface names a ramp step directly it stops following,
    and the result is the failure that makes configurable themes look cheap — a page in the
    colour somebody chose, carrying panels in the colour somebody else chose.

    The derivation is checked by structure rather than by value, because the values are the
    point of being able to change them. What is pinned is that each surface is a `color-mix`
    *of the tint*, and that the three surfaces which are meant to BE the page colour say so
    with a plain `var(--tint)`.

    `--field` is the one to watch. On a grey panel a control has to be the page colour or it is
    invisible; it is in the "is the tint" group for exactly that reason, and a later edit that
    points it at a ramp step would pass every other check in this file.
    """
    tokens = _css("tokens.css")

    light = tokens[: tokens.index("@media (prefers-color-scheme: dark)")]

    for name in ("--canvas", "--field", "--surface-raised"):
        assert re.search(rf"{name}\s*:\s*var\(--tint\)\s*;", light), (
            f"{name} is not `var(--tint)`. These three are the page colour itself: the page, "
            "the fields on it and anything floating above it. A field that is not the page "
            "colour is a field you cannot see the edge of on a grey card."
        )

    for name in ("--surface", "--surface-hover", "--surface-sunken", "--line", "--line-strong"):
        declaration = re.search(rf"{name}\s*:\s*([^;]+);", light)
        assert declaration, f"{name} is not defined"
        value = declaration.group(1)
        assert "color-mix" in value and "var(--tint)" in value, (
            f"{name} is `{value.strip()}`, which does not follow the page colour. Every surface "
            "is the tint mixed toward a grey; one that names a ramp step directly stays put "
            "while the page around it moves."
        )


def test_the_default_page_is_off_white_rather_than_white() -> None:
    """The stated default: near white, with a little black in it. Not #FFFFFF.

    A pure white page and a grey panel are separated by the panel fill alone, and that fill is
    the first thing to disappear on the cheap office monitors this actually runs on. Starting
    the page a couple of points below white gives every surface above it somewhere to sit.

    Asserted as a band rather than as an exact hex, because which off-white is a taste and the
    dialog offers five. What is not a taste is the two ways it can be wrong: pure white, which
    is the value this replaced, and anything dark enough to need the light theme's inks
    rethought.
    """
    tokens = _css("tokens.css")

    light = tokens[: tokens.index("@media (prefers-color-scheme: dark)")]
    declared = re.search(r"--tint\s*:\s*(#[0-9a-fA-F]{6})\s*;", light)
    assert declared, "the light theme does not declare a --tint"
    tint = declared.group(1).lower()

    assert tint != "#ffffff", (
        "the default page is pure white again. The design asks for an off-white with a little "
        "black in it; white is offered in the dialog as one of five papers, not as the default."
    )

    def channel(part: str) -> float:
        value = int(part, 16) / 255
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    luminance = (
        0.2126 * channel(tint[1:3]) + 0.7152 * channel(tint[3:5]) + 0.0722 * channel(tint[5:7])
    )
    assert 0.9 <= luminance < 1.0, (
        f"the default page {tint} has a relative luminance of {luminance:.3f}. It is meant to be "
        "just below white -- far enough down to give the panels somewhere to sit, nowhere near "
        "far enough to change what colour the text on it has to be."
    )

    # Both dark blocks declare one too, or switching appearance leaves the light page's colour
    # on a dark theme.
    dark_blocks = re.findall(r"--tint\s*:\s*#[0-9a-fA-F]{6}\s*;", tokens)
    assert len(dark_blocks) == 3, (
        f"expected a --tint in all three theme blocks, found {len(dark_blocks)}. The light "
        "theme, the `prefers-color-scheme` block and the `[data-theme=dark]` block each declare "
        "the page colour, and a block missing one inherits the wrong theme's page."
    )


def test_one_attribute_carries_the_theme() -> None:
    """The stylesheets watch the attribute the store actually sets, and there is only one.

    This is a regression test for a bug that shipped and hid well. `tokens.css` keyed its two
    dark blocks on `[data-theme="dark"]` while `store.js` wrote `data-bs-theme`, so the console
    palette never switched from the in-app toggle. Nothing looked wrong from either end: on a
    dark machine the `prefers-color-scheme` block supplied the dark palette anyway, and
    Bootstrap's own components read their dark values out of `theme.css`, which was keyed
    correctly. The only broken path was choosing Dark explicitly on a light machine -- dark
    controls on a near-white page -- and no test went down it.

    So what is asserted is that one name is used everywhere: whatever `applyTheme` sets is what
    every stylesheet selects on. `data-bs-theme` is that name because Bootstrap's components
    already watch it and cannot be taught another.
    """
    store = _strip_comments((SRC / "store.js").read_text(encoding="utf-8"))

    written = set(re.findall(r"""setAttribute\(['"]([\w-]+)['"]\s*,""", store))
    themed = {name for name in written if "theme" in name}
    assert themed == {"data-bs-theme"}, (
        f"store.js sets {sorted(themed) or 'no theme attribute'}; the theme travels on "
        "`data-bs-theme` alone, because Bootstrap's own components watch that one and a second "
        "name is a second source of truth for which theme is on."
    )

    for name in ("tokens.css", "theme.css", "base.css", "sis.css"):
        stray = re.findall(r"\[(data-[\w-]*theme)", _css(name))
        wrong = sorted({attribute for attribute in stray if attribute != "data-bs-theme"})
        assert not wrong, (
            f"{name} selects on {wrong}, which nothing sets. The store writes `data-bs-theme`; "
            "a block keyed on any other name is a block that never applies, and it fails "
            "silently because the `prefers-color-scheme` rules cover for it on a dark machine."
        )


def test_appearance_has_exactly_light_and_dark() -> None:
    """Phase 2 exposes two formal modes and no custom palette disguised as more themes."""
    store = _strip_comments((SRC / "store.js").read_text(encoding="utf-8"))
    settings = _strip_comments(
        (SRC / "components" / "Settings.jsx").read_text(encoding="utf-8")
    )

    appearances = re.findall(
        r"value:\s*['\"](light|dark|system)['\"]\s*,\s*label:", settings
    )
    assert appearances == ["light", "dark"]
    assert "setTint" not in store
    assert "currentTint" not in store
    assert "matchMedia" not in store
    assert 'type="color"' not in settings
    assert "Page colour" not in settings


def test_every_literal_ui_sentence_has_an_arabic_translation() -> None:
    """Arabic mode must not silently fall back to English for static interface copy."""
    locale = (SRC / "locale" / "ar.js").read_text(encoding="utf-8")
    translated = set(
        re.findall(r"(?m)^\s*(?:'([^']+)'|\"([^\"]+)\"|([A-Za-z][A-Za-z ]*))\s*:", locale)
    )
    translated_keys = {next(part for part in match if part) for match in translated}

    used: dict[str, list[str]] = {}
    # Machine identifiers and literal CSV headers stay Latin in both directions so they
    # continue to match files and records outside the UI.
    intentionally_latin = {
        "#",
        "NC-2025-2026",
        "percentage",
        "student_number",
        "student_number,full_name_ar,full_name_en",
        "student_number,phone,full_name_ar,full_name_en,relationship,is_primary_contact",
        "student_number,subject_code,percentage",
    }
    literal = re.compile(r"\bt\(\s*(['\"])(.*?)\1", re.DOTALL)
    for path, body in _sources():
        if path.endswith("locale/ar.js"):
            continue
        for match in literal.finditer(_strip_comments(body)):
            key = match.group(2).replace("\\'", "'").replace('\\"', '"')
            used.setdefault(key, []).append(path)

    missing = sorted(
        key for key in used if key not in translated_keys and key not in intentionally_latin
    )
    assert not missing, (
        "Arabic mode falls back to English for these literal UI strings:\n  "
        + "\n  ".join(f"{key} ({', '.join(sorted(set(used[key])))})" for key in missing)
    )


def test_language_switch_controls_document_language_and_direction() -> None:
    store = _strip_comments((SRC / "store.js").read_text(encoding="utf-8"))
    assert "root.setAttribute('lang', lang)" in store
    assert "root.setAttribute('dir', lang === 'ar' ? 'rtl' : 'ltr')" in store

    base = _css("base.css")
    assert '[dir="rtl"] body' in base
    assert "text-align: right" in base


def test_kg_is_the_stage_label_in_arabic_and_english_ui() -> None:
    school = (SRC / "views" / "School.jsx").read_text(encoding="utf-8")
    locale = (SRC / "locale" / "ar.js").read_text(encoding="utf-8")
    assert "{ key: 'garden', label: 'Garden' }" in school
    assert re.search(r"'Garden'\s*:\s*'KG'", locale)
    assert "روضة" not in locale
    assert "رياض الأطفال" not in locale


def test_every_duration_is_scaled_by_one_variable() -> None:
    """Animation timing goes through `--motion-scale`, so one line disables all motion.

    That single lever is what makes the reduced-motion rule a rule rather than a convention
    — no component carries its own `prefers-reduced-motion` branch, so none of them can be
    written without one. A literal `200ms` in a transition is invisible to it.
    """
    offenders: list[str] = []
    for name in ("theme.css", "base.css", "sis.css"):
        body = _css(name)
        for m in re.finditer(r"(?:transition|animation)(?:-duration|-delay)?\s*:[^;]*", body):
            declaration = m.group(0)
            if re.search(r"\b\d+(?:\.\d+)?m?s\b", declaration):
                offenders.append(
                    f"{name}:{body[: m.start()].count(chr(10)) + 1} {declaration.strip()[:70]}"
                )

    tokens = _css("tokens.css")
    for m in re.finditer(r"--dur-[\w-]+\s*:\s*([^;]+)", tokens):
        assert "--motion-scale" in m.group(1), f"duration token not scaled: {m.group(0).strip()}"

    assert not offenders, (
        "use a --dur-* token so --motion-scale governs it:\n  " + "\n  ".join(offenders)
    )


def test_the_build_is_pinned_and_fetches_nothing_at_runtime() -> None:
    """Bootstrap and React are dependencies, not downloads.

    Vendoring was the point of the previous build's `vendor/` directory and it is still the
    point now that npm supplies them: a school behind a filter, or offline, gets a working
    console. This reads `package.json` rather than the bundle because that is where the
    decision is made.
    """
    manifest = json.loads((SRC.parent / "package.json").read_text(encoding="utf-8"))
    deps = manifest.get("dependencies", {})
    for name in ("react", "react-dom", "bootstrap"):
        assert name in deps, f"{name} must be a dependency so it is bundled, not fetched"

    index = (SRC.parent / "index.html").read_text(encoding="utf-8")
    assert "http" not in index.replace("https://www.w3.org", ""), (
        "index.html must reference no external URL; it is the one file a browser reads before "
        "any of our code runs"
    )
