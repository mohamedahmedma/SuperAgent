"""The Jinja2 environment, the context every page gets for free, and three filters.

Two things in this module are load-bearing and the rest is plumbing.

**Autoescape is on, everywhere.** Student names come out of a spreadsheet a school
secretary typed, and a name containing `<` is a name, not markup. With autoescape on it
renders as itself; with it off — or with a `| safe` somebody added to fix an unrelated
display bug — it becomes markup, and a roster file is an injection vector into the
registrar's own browser, where the session cookie for the whole student register lives.
**Never apply `| safe` to anything that came from a spreadsheet, a form or the database.**
The only values in this UI that may legitimately be marked safe are string constants
written in Python, and there are none.

**The `grade` filter is invariant 1.** A blank mark renders as an em dash and an earned
zero renders as `0%`, and the filter exists so that no template ever writes
`{{ g.percentage or 0 }}` — which reads a real zero as missing, then prints it as a
failure the child did not have. Every mark on every screen goes through this one
function, so the rule is enforced once instead of trusted to each page.

`StrictUndefined` is deliberate. Jinja's default renders a misspelled variable as empty
string, so `{{ student.full_name_ar }}` mistyped as `full_name_er` produces a register
with a blank name column and no error anywhere — a page that is silently wrong looks
exactly like a page whose data is missing. Here it raises instead. A genuinely optional
value is written `{{ maybe | default('—') }}` or guarded with `{% if x is defined %}`,
both of which say out loud that absence was expected.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlencode

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from starlette.responses import HTMLResponse
from starlette.requests import Request

from sis.api import deps as api_deps
from sis.domain.grades import SubjectGrade
from sis.domain.imports import RowOutcome
from sis.domain.structure import AcademicYear
from sis.domain.value_objects import Percentage
from sis.ui import deps as ui_deps

logger = logging.getLogger(__name__)

HERE: Final[Path] = Path(__file__).resolve().parent
TEMPLATES_DIR: Final[Path] = HERE / "templates"
STATIC_DIR: Final[Path] = HERE / "static"

NOT_GRADED: Final[str] = "—"
"""The em dash that means "no mark has been entered". Never `0`, never blank, never `-`.

A distinct glyph on purpose: an empty cell reads as a rendering fault and sends the
registrar to check whether the page loaded, while an em dash is a positive statement that
the school has nothing recorded for this subject yet.
"""

_YEAR_ATTR: Final[str] = "_sis_current_year"


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def render_grade(
    value: object, precision: int = 1, blank: str = NOT_GRADED
) -> str:
    """Render a mark. `None` becomes an em dash; a stated zero becomes `0%`.

    The single most important function in this package, and the reason it takes the
    grade object rather than a number: the null-versus-zero decision is made once, here,
    against a type that distinguishes them, instead of at forty call sites that each have
    to remember. `SubjectGrade.percentage` is `None` for "not marked yet" and
    `Percentage(0)` for "the teacher awarded nothing", and those are different facts
    about a child — one is an absence of information, the other is a result. A parent
    told the first as the second is told their daughter failed an exam that has not been
    marked, and acts on it.

    `or` is what makes that mistake, which is why it appears nowhere in this function:
    `Percentage(0)` is a truthy object today, but `0.0` is falsy, and any refactor that
    unwraps the value turns `x or blank` into a zero rendered as "not graded". Every test
    below is against `None` explicitly.

    Rounding is presentation only and happens here rather than in the domain, where
    `Percentage` documents that a stated figure is stored exactly as given. `66.66667`
    prints as `66.7%`; the stored number is untouched.

    An unrecognised type raises rather than falling back to `blank`. A silent em dash
    would report "no mark" for a value that is in fact a mark this function did not know
    how to read, which is the exact failure the whole filter exists to prevent.
    """
    percentage = _percentage_of(value)
    if percentage is None:
        return blank
    rounded = round(percentage, precision)
    # `:g` drops a trailing `.0`, so a whole number prints as `90%` and not `90.0%`,
    # while `66.7%` keeps the digit that matters.
    return f"{rounded:g}%"


def _percentage_of(value: object) -> float | None:
    """Unwrap the several shapes a mark arrives in. `None` means genuinely not graded."""
    if value is None:
        return None
    if isinstance(value, SubjectGrade):
        # Goes through the domain object's own accessor rather than truth-testing it.
        return None if value.percentage is None else value.percentage.value
    if isinstance(value, Percentage):
        return value.value
    # `GradeLine` and anything else pairing a mark with its subject. Duck-typed rather
    # than imported so a future read-model wrapper works without editing this file.
    inner = getattr(value, "grade", None)
    if isinstance(inner, SubjectGrade):
        return None if inner.percentage is None else inner.percentage.value
    if isinstance(value, bool):
        # Before the int check: `True` is an int in Python and would render as `1%`.
        raise TypeError("the `grade` filter was given a bool, which is not a mark")
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    raise TypeError(
        f"the `grade` filter cannot render {type(value).__name__}; pass a SubjectGrade, "
        "a GradeLine, a Percentage, a number, or None for a mark that has not been "
        "entered"
    )


_ARABIC_MONTHS: Final[tuple[str, ...]] = (
    "يناير",
    "فبراير",
    "مارس",
    "أبريل",
    "مايو",
    "يونيو",
    "يوليو",
    "أغسطس",
    "سبتمبر",
    "أكتوبر",
    "نوفمبر",
    "ديسمبر",
)


def render_date_ar(value: object, blank: str = NOT_GRADED) -> str:
    """A date as a registrar reads it: `13 أغسطس 2026`. A datetime adds the time.

    Gregorian months with Arabic names, because that is the calendar a school term is
    scheduled on here while the language of the office is Arabic. Written out rather than
    numbered: `03/04/2026` is the fourth of March to one reader and the third of April to
    the next, and a term boundary read a month wrong files a whole term of marks under
    the wrong report.

    Digits stay Western (`2026`, not `٢٠٢٦`). Arabic-Indic digits render correctly but
    stop matching anything the registrar types into a search box or pastes into a
    spreadsheet, and every other number on these screens — a student number, a
    percentage — is Western already.

    An aware datetime is converted to UTC and labelled, rather than quietly shown in the
    server's timezone. "Committed at 14:32" with no zone is the kind of detail that gets
    trusted during an incident and is wrong by three hours.
    """
    if value is None:
        return blank
    if isinstance(value, datetime):
        moment = value.astimezone(UTC) if value.tzinfo is not None else value
        stamp = f"{_arabic_date(moment.date())} — {moment:%H:%M}"
        return f"{stamp} UTC" if value.tzinfo is not None else stamp
    if isinstance(value, date):
        return _arabic_date(value)
    raise TypeError(
        f"the `date_ar` filter cannot render {type(value).__name__}; pass a date, a "
        "datetime, or None"
    )


def _arabic_date(day: date) -> str:
    return f"{day.day} {_ARABIC_MONTHS[day.month - 1]} {day.year}"


_OUTCOME_BADGES: Final[Mapping[str, str]] = {
    RowOutcome.CREATED.value: "text-bg-success",
    RowOutcome.UPDATED.value: "text-bg-primary",
    RowOutcome.UNCHANGED.value: "text-bg-secondary",
    # Not a defect the registrar has to fix — a blank line, a totals row — so it is
    # visibly quieter than a rejection. Colouring the two alike makes a clean file look
    # broken and sends somebody hunting for errors that were never errors.
    RowOutcome.SKIPPED.value: "text-bg-light border",
    RowOutcome.REJECTED.value: "text-bg-danger",
}


def render_outcome_badge(value: object) -> str:
    """The Bootstrap badge classes for one row outcome. Returns classes, never markup.

    A class string rather than a rendered `<span>` so the template keeps control of the
    element, its text and its `title` — and, more to the point, so nothing here is ever
    marked safe. Every value comes from this fixed table; an unknown outcome falls back
    to the neutral badge instead of interpolating whatever it was handed.
    """
    key = value.value if isinstance(value, RowOutcome) else str(value)
    return _OUTCOME_BADGES.get(key, "text-bg-secondary")


# ---------------------------------------------------------------------------
# Globals available in every template
# ---------------------------------------------------------------------------

NAV_ITEMS: Final[tuple[tuple[str, str, str], ...]] = (
    ("dashboard", "Dashboard", f"{ui_deps.UI_PREFIX}/"),
    ("structure", "Structure", f"{ui_deps.UI_PREFIX}/structure"),
    ("roster", "Roster", f"{ui_deps.UI_PREFIX}/roster"),
    ("grades", "Grades", f"{ui_deps.UI_PREFIX}/grades"),
    ("imports", "Imports", f"{ui_deps.UI_PREFIX}/imports"),
)
"""`(key, label, path)`. A page names its `key` as `active_nav`; base.html does the rest."""

_ASSET_VERSIONS: dict[str, str] = {}


def static_url(filename: str) -> str:
    """A URL under `/ui/static`, stamped with the file's own mtime and size.

    The stamp is a cache-buster. Without it a registrar whose browser cached last
    month's `app.css` sees a subtly broken layout that no amount of reloading fixes and
    that nobody else can reproduce. Computed once per file per process; a missing file
    is served unstamped so a typo shows up as a plain 404 rather than a crash in the
    layout template.
    """
    version = _ASSET_VERSIONS.get(filename)
    if version is None:
        try:
            stat = (STATIC_DIR / filename).stat()
            version = f"{int(stat.st_mtime)}-{stat.st_size}"
        except OSError:
            version = ""
        _ASSET_VERSIONS[filename] = version
    base = f"{ui_deps.STATIC_PREFIX}/{filename}"
    return f"{base}?v={version}" if version else base


def query_string(params: Mapping[str, Any] | None = None, **overrides: Any) -> str:
    """Build `?a=1&b=2` from a mapping plus overrides, dropping empties.

    Used by the pagination macro to carry the current filters into the next page's link.
    Rebuilding the query by hand in a template is how a registrar paging through a
    filtered roster silently loses her class filter on page two and reads the wrong list.
    `None` and `""` are dropped so an absent filter leaves no `&class_code=` behind.
    """
    merged: dict[str, Any] = {**(params or {}), **overrides}
    pairs = [
        (key, str(value))
        for key, value in merged.items()
        if value is not None and str(value) != ""
    ]
    return f"?{urlencode(pairs)}" if pairs else ""


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------


def build_environment() -> Environment:
    """The one Jinja environment. A function so a test can build a second one."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        # Not `select_autoescape`: that turns escaping *off* for extensions it does not
        # recognise, and the first `.txt` or extension-less template somebody adds would
        # be unescaped without anyone choosing it. On, unconditionally.
        autoescape=True,
        undefined=StrictUndefined,
        # Whitespace control, so a table of 600 rows is not 600 blank lines of `{% for %}`
        # scaffolding in the source a registrar's IT department may well read.
        trim_blocks=True,
        lstrip_blocks=True,
        auto_reload=False,
    )
    env.filters["grade"] = render_grade
    env.filters["date_ar"] = render_date_ar
    env.filters["outcome_badge"] = render_outcome_badge

    env.globals.update(
        static_url=static_url,
        query_string=query_string,
        nav_items=NAV_ITEMS,
        not_graded=NOT_GRADED,
        ui_prefix=ui_deps.UI_PREFIX,
        login_path=ui_deps.LOGIN_PATH,
        logout_path=ui_deps.LOGOUT_PATH,
        dashboard_path=ui_deps.DASHBOARD_PATH,
    )
    return env


environment: Final[Environment] = build_environment()


# ---------------------------------------------------------------------------
# The response helper
# ---------------------------------------------------------------------------


def current_academic_year(request: Request) -> AcademicYear | None:
    """The year the registrar has marked current, read once per request.

    Fetched through `QueryService` like every other read in this UI — the footer is not
    a special case that gets to touch a repository. Memoised on the request because
    `base.html` shows it and a page may show it again, and one page render should be one
    query.

    A failure here is logged and answered with `None`. The footer is a label; if the
    database is unreachable the handler's own query will fail on its own terms and
    produce a real error, and a page that would otherwise render should not be replaced
    by a stack trace because the *footer* could not be filled in.
    """
    cached = getattr(request.state, _YEAR_ATTR, False)
    if cached is not False:
        return cached
    year: AcademicYear | None
    try:
        year = api_deps.get_query_service().current_academic_year()
    except Exception:  # noqa: BLE001 - a label must not be able to fail a whole page
        logger.warning("Could not read the current academic year for the page footer")
        year = None
    setattr(request.state, _YEAR_ATTR, year)
    return year


class Templates:
    """Renders a page with the context every page in this UI is entitled to assume.

    The injected values — caller, flashes, active nav item, current academic year — are
    added here rather than by each handler because a handler that forgets one produces a
    page with no navigation highlight, or worse, silently swallows the flash message
    confirming the import the registrar just ran. Injecting centrally makes forgetting
    impossible; a handler passes only the data its own page is about.
    """

    def __init__(self, env: Environment) -> None:
        self._env = env

    def TemplateResponse(  # noqa: N802 - matches Starlette's spelling, for familiarity
        self,
        request: Request,
        name: str,
        context: Mapping[str, Any] | None = None,
        *,
        status_code: int = 200,
        active_nav: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HTMLResponse:
        """Render `name` and return finished HTML.

        `request` comes first, matching current Starlette, so a handler written from
        habit does not silently pass the template name as the request.

        Flashes are taken *and* cleared as part of rendering: `take_flashes` reads what
        arrived, and `write_flashes` deletes the cookie on this response so the message
        cannot reappear on the next page. One message, shown once, which is the whole
        contract of a flash.

        `no-store` on every page. These screens list named children and their marks, and
        an office machine's back button after a logout is a real way for the next person
        at that desk to read them.
        """
        merged: dict[str, Any] = {
            "request": request,
            "caller": getattr(request.state, "caller", None) or _caller_of(request),
            "flashes": ui_deps.take_flashes(request),
            "active_nav": active_nav or "",
            "current_year": current_academic_year(request),
            **(context or {}),
        }
        html = self._env.get_template(name).render(merged)
        response = HTMLResponse(html, status_code=status_code)
        response.headers["Cache-Control"] = "no-store"
        for key, value in (headers or {}).items():
            response.headers[key] = value
        ui_deps.write_flashes(request, response)
        return response


def _caller_of(request: Request) -> Any:
    """The signed-in caller if this request has one, without forcing a redirect.

    `base.html` shows the key prefix and a sign-out control, and the login page renders
    through the same layout while nobody is signed in. Reading the caller optionally lets
    one layout serve both instead of needing a second, near-identical one that would
    drift.
    """
    try:
        return ui_deps.optional_caller(request)
    except Exception:  # noqa: BLE001 - the layout must render even if auth is unavailable
        logger.warning("Could not resolve the caller while rendering a page")
        return None


templates: Final[Templates] = Templates(environment)
"""The instance every router imports: `from sis.ui.templates import templates`."""


__all__ = [
    "NAV_ITEMS",
    "NOT_GRADED",
    "STATIC_DIR",
    "TEMPLATES_DIR",
    "Templates",
    "build_environment",
    "current_academic_year",
    "environment",
    "query_string",
    "render_date_ar",
    "render_grade",
    "render_outcome_badge",
    "static_url",
    "templates",
]
