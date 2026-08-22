"""The signed-in parent's children, read once and cached for the conversation.

Today the roster is fetched inside `get_student_records`, before any branch, on every
invocation — up to four times in one turn under the tool's own budget. Worse, it is
reachable *only* from that tool, so a question that depends on the child but is answered
from the knowledge base ("what are the fees for my son's year?") has no way to learn
which child, or which year they are in.

This module lifts the read out of the tool so both can use it, and puts a short cache in
front so lifting it out does not multiply the traffic.

## Why this may be cached when the authorization behind it may not

`records/guardian_directory.py` refuses to cache, and says why: *"A registrar revoking
access the minute a court order arrives must take effect on the next question, and a
cached 'yes' would keep answering for as long as the entry lived."* That reasoning is
correct and it does not apply here, because the two are not the same object.

That class caches a **decision**. This caches a **hint** — which children to consider,
and what to call them. Nothing here authorises anything: every records read is re-checked
against the guardian link by the service that answers it. A stale entry can therefore
name a child the parent may no longer read, and the read then fails closed at the facade
with `not_authorized`. The failure mode of a stale hint is a refusal, not a disclosure.

Keeping that true is the whole licence for this file. If anything downstream ever treats
the presence of a child in this list as permission, the cache has to go.

## Keyed by guardian, not by user

`guardian_external_id` is indexed-but-not-unique in identity, so two accounts may
legitimately resolve to one guardian and should share the entry. And an administrator can
rebind an account to a different guardian (`identity/routes.py:342`) — a user-keyed entry
would survive that rebind and be wrong-family; a guardian-keyed one moves with the
binding.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple
from urllib.parse import quote

import requests

from backend.infra.cache import cache

logger = logging.getLogger(__name__)

# Same service and the same credentials the records tool uses; each module reads the
# environment itself rather than importing the other, because `backend.tools.records`
# already imports from `backend.chat` and reversing that would be a cycle.
BASE_URL = os.getenv("RECORDS_BASE_URL", "http://localhost:8100").rstrip("/")
API_KEY = os.getenv("RECORDS_API_KEY", "")

# Short enough that a registrar's change is visible within a question or two, long
# enough to cover a conversation's worth of turns. Passed explicitly — the cache's own
# 300 s default was chosen for corpus chunk text, which does not change under anybody.
#
# Zero or less turns the cache OFF, for both reading and writing. That is an operational
# switch worth having — a deployment that suspects it is serving a stale roster can stop
# it without a deploy — and it is what keeps a test suite from depending on whether a
# Redis happened to be running on the machine.
ROSTER_TTL_SECONDS = int(os.getenv("CHILD_ROSTER_TTL_SECONDS") or 90)
# Deliberately shorter than the records tool's 8 s. This read happens before the agent
# is built, in the one stretch of a streamed turn that emits nothing, and a hint that
# arrives late is worth less than one that does not arrive at all.
ROSTER_TIMEOUT_SECONDS = float(os.getenv("CHILD_ROSTER_TIMEOUT_SECONDS") or 3)

OK = "ok"
NONE = "none"
UNAVAILABLE = "unavailable"
# Kept distinct from `unavailable`, and the distinction is the whole point: telling a
# parent whose sign-in expired that "records are temporarily unavailable" sends them
# away to wait for a service that is working fine. These strings are the outcome names
# `tools/records_result.j2` already branches on, so the tool relays them unchanged.
NOT_AUTHORIZED = "not_authorized"


@dataclass(frozen=True, slots=True)
class ChildOption:
    """One child this guardian may be told about, as far as the facade is concerned."""

    student_id: str
    #: Arabic-first, chosen here rather than in a template so that no second place can
    #: choose differently and name the same child two ways in one turn.
    label: str
    #: "male" | "female" | "unknown". Everything is `unknown` until the registrar
    #: uploads it — see Phase 7. `unknown` must never select a child on its own.
    gender: str = "unknown"
    #: The child's year group, when the facade reports one. Empty is the normal case
    #: today; SIS does not carry it on this route yet.
    year_level: str = ""


def _ttl() -> int:
    """Read at call time, not import time, so the switch can be flipped in a running
    process and so a test can turn the cache off around one case."""
    try:
        return int(os.getenv("CHILD_ROSTER_TTL_SECONDS") or ROSTER_TTL_SECONDS)
    except ValueError:
        return ROSTER_TTL_SECONDS


def _cache_key(guardian_id: str) -> str:
    """One variable segment, placed last, percent-quoted.

    `chat_messages:{user_id}:{session_id}` uses `:` as both the prefix separator and a
    field separator over an unconstrained username, which is a collision waiting to be
    found. A new key does not get to repeat that.
    """
    return f"guardian_students:{quote(guardian_id, safe='')}"


def _fetch(guardian_id: str, token: str, request_id: str) -> Tuple[str, list]:
    """One GET against the facade. Never raises."""
    try:
        response = requests.get(
            f"{BASE_URL}/v1/guardians/{quote(guardian_id, safe='')}/students",
            headers={
                "X-API-Key": API_KEY,
                "Authorization": f"Bearer {token}",
                "X-Request-Id": request_id or "",
            },
            timeout=ROSTER_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        logger.warning("child roster unreachable: %s", exc)
        return UNAVAILABLE, []

    if response.status_code in (401, 403):
        # The session is not authorised for this guardian — most often an expired
        # sign-in. Neither "no children" nor "the service is down", and saying either
        # would send the parent to do the wrong thing about it.
        logger.warning("child roster rejected identity: %s", response.status_code)
        return NOT_AUTHORIZED, []
    if response.status_code == 404:
        return NONE, []
    if response.status_code != 200:
        logger.warning("child roster refused: %s", response.status_code)
        return UNAVAILABLE, []
    try:
        return OK, response.json().get("students") or []
    except ValueError:
        return UNAVAILABLE, []


def _as_options(rows: Sequence[dict]) -> List[ChildOption]:
    options: List[ChildOption] = []
    for row in rows or []:
        student_id = str((row or {}).get("student_id") or "")
        if not student_id:
            continue
        label = (
            str(row.get("full_name_ar") or "").strip()
            or str(row.get("full_name_en") or "").strip()
            or student_id
        )
        options.append(
            ChildOption(
                student_id=student_id,
                label=label,
                gender=str(row.get("gender") or "unknown").strip().lower() or "unknown",
                year_level=str(row.get("year_level") or row.get("grade_level") or "").strip(),
            )
        )
    return options


def load_roster(
    ctx, *, fetch: Optional[Callable[[str, str, str], Tuple[str, list]]] = None
) -> Tuple[str, List[ChildOption]]:
    """`(outcome, children)` where outcome is "ok" | "none" | "unavailable".

    Three-valued, and the third value is the point. "This parent has no readable
    children" is a fact the records tool has careful copy for; "the directory did not
    answer" is an outage. Collapsing them tells a parent whose network blipped that the
    school has no record of their child.

    Never raises. `fetch` is injectable so callers and tests can drive this without a
    network, the same arrangement `resolve_question(invoke=…)` uses.
    """
    guardian_id = getattr(ctx, "guardian_id", "") or ""
    token = getattr(ctx, "guardian_token", "") or ""
    if not guardian_id or not token:
        # Not a parent session. Not an error, and nothing to look up.
        return NONE, []

    ttl = _ttl()
    key = _cache_key(guardian_id)
    if ttl > 0:
        cached = cache.get_json(key)
        if isinstance(cached, list) and cached:
            return OK, _as_options(cached)

    outcome, rows = (fetch or _fetch)(
        guardian_id, token, getattr(ctx, "session_id", "") or ""
    )
    # Written only on a positive, non-empty answer. Caching an outage would turn a
    # three-second blip into ninety seconds of a parent being told nothing is there,
    # and caching an empty list would do the same for any future 200-with-[].
    if ttl > 0 and outcome == OK and rows:
        cache.set_json(key, rows, ttl=ttl)
    if outcome == OK and not rows:
        outcome = NONE
    return outcome, _as_options(rows)


class _Prefetch:
    """A roster read already in flight.

    A bare thread rather than a pooled executor. There is exactly one of these per
    parent turn, it is joined a second or two later by the code that started it, and a
    module-level pool would add a fixed resource with an exhaustion mode — a queue of
    turns waiting on a full pool — in exchange for saving a thread creation that costs
    microseconds against a network call that costs seconds.
    """

    __slots__ = ("_thread", "_result")

    def __init__(self, ctx, fetch=None):
        import threading

        self._result: Tuple[str, List[ChildOption]] = (UNAVAILABLE, [])

        def run() -> None:
            try:
                self._result = load_roster(ctx, fetch=fetch)
            except Exception:  # pragma: no cover - load_roster does not raise
                logger.warning("child roster prefetch failed", exc_info=True)

        self._thread = threading.Thread(
            target=run, name="child-roster-prefetch", daemon=True
        )
        self._thread.start()

    def result(self, timeout: float | None = None) -> Tuple[str, List[ChildOption]]:
        """What came back, or `unavailable` if it has not by now.

        Daemon and never joined beyond the timeout, so a hung facade cannot hold a turn
        open past its own deadline; the thread dies with the process. The timeout here
        is a second ceiling on top of the HTTP one, because a socket that never returns
        does not respect a read timeout.
        """
        self._thread.join(timeout if timeout is not None else ROSTER_TIMEOUT_SECONDS)
        return self._result


def prefetch(ctx, *, fetch=None) -> Optional[_Prefetch]:
    """Begin reading the roster now, to be collected later in the turn.

    Started speculatively, before anything knows whether this turn is about a child,
    because the only moment that can overlap the wait is *before* the classifier runs —
    and by the time the classifier has answered, the answer is already here.

    Speculating is close to free. The read is cached per guardian, so on every turn but
    the first of a conversation this is a Redis lookup; and on the first, it is a call
    the tool would have made anyway a moment later.

    Returns None for anyone who is not a signed-in parent, so a staff session, a
    background job or a test starts no thread and touches no network.
    """
    if not (getattr(ctx, "guardian_id", "") and getattr(ctx, "guardian_token", "")):
        return None
    return _Prefetch(ctx, fetch=fetch)


def forget(ctx) -> None:
    """Drop the cached roster for this guardian.

    Called when the facade refuses a read — direct evidence that what is cached no
    longer matches what the school will authorise. Best-effort by nature: `cache.delete`
    swallows every exception unlogged, so the TTL is the guarantee and this is the
    optimisation. Never `delete_pattern` on a request path — that is `KEYS`, which is
    O(keyspace) and blocks the server.
    """
    guardian_id = getattr(ctx, "guardian_id", "") or ""
    if guardian_id:
        cache.delete(_cache_key(guardian_id))


__all__ = [
    "NONE",
    "prefetch",
    "NOT_AUTHORIZED",
    "OK",
    "ROSTER_TTL_SECONDS",
    "UNAVAILABLE",
    "ChildOption",
    "forget",
    "load_roster",
]
