"""Load the chat endpoint with many parents at once, and watch what gives first.

RAG_FIX_PLAN item 45: every Phase 6 change is judged against this, before and after.

Each virtual parent owns one conversation and asks questions one after another, reading
the whole streamed reply before asking the next — a parent, not a firehose. A run steps
through rising numbers of parents at once, and for each level reports:

  * TTFT       — request sent to the first answer word on the wire; what a parent feels
  * turn       — request sent to `[DONE]`
  * closed     — request sent to the connection closing, which waits for the save
  * errors     — non-200s, `error` events, timeouts, streams that never reached `[DONE]`
  * turns/s    — completed turns per second across all parents
  * pg peak    — the most Postgres connections the backend held at once, and how many of
                 them sat `idle in transaction` (item 33's signature), sampled from
                 `pg_stat_activity` every 250 ms

Run it against the stub provider (tests/load/stub_provider.py) to measure this
infrastructure alone — the model's latency is then a constant, and anything that moves
between two runs is ours. Run it against the real providers to see what a parent gets.

    # 1. the stub, in its own terminal
    python -m tests.load.stub_provider --port 8900

    # 2. the backend under test, every model and embedding call pointed at the stub
    #    (see `backend_env()` below for the exact variables)

    # 3. the load
    LOAD_USERNAME=... LOAD_PASSWORD=... LOAD_DATABASE_URL=postgresql://... \\
        python -m tests.load.chat_load --levels 1,5,10,20,40 --turns 3

Token: `LOAD_TOKEN` directly, or `LOAD_USERNAME`/`LOAD_PASSWORD`, which log in once
through the identity service. The same token serves every virtual parent — each still has
its own session, which is what the conversation store, the save ordering and the
per-session locks key on. (A per-USER limit, item 38, would need distinct accounts.)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def backend_env(stub: str = "http://127.0.0.1:8900/v1") -> dict:
    """What the backend under test needs so that no call leaves this machine.

    Every chat role resolves `MODEL`/`FAST_MODEL`/`GRADE_MODEL`/`BASE_URL`/`ARK_API_KEY` unless an
    `LLM_PROVIDER` block overrides them, so the provider selector is emptied here — the
    process environment wins over both `.env` files (backend/env.py).
    """
    return {
        "LLM_PROVIDER": "",
        "BASE_URL": stub, "ARK_API_KEY": "stub", "MODEL": "stub-model", "FAST_MODEL": "stub-fast",
        "GRADE_MODEL": "stub-grade",
        "EMBEDDING_BACKEND": "openai", "EMBEDDING_BASE_URL": stub,
        "EMBEDDING_API_KEY": "stub", "EMBEDDING_MODEL": "stub-embed",
        "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false",
    }


#: Routes that end a turn without answering it. A turn on one of these is a failure for
#: the load report, whatever the stream looked like.
DEGRADED_ROUTES = {"retrieval_error", "error", "model_error"}


def questions() -> list[str]:
    from tests.evals.school_dataset import CASES as AR
    from tests.evals.school_dataset_en import CASES as EN

    return [c.question for pair in zip(EN, AR) for c in pair]


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))]


@dataclass
class Level:
    parents: int
    ttft: list[float] = field(default_factory=list)
    turn: list[float] = field(default_factory=list)
    closed: list[float] = field(default_factory=list)
    errors: dict[str, int] = field(default_factory=dict)
    wall: float = 0.0
    pg_peak_total: int = 0
    pg_peak_idle_tx: int = 0

    def fail(self, kind: str) -> None:
        self.errors[kind] = self.errors.get(kind, 0) + 1

    def row(self) -> dict:
        done = len(self.turn)

        def ms(values: list[float], p: float):
            # None, not a crash, when nothing completed — a level where every turn failed
            # is exactly the one whose report must still be printed.
            value = pct(values, p)
            return None if math.isnan(value) else round(value)

        return {
            "parents": self.parents,
            "ttft_p50_ms": ms(self.ttft, 50), "ttft_p95_ms": ms(self.ttft, 95),
            "turn_p50_ms": ms(self.turn, 50), "turn_p95_ms": ms(self.turn, 95),
            "closed_p95_ms": ms(self.closed, 95),
            "turns_per_s": round(done / self.wall, 2) if self.wall else 0.0,
            "completed": done, "errors": dict(self.errors),
            "pg_peak_connections": self.pg_peak_total,
            "pg_peak_idle_in_transaction": self.pg_peak_idle_tx,
        }


class PostgresSampler:
    """Counts the backend's connections by state while a level runs."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._stop = threading.Event()
        self.peak_total = 0
        self.peak_idle_tx = 0
        self._thread: threading.Thread | None = None

    def __enter__(self):
        if not self._url:
            return self
        import psycopg2

        self._conn = psycopg2.connect(self._url)
        self._conn.autocommit = True

        def loop():
            with self._conn.cursor() as cur:
                while not self._stop.is_set():
                    cur.execute(
                        "SELECT count(*), count(*) FILTER (WHERE state = 'idle in transaction') "
                        "FROM pg_stat_activity WHERE datname = current_database() "
                        "AND pid <> pg_backend_pid() AND backend_type = 'client backend'"
                    )
                    total, idle_tx = cur.fetchone()
                    self.peak_total = max(self.peak_total, total)
                    self.peak_idle_tx = max(self.peak_idle_tx, idle_tx)
                    self._stop.wait(0.25)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
            self._conn.close()


async def one_parent(client: httpx.AsyncClient, url: str, token: str, asks: list[str],
                     level: Level, timeout: float) -> None:
    session = f"load-{uuid.uuid4().hex[:12]}"
    headers = {"Authorization": f"Bearer {token}", "X-Thread-ID": session}
    for question in asks:
        started = time.perf_counter()
        first = done = None
        route = ""
        try:
            async with client.stream("POST", f"{url}/chat/stream", headers=headers, timeout=timeout,
                                     json={"message": question, "session_id": session}) as response:
                if response.status_code != 200:
                    level.fail(f"http_{response.status_code}")
                    await response.aread()
                    continue
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        done = time.perf_counter()
                        continue
                    try:
                        event = json.loads(payload)
                    except ValueError:
                        continue
                    kind = event.get("type")
                    if kind == "content" and event.get("content") and first is None:
                        first = time.perf_counter()
                    elif kind == "error":
                        level.fail("error_event")
                    elif kind == "trace":
                        route = str((event.get("rag_trace") or {}).get("route") or "")
            closed = time.perf_counter()
        except httpx.TimeoutException:
            level.fail("timeout")
            continue
        except httpx.HTTPError as exc:
            level.fail(type(exc).__name__)
            continue
        if done is None:
            level.fail("no_done")
            continue
        if route in DEGRADED_ROUTES:
            # A static apology streams and reaches [DONE] like an answer, and it is fast.
            # Counting it would let a backend failing under load read as a quick one.
            level.fail(f"route_{route}")
            continue
        if first is not None:
            level.ttft.append((first - started) * 1000)
        level.turn.append((done - started) * 1000)
        level.closed.append((closed - started) * 1000)


async def run_level(url: str, token: str, parents: int, turns: int, pool: list[str],
                    database_url: str, timeout: float) -> Level:
    level = Level(parents)
    limits = httpx.Limits(max_connections=parents * 2 + 10, max_keepalive_connections=parents * 2 + 10)
    async with httpx.AsyncClient(limits=limits) as client:
        with PostgresSampler(database_url) as sampler:
            started = time.perf_counter()
            await asyncio.gather(*(
                one_parent(client, url, token,
                           [pool[(p * turns + t) % len(pool)] for t in range(turns)], level, timeout)
                for p in range(parents)
            ))
            level.wall = time.perf_counter() - started
        level.pg_peak_total = sampler.peak_total
        level.pg_peak_idle_tx = sampler.peak_idle_tx
    return level


def get_token(identity: str) -> str:
    token = (os.getenv("LOAD_TOKEN") or "").strip()
    if token:
        return token
    username, password = os.getenv("LOAD_USERNAME"), os.getenv("LOAD_PASSWORD")
    if not (username and password):
        raise SystemExit("set LOAD_TOKEN, or LOAD_USERNAME and LOAD_PASSWORD")
    response = httpx.post(f"{identity}/v1/auth/login", timeout=20,
                          json={"username": username, "password": password})
    response.raise_for_status()
    return response.json()["access_token"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default=os.getenv("LOAD_BACKEND_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--identity", default=os.getenv("LOAD_IDENTITY_URL", "http://127.0.0.1:8200"))
    parser.add_argument("--levels", default="1,5,10,20,40")
    parser.add_argument("--turns", type=int, default=3, help="questions per parent per level")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--label", default="run")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    token = get_token(args.identity)
    pool = questions()
    database_url = os.getenv("LOAD_DATABASE_URL", "")
    rows = []
    print(f"{args.label}: {args.url}, {args.turns} turn(s) per parent")
    print(f"{'parents':>7} {'ttft p50':>9} {'ttft p95':>9} {'turn p50':>9} {'turn p95':>9} "
          f"{'turns/s':>8} {'ok':>5} {'pg conns':>9} {'idle tx':>8}  errors")
    for parents in [int(x) for x in args.levels.split(",")]:
        level = asyncio.run(run_level(args.url, token, parents, args.turns, pool,
                                      database_url, args.timeout))
        row = level.row()
        rows.append(row)
        shown = {k: ("-" if v is None else v) for k, v in row.items()}
        print(f"{parents:>7} {shown['ttft_p50_ms']:>9} {shown['ttft_p95_ms']:>9} "
              f"{shown['turn_p50_ms']:>9} {shown['turn_p95_ms']:>9} {row['turns_per_s']:>8} "
              f"{row['completed']:>5} {row['pg_peak_connections']:>9} "
              f"{row['pg_peak_idle_in_transaction']:>8}  {row['errors'] or ''}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"label": args.label, "levels": rows}, indent=2),
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
