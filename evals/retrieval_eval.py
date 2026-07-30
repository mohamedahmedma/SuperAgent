"""Retrieval eval against the live KB.

Every gold value below was read out of the indexed chunks, not guessed, so a MISS
here is a real retrieval failure rather than a mislabelled oracle. Gold is a list
of substrings and matching ANY of them counts: several questions are answered by
more than one chunk, and pinning the oracle to one chunk_id would report a false
failure whenever an equally correct chunk was returned instead.

Run:
    .venv/Scripts/python.exe evals/retrieval_eval.py
    .venv/Scripts/python.exe evals/retrieval_eval.py --rerank
    .venv/Scripts/python.exe evals/retrieval_eval.py --top-k 8 --pool 30

Reranked runs are throttled: a Cohere trial key allows 10 calls/minute and
silently degrades to fusion order on 429, which would quietly invalidate the run.
The harness counts 429s and marks the result INVALID rather than reporting a
number that looks fine.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from backend.rag import utils as u  # noqa: E402

# (id, language, query, [acceptable gold substrings])
CASES: list[tuple[str, str, str, list[str]]] = [
    # ---------- factual lookup ----------
    ("location", "en", "Which area is the company in?", ["Heliopolis"]),
    ("expire", "en", "Do barcodes expire?", ["do not expire on their own"]),
    ("batch-expiry", "en", "which barcode type carries batch number and expiry date?", ["GS1-128"]),
    ("status-col", "en", "product status", ["Product Status column"]),

    # ---------- negation / indirect phrasing ----------
    # The answer is a note denying intermediaries; the query asks to use one.
    ("intermediary", "en", "Can I deal with an intermediary instead of GS1?",
     ["does not have any agents"]),
    ("no-agents", "en", "can I subscribe through a local reseller or agent?",
     ["does not have any agents"]),
    # Penalty is a flat 20% — a correct answer must deny the premise.
    ("penalty-grow", "en", "Does the penalty grow if I stay late?",
     ["20% penalty", "grace period"]),
    ("refund", "en", "Can I get a partial refund?", ["non-refundable"]),
    ("license-optional", "ar", "ترخيص المخزن اختياري؟", ["Warehouse license image (optional)"]),

    # ---------- specific value retrieval ----------
    ("freeze-3mo", "en", "What percentage is a 3-month freeze?", ["3 Months Suspension: 25%"]),
    ("freeze-cost", "en", "how much do I pay if I want to pause my account for a year?",
     ["1 Year Suspension"]),
    ("onetrace-cost", "en", "what is the total cost of the farm to consumer traceability system?",
     ["250,000"]),
    ("img-max", "ar", "أقصى صور أقدر أرفعها كام صورة؟", ["maximum 20 images", "Maximum 20 images"]),
    ("img-png", "en", "Can I upload PNG images?", ["JPG, JPEG, or PNG"]),
    ("links-valid", "en", "Do the links need to be correct?", ["must be correct and active"]),

    # ---------- procedural ----------
    ("upgrade", "en", "I ran out of codes in my current package and need more products",
     ["cancel your current package"]),
    ("upgrade-10-50", "ar", "ممكن أرفع باكدج من 10 لـ 50؟", ["10 → 50"]),
    ("bulk-excel", "en", "Can I send an Excel list of products?",
     ["Add Product Group", "Bulk Upload", "download the template"]),
    ("drug-edit", "ar", "ينفع أعدّل دواء بنفسي؟", ["cannot be self-edited"]),
    ("cancel-notice", "en", "How much notice must I give before ending my subscription?",
     ["two months before"]),
    ("payment", "en", "what ways can I pay my renewal invoice?",
     ["Direct bank account debit", "Payment Methods", "Bank transfer"]),

    # ---------- documents / contracting ----------
    ("signatory-id", "en", "Does the authorized signatory need a national ID?",
     ["National ID of the authorized signatory"]),
    ("name-change-fee", "ar", "لو عايز أغير اسم الشركة، في رسوم؟",
     ["Type 1 — No Fees", "No Fees (Non-Legal Changes)"]),

    # ---------- troubleshooting ----------
    ("upa-gln-error", "en", "Does UPA return an error on GLN?", ["Error while uploading GLN code"]),
    ("einvoice", "en", "is electronic invoicing charged separately?", ["not a separate service"]),
    ("imported-bc", "en",
     "I import products that already have international barcodes on them, what should I do?",
     ["GSync"]),
]

MIN_GAP = 7.0  # trial key: 10 req/min


def rank_of(docs: list[dict], gold: list[str]) -> int | None:
    for i, doc in enumerate(docs, 1):
        text = (doc.get("text") or "").lower()
        if any(g.lower() in text for g in gold):
            return i
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true", help="enable the configured reranker")
    ap.add_argument("--top-k", type=int, default=10, help="depth to score (recall@1/4 derived)")
    ap.add_argument("--pool", type=int, default=None, help="override RETRIEVAL_CANDIDATE_K")
    args = ap.parse_args()

    if args.pool:
        u._RETRIEVAL_CANDIDATE_K_RAW = str(args.pool)
    u.RERANK_ENABLED = bool(args.rerank) and u.RERANK_ENABLED
    if args.rerank and not u.RERANK_ENABLED:
        print("!! --rerank requested but RERANK_* env is unset/placeholder; running without it\n")

    pool = u.resolve_candidate_k(args.top_k)[0]
    print(f"cases={len(CASES)}  top_k={args.top_k}  candidate_k={pool}  "
          f"rerank={'ON' if u.RERANK_ENABLED else 'OFF'}\n")

    ranks: dict[str, int | None] = {}
    lats: list[float] = []
    throttled = 0
    last = 0.0

    for cid, lang, query, gold in CASES:
        if u.RERANK_ENABLED:
            wait = MIN_GAP - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.time()
        t0 = time.perf_counter()
        res = u.retrieve_documents(query, top_k=args.top_k)
        lats.append((time.perf_counter() - t0) * 1000)
        if "429" in str(res["meta"].get("rerank_error") or ""):
            throttled += 1
        rank = rank_of(res["docs"], gold)
        ranks[cid] = rank
        mark = f"@{rank}" if rank else "MISS"
        print(f"  [{lang}] {cid:<16} {mark:<6} ({len(res['docs'])} docs, {round(lats[-1])}ms)")

    def recall(at: int, subset=None) -> str:
        items = [(c, ranks[c[0]]) for c in CASES if subset is None or c[1] == subset]
        hit = sum(1 for _, r in items if r and r <= at)
        return f"{hit}/{len(items)} ({hit / len(items):.0%})"

    def mrr(subset=None) -> float:
        items = [ranks[c[0]] for c in CASES if subset is None or c[1] == subset]
        return sum(1 / r for r in items if r) / len(items)

    lats.sort()
    print("\n" + "=" * 58)
    if throttled:
        print(f"!! {throttled}/{len(CASES)} calls RATE-LIMITED -> results INVALID\n")
    print(f"{'':<10}{'recall@1':>12}{'recall@4':>12}{'recall@10':>12}{'MRR':>10}")
    for label, subset in (("overall", None), ("english", "en"), ("arabic", "ar")):
        print(f"{label:<10}{recall(1, subset):>12}{recall(4, subset):>12}"
              f"{recall(10, subset):>12}{mrr(subset):>10.3f}")
    print(f"\nlatency  p50={round(lats[len(lats) // 2])}ms  p95={round(lats[int(len(lats) * .95)])}ms")

    missed = [c for c, r in ranks.items() if r is None]
    if missed:
        print(f"\nnot retrieved at all: {', '.join(missed)}")
    return 1 if throttled else 0


if __name__ == "__main__":
    raise SystemExit(main())
