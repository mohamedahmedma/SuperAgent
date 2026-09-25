"""Measure a hosted bge-m3 against the local one before switching to it.

Two questions a provider's documentation does not answer, and this does:

1. IS IT THE SAME MODEL, AT FULL PRECISION? Every text is embedded twice — locally, by
   the exact code production runs today (`BAAI/bge-m3`, fp32 on CPU, normalised), and
   through the provider — and the two vectors are compared. Same weights served at fp32
   agree to about 1.0000; a quantised or different model does not. This answers the
   precision question empirically, which matters because most providers do not say
   what they serve. It also answers whether the existing index can be kept: the stored
   vectors came from the local model, so a provider that disagrees with it needs a
   reindex before its query vectors can be trusted against them.

2. HOW LONG DOES A CALL TAKE FROM HERE? Latency is mostly the distance between this
   machine and the provider's region, so it has to be measured where the backend runs.
   Run this ON THE PRODUCTION SERVER, not on a laptop.

Providers are driven through `_RemoteEmbedder`, the class production would use, so the
numbers describe the code that will ship — pooled connections and all.

    python -m tests.load.embedding_provider_check \\
        --provider novita    https://api.novita.ai/v3/openai     baai/bge-m3 NOVITA_API_KEY \\
        --provider deepinfra https://api.deepinfra.com/v1/openai BAAI/bge-m3 DEEPINFRA_API_KEY

Each `--provider` is: a label, the OpenAI-compatible base URL (without `/embeddings`),
the provider's model id, and the NAME of the environment variable holding its key — the
key itself never goes on the command line.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Rules of thumb for bge-m3, not guarantees: fp32 vs fp32 differs only by kernel
# rounding; fp16/bf16 differ in the 4th-5th decimal; an int8 or fp8 quantisation, or a
# different checkpoint, shows in the 3rd.
SAME_WEIGHTS = 0.9999
RETRIEVAL_EQUIVALENT = 0.999


def probe_texts(limit: int) -> dict[str, list[str]]:
    """Real questions and real corpus snippets, in both languages the corpus serves."""
    from tests.evals.school_dataset import CASES as ARABIC
    from tests.evals.school_dataset_en import CASES as ENGLISH

    def spans(cases):
        out = []
        for case in cases:
            for span in case.required or ():
                text = span if isinstance(span, str) else " ".join(map(str, span))
                if len(text) > 12:
                    out.append(text)
        return out

    return {
        "arabic questions": [c.question for c in ARABIC][:limit],
        "english questions": [c.question for c in ENGLISH][:limit],
        "corpus snippets": spans(ENGLISH)[:limit],
    }


def local_embedder():
    """Exactly what `EMBEDDING_BACKEND=local` builds in production."""
    os.environ["EMBEDDING_BACKEND"] = "local"
    from backend.indexing.embedding import _create_dense_embedder

    return _create_dense_embedder()


def remote_embedder(base_url: str, model: str, key_env: str):
    key = (os.getenv(key_env) or "").strip()
    if not key:
        raise SystemExit(f"{key_env} is not set — export the provider's key under that name")
    os.environ["EMBEDDING_BASE_URL"] = base_url
    os.environ["EMBEDDING_MODEL"] = model
    os.environ["EMBEDDING_API_KEY"] = key
    from backend.indexing.embedding import _RemoteEmbedder

    return _RemoteEmbedder()


def norm(vector) -> float:
    return math.sqrt(sum(v * v for v in vector))


def cosine(a, b) -> float:
    na, nb = norm(a), norm(b)
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def percentile(values, p) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))]


def timed(call):
    start = time.perf_counter()
    result = call()
    return (time.perf_counter() - start) * 1000, result


def latency(embedder, texts, *, calls: int, concurrency: int) -> dict:
    """Single-query calls, which is the shape of the per-turn hot path."""
    for text in texts[:3]:  # warm the connection pool and any provider cold start
        embedder.embed_documents([text])

    sequential = [timed(lambda t=t: embedder.embed_documents([t]))[0]
                  for t in (texts * 10)[:calls]]

    errors = 0

    def one(text):
        nonlocal errors
        try:
            return timed(lambda: embedder.embed_documents([text]))[0]
        except Exception:
            errors += 1
            return None

    wall, results = timed(lambda: list(ThreadPoolExecutor(concurrency).map(one, (texts * 10)[:calls])))
    parallel = [r for r in results if r is not None]

    batch_ms, _ = timed(lambda: embedder.embed_documents(texts[:32]))
    return {
        "single_p50_ms": round(statistics.median(sequential), 1),
        "single_p95_ms": round(percentile(sequential, 95), 1),
        "single_max_ms": round(max(sequential), 1),
        f"parallel_{concurrency}_p95_ms": round(percentile(parallel, 95), 1) if parallel else None,
        f"parallel_{concurrency}_wall_ms": round(wall, 1),
        "parallel_errors": errors,
        "batch_of_32_ms": round(batch_ms, 1),
    }


def fidelity(reference: dict, vectors: dict) -> dict:
    report = {}
    for group, ref in reference.items():
        got = vectors[group]
        scores = [cosine(a, b) for a, b in zip(ref, got)]
        report[group] = {
            "min_cosine": round(min(scores), 6),
            "median_cosine": round(statistics.median(scores), 6),
        }
    first = next(iter(vectors.values()))[0]
    report["dimensions"] = len(first)
    report["returned_normalised"] = abs(norm(first) - 1.0) < 1e-3
    worst = min(group["min_cosine"] for key, group in report.items() if isinstance(group, dict))
    report["verdict"] = (
        "same weights at full precision — keep the index" if worst >= SAME_WEIGHTS
        else "retrieval-equivalent (likely fp16/bf16) — keep the index, run the eval once"
        if worst >= RETRIEVAL_EQUIVALENT
        else "NOT the same vectors (quantised or a different model) — reindex and run the eval"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", nargs=4, action="append", required=True,
                        metavar=("LABEL", "BASE_URL", "MODEL", "KEY_ENV"))
    parser.add_argument("--texts", type=int, default=40, help="texts per group")
    parser.add_argument("--calls", type=int, default=40, help="latency calls per mode")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", default="embedding_provider_check.json")
    args = parser.parse_args()

    texts = probe_texts(args.texts)
    everything = [t for group in texts.values() for t in group]

    print("local reference (the model production runs today)…", flush=True)
    local = local_embedder()
    reference = {group: local.embed_documents(items) for group, items in texts.items()}
    local_single = [timed(lambda t=t: local.embed_documents([t]))[0] for t in everything[:20]]
    results = {"local": {"single_p50_ms": round(statistics.median(local_single), 1)}}

    for label, base_url, model, key_env in args.provider:
        print(f"{label}…", flush=True)
        embedder = remote_embedder(base_url, model, key_env)
        vectors = {group: embedder.embed_documents(items) for group, items in texts.items()}
        results[label] = {
            "fidelity": fidelity(reference, vectors),
            "latency": latency(embedder, everything, calls=args.calls, concurrency=args.concurrency),
        }

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
