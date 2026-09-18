"""Re-ingest documents that are already in the knowledge base, idempotently.

    uv run python -m backend.indexing.reindex --all
    uv run python -m backend.indexing.reindex "Aurexis_Knowledge_Base_Mock_Egypt.docx"
    uv run python -m backend.indexing.reindex --all --dry-run

A chunking change is only true of documents that were re-ingested after it. There was no
way to do that except re-uploading each file through a running backend, which made
"reindex the corpus" a manual instruction rather than something a person or CI could run
and check. That matters now because the size bound this pipeline relies on is established
at INDEXING: a document still holding pre-split figure chunks hands the grader a chunk far
larger than the bound says exists, which is the failure the bound was added to prevent.

It runs the same steps as the upload route, in the same order and through the same
services, so there is one ingestion path and not two:

    parse (figure enrichment included) -> delete the old version -> parent chunks -> vectors

The delete is what makes it idempotent, and it happens AFTER the parse rather than before,
for the reason `DocumentRemover.remove` documents: enrichment runs inside `load_document`
and commits this filename's `document_assets` rows, so a cleanup afterwards would delete
what the parse just wrote. Milvus is insert-only on an auto-id primary key, so skipping the
delete would not replace the document, it would duplicate every chunk of it.

Nothing here re-extracts an image: extractions are cached on (sha256, profile,
dossier_version) and a reindex changes none of those, so no vision calls are spent.

Exit codes: 0 every document rebuilt, 1 at least one failed, 2 nothing to do.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional, Tuple


def _services():
    """The process container, so this rebuilds through the SAME loader, parent store and
    writer the upload route uses rather than a second set built here."""
    from backend.composition import default_services

    return default_services()


def _source_path(services, filename: str) -> Optional[str]:
    """Where this document's file is, preferring what the index recorded.

    A document is re-ingested from the same bytes it was ingested from, so the file has
    to still exist. `file_path` is stored on every chunk; the upload directory and the
    repository root are checked after it, because a corpus moved between machines keeps
    its filename and loses its absolute path.
    """
    recorded = ""
    try:
        rows = services.milvus.query(
            filter_expr=f"filename == {json.dumps(filename, ensure_ascii=False)}",
            output_fields=["file_path"],
            limit=1,
        )
        recorded = (rows[0].get("file_path") if rows else "") or ""
    except Exception:  # noqa: BLE001 — a missing lookup is not a reason to fail
        recorded = ""

    candidates = [recorded, filename, os.path.join("data", "uploads", filename)]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def reindex(services, filename: str, *, dry_run: bool = False) -> Tuple[bool, str]:
    """Rebuild one document. Returns (ok, message)."""
    path = _source_path(services, filename)
    if path is None:
        return False, "source file not found; re-upload it instead"
    if dry_run:
        return True, f"would reindex from {path}"

    started = time.perf_counter()
    documents = services.document_loader.load_document(path, filename)
    if not documents:
        return False, "parsed to nothing; the old version is left in place"

    parents = [d for d in documents if int(d.get("chunk_level", 0) or 0) in (1, 2)]
    leaves = [d for d in documents if int(d.get("chunk_level", 0) or 0) == 3]
    if not leaves:
        return False, "no retrievable leaf chunks; the old version is left in place"

    # Only once the parse has succeeded, so a failure leaves the corpus as it was.
    services.document_remover.remove(filename, include_assets=False)
    services.parent_chunks.upsert_documents(parents)
    services.milvus_writer.write_documents(leaves)

    figures = sum(1 for d in leaves if d.get("modality") == "figure")
    widest = max((len(d.get("text") or "") for d in leaves), default=0)
    return True, (
        f"{len(parents)} parents, {len(leaves)} leaves ({figures} from figures), "
        f"largest leaf {widest} chars, {time.perf_counter() - started:.1f}s"
    )


def main(argv: Optional[list] = None) -> int:
    # A maintenance run is not a conversation worth recording, and an over-quota LangSmith
    # answers 429 on every call — the reason conftest.py does the same.
    for tracing in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        os.environ.setdefault(tracing, "false")
    from backend.env import load_env

    load_env()

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("filenames", nargs="*", help="documents to rebuild")
    parser.add_argument("--all", action="store_true", help="every indexed document")
    parser.add_argument("--dry-run", action="store_true", help="say what would happen")
    args = parser.parse_args(argv)

    services = _services()

    filenames = list(args.filenames)
    if args.all:
        try:
            services.milvus.init_collection()
            filenames = sorted({
                str(item.get("filename"))
                for item in services.milvus.query(output_fields=["filename"])
                if item.get("filename")
            })
        except Exception as exc:  # noqa: BLE001 — the message matters more than the type
            print(f"!! could not list indexed documents: {exc}")
            return 1
    if not filenames:
        print("nothing to do: name a document, or pass --all")
        return 2

    failures = 0
    for filename in filenames:
        ok, message = reindex(services, filename, dry_run=args.dry_run)
        print(f"{'ok  ' if ok else 'FAIL'}  {filename}: {message}")
        failures += 0 if ok else 1

    print(f"\n{len(filenames) - failures}/{len(filenames)} document(s) rebuilt")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
