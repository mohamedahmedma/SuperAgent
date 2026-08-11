"""Build or refresh the scope catalogue for the active profile.

    uv run python -m backend.indexing.build_scope_index
    uv run python -m backend.indexing.build_scope_index --dry-run
    uv run python -m backend.indexing.build_scope_index --force

Run after ingesting or editing a corpus. Sections whose text is unchanged are reused,
so a re-run after editing one page costs one model call, and a re-run after editing
nothing costs none.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Sequence

from sqlalchemy import select

from backend.db.models import ParentChunk
from backend.indexing.section_summary import (
    SectionRecord,
    build_corpus_digest,
    corpus_catalogue,
    plan_sections,
    sections_fingerprint,
    summarise_section,
)
from backend.indexing.summary_store import (
    DigestRecord,
    delete_missing,
    existing_hashes,
    load_digest,
    load_records,
    save_digest,
    save_records,
)
from backend.infra.database import SessionLocal
from backend.profiles import get_profile

logger = logging.getLogger(__name__)


def load_sections(level: int) -> List[dict]:
    """The corpus sections to catalogue, from the chunk hierarchy already built.

    Reads Postgres rather than Milvus: the parent tiers live here, and the top of the
    hierarchy is what a section means. Ordered so a partial run is reproducible.
    """
    with SessionLocal() as session:
        rows = session.execute(
            select(ParentChunk)
            .where(ParentChunk.chunk_level == level)
            .order_by(ParentChunk.filename, ParentChunk.chunk_idx)
        ).scalars().all()
    return [
        {
            "chunk_id": row.chunk_id,
            "text": row.text or "",
            "filename": row.filename or "",
            "chunk_level": row.chunk_level,
        }
        for row in rows
    ]


def _structured_invoke(model_name: str, profile):
    """A callable that runs one structured summarisation call at temperature 0.

    Reuses the vision module's cascade, which already handles providers that reject
    `json_schema` by falling back to function calling and then to prompted JSON — the
    same failures would otherwise surface here as an empty catalogue.
    """
    from backend.assets.vision import call_with_rate_limit_retry, invoke_structured
    from langchain.chat_models import init_chat_model
    import os

    model = init_chat_model(
        model=model_name,
        model_provider="openai",
        api_key=os.getenv("ARK_API_KEY"),
        base_url=os.getenv("BASE_URL"),
        # Zero, so the same section yields the same catalogue on every run. Constant
        # behaviour here is not a preference: a drifting scope boundary between
        # re-indexes is untestable and unexplainable.
        temperature=0.0,
    )

    def invoke(prompt: str, schema):
        # Retried on the provider's own terms. A catalogue build is dozens of calls in
        # a row, which is exactly the shape that trips a per-minute token quota — and
        # a section dropped for a 429 would leave a hole in the scope boundary that
        # nothing downstream could detect.
        return call_with_rate_limit_retry(
            lambda: invoke_structured(model, schema, [{"role": "user", "content": prompt}]),
            config=profile.assets,
            description="section summary",
        )

    return invoke


def build(dry_run: bool = False, force: bool = False) -> dict:
    profile = get_profile()
    config = profile.rag
    level = int(getattr(config, "scope_section_level", 1))
    vocabulary = list(getattr(config, "scope_topic_vocabulary", None) or [])

    sections = load_sections(level)
    if not sections:
        logger.error(
            "no chunk_level=%s sections found — ingest a document first, or set "
            "rag.scope_section_level to a level this corpus has", level,
        )
        return {"sections": 0, "summarised": 0, "reused": 0, "removed": 0}

    known = {} if force else existing_hashes(profile.name)
    plan = plan_sections(sections, known)
    logger.info(
        "%d section(s): %d to summarise, %d reused",
        len(sections), len(plan["summarise"]), len(plan["reuse"]),
    )
    if dry_run:
        return {
            "sections": len(sections),
            "summarised": 0,
            "reused": len(plan["reuse"]),
            "removed": 0,
            "would_summarise": len(plan["summarise"]),
        }

    invoke = _structured_invoke(getattr(config, "scope_summary_model", None) or _default_model(), profile)
    records: List[SectionRecord] = []
    for section in plan["summarise"]:
        result = summarise_section(
            section["text"],
            invoke=invoke,
            vocabulary=vocabulary,
            persona=profile.identity.persona,
            languages=getattr(config, "scope_question_languages", ""),
            min_questions=int(getattr(config, "scope_min_questions", 4)),
            max_questions=int(getattr(config, "scope_max_questions", 10)),
        )
        if not result:
            logger.warning("skipped %s — no usable catalogue entry", section["chunk_id"])
            continue
        records.append(
            SectionRecord(
                chunk_id=section["chunk_id"],
                content_sha256=section["content_sha256"],
                filename=section["filename"],
                chunk_level=section["chunk_level"],
                model_used=getattr(config, "scope_summary_model", "") or _default_model(),
                **result,
            )
        )
        logger.info("catalogued %s: %d question(s)", section["chunk_id"], len(result["answers"]))

    _embed_questions(records)
    written = save_records(profile.name, records)
    removed = delete_missing(profile.name, [item["chunk_id"] for item in sections])

    # Over EVERY stored record, not just the ones summarised this run. A section can be
    # current by content hash and still have no vectors — a transient failure on one run
    # leaves the old summary in place, and the hash then says "nothing to do" forever.
    # Vector coverage is its own precondition and has to be checked on its own terms.
    repaired = _repair_missing_vectors(profile.name)

    stored = load_records(profile.name)
    report = verify(stored, expected=len(sections))
    logger.info(
        "catalogue: %d/%d section(s), %d question(s); topics: %s",
        report["with_questions"], len(sections), report["questions"],
        corpus_catalogue(stored) or "none",
    )
    for problem in report["problems"]:
        logger.error("INCOMPLETE: %s", problem)

    digested = _refresh_digest(profile, stored, force=force)

    return {
        "sections": len(sections),
        "summarised": written,
        "reused": len(plan["reuse"]),
        "removed": removed,
        "repaired": repaired,
        "digest": digested,
        "complete": report["complete"],
        "problems": report["problems"],
    }


def _refresh_digest(profile, records: List[SectionRecord], *, force: bool = False) -> str:
    """Write the corpus paragraph and the derived floor, if the corpus moved.

    Both belong to the ingest phase for the same reason the question vectors do: they
    describe the whole corpus, they are expensive, and nothing about them changes
    between requests. Computing them here means a restart reads two columns instead of
    calling a model and running a quadratic pass.

    Skipped when the section set is unchanged, so re-running an untouched corpus still
    costs nothing. `--force` overrides that, which is the way to pick up an edited
    digest prompt without editing the corpus.
    """
    usable = [record for record in records if record.usable]
    if not usable:
        logger.warning("no usable sections; leaving any existing digest alone")
        return "skipped"

    fingerprint = sections_fingerprint(usable)
    existing = load_digest(profile.name)
    if not force and existing.paragraph and existing.sections_sha256 == fingerprint:
        logger.info("corpus digest current (%d section(s)); not rewriting", existing.section_count)
        paragraph, outcome = existing.paragraph, "reused"
    else:
        model_name = getattr(profile.rag, "scope_summary_model", "") or _default_model()
        invoke = _text_invoke(model_name, profile)
        logger.info("writing corpus digest from %d section summar(ies)", len(usable))
        paragraph = build_corpus_digest(
            usable,
            invoke=invoke,
            persona=profile.identity.persona,
            languages=getattr(profile.rag, "scope_question_languages", ""),
        )
        if not paragraph:
            # Keeping the previous paragraph beats blanking it: one ingest out of date
            # still describes the corpus far better than the topic-list fallback.
            logger.error("corpus digest could not be written; keeping the previous one")
            paragraph, outcome = existing.paragraph, "failed"
        else:
            outcome = "written"

    floor, floor_key, questions = _derive_floor_for(profile, usable)
    save_digest(
        profile.name,
        DigestRecord(
            paragraph=paragraph,
            sections_sha256=fingerprint,
            section_count=len(usable),
            floor=floor,
            floor_sha256=floor_key,
            question_count=questions,
            model_used=getattr(profile.rag, "scope_summary_model", "") or _default_model(),
        ),
    )
    logger.info("corpus digest %s; floor %.4f over %d question(s)", outcome, floor, questions)
    return outcome


def _derive_floor_for(profile, records: List[SectionRecord]):
    """The scope floor for this catalogue, computed once here rather than at every boot."""
    import os

    from backend.rag.scope_index import derive_floor, floor_fingerprint

    questions: List[str] = []
    chunk_ids: List[str] = []
    vectors: List[Sequence[float]] = []
    for record in records:
        if len(record.question_vectors) != len(record.answers):
            # Positional pairing, exactly as build_index does it: a record whose vectors
            # do not line up with its questions contributes nothing rather than
            # contributing a misalignment.
            continue
        for question, vector in zip(record.answers, record.question_vectors):
            questions.append(question)
            chunk_ids.append(record.chunk_id)
            vectors.append(vector)

    if len(vectors) < 2:
        return 0.0, "", len(vectors)

    point = float(getattr(profile.rag, "scope_floor_percentile", 10.0))
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    floor = derive_floor(vectors, chunk_ids, point)
    return floor, floor_fingerprint(questions, chunk_ids, model_name, point), len(vectors)


def _text_invoke(model_name: str, profile):
    """A callable returning plain text for one prompt, at temperature 0.

    Separate from `_structured_invoke` because the digest wants prose, and asking for it
    inside a one-field schema buys nothing but a parse step that can fail.
    """
    import os

    from langchain.chat_models import init_chat_model

    from backend.assets.vision import call_with_rate_limit_retry

    model = init_chat_model(
        model=model_name,
        model_provider="openai",
        api_key=os.getenv("ARK_API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=0.0,
    )

    def invoke(prompt: str) -> str:
        response = call_with_rate_limit_retry(
            lambda: model.invoke([{"role": "user", "content": prompt}]),
            config=profile.assets,
            description="corpus digest",
        )
        return getattr(response, "content", "") or ""

    return invoke


def verify(records, expected: int) -> dict:
    """Is this catalogue actually usable, and if not, exactly which sections are not.

    Worth being explicit about because the failure is silent by nature: a section that
    did not summarise leaves no error behind once the run ends, and the gate simply has
    a hole in it — questions about that part of the corpus score low and get escalated
    forever, which looks like the model being cautious rather than like a missing row.
    """
    problems = []
    with_questions = [record for record in records if record.answers]
    without = [record.chunk_id for record in records if not record.answers]
    unvectored = [
        record.chunk_id for record in with_questions
        if len(record.question_vectors) != len(record.answers)
    ]

    if len(with_questions) < expected:
        missing = expected - len(with_questions)
        problems.append(f"{missing} section(s) have no catalogue entry")
    for chunk_id in without:
        problems.append(f"no questions: {chunk_id}")
    for chunk_id in unvectored:
        problems.append(f"no stored vectors: {chunk_id}")

    return {
        "complete": not problems,
        "sections": len(records),
        "with_questions": len(with_questions),
        "questions": sum(len(record.answers) for record in with_questions),
        "problems": problems,
    }


def _repair_missing_vectors(profile_name: str) -> int:
    """Embed and store vectors for any catalogued section that lacks them."""
    import os

    stored = load_records(profile_name)
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    stale = [
        record for record in stored
        if record.answers
        and (len(record.question_vectors) != len(record.answers)
             or record.embedding_model != model_name)
    ]
    if not stale:
        return 0
    logger.info("repairing vectors for %d section(s)", len(stale))
    _embed_questions(stale)
    save_records(profile_name, stale)
    return len(stale)


def _embed_questions(records: List[SectionRecord]) -> None:
    """Embed each record's questions here, at build time, not at boot.

    Measured before this existed: 22.4 seconds to embed 222 questions on CPU, paid by
    whichever user happened to arrive first after a deploy. The vectors are a pure
    function of the questions and the model, so computing them once and storing them
    turns that into a database read.

    One batched call rather than one per record — the embedder amortises a batch, and
    a per-record loop would spend most of its time in setup.
    """
    import os

    pending = [record for record in records if record.answers]
    if not pending:
        return

    from backend.indexing.embedding import embedding_service

    flat: List[str] = []
    for record in pending:
        flat.extend(record.answers)

    logger.info("embedding %d question(s) for %d section(s)", len(flat), len(pending))
    vectors = embedding_service.get_embeddings(flat)
    if len(vectors) != len(flat):
        # Leaving the vectors empty is safe: the index falls back to embedding at boot,
        # which is slow but correct. Storing a misaligned list would not be.
        logger.error("embedder returned %d vectors for %d questions; not storing any",
                     len(vectors), len(flat))
        return

    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    cursor = 0
    for record in pending:
        count = len(record.answers)
        record.question_vectors = [list(v) for v in vectors[cursor:cursor + count]]
        record.embedding_model = model_name
        cursor += count


def _default_model() -> str:
    import os

    return os.getenv("FAST_MODEL") or os.getenv("MODEL") or ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report what would change, call nothing")
    parser.add_argument("--force", action="store_true", help="re-summarise every section, ignoring the cache")
    parser.add_argument("--check", action="store_true",
                        help="report catalogue completeness and exit; calls nothing")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    if args.check:
        stored = load_records(get_profile().name)
        expected = len(load_sections(int(getattr(get_profile().rag, "scope_section_level", 1))))
        report = verify(stored, expected)
        print(f"sections={report['with_questions']}/{expected} questions={report['questions']}")
        for problem in report["problems"]:
            print(f"  INCOMPLETE: {problem}")
        print("catalogue complete" if report["complete"] else "catalogue INCOMPLETE")
        return 0 if report["complete"] else 2

    result = build(dry_run=args.dry_run, force=args.force)
    print(
        f"sections={result['sections']} summarised={result['summarised']} "
        f"reused={result['reused']} repaired={result.get('repaired', 0)} "
        f"removed={result['removed']} digest={result.get('digest', 'skipped')}"
    )
    if args.dry_run:
        return 0
    if result.get("complete"):
        print("catalogue complete")
        return 0
    # Non-zero so a CI step or a deploy script cannot pass over a partial catalogue.
    for problem in result.get("problems", []):
        print(f"  INCOMPLETE: {problem}")
    print("re-run to fill the gaps — summaries already built are reused")
    return 2


if __name__ == "__main__":
    sys.exit(main())
