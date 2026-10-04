"""Adapters. Everything that talks to something that is not this process.

Indexed by TECHNOLOGY, not by context, which is what all three sibling services do
(`identity/infrastructure/{crypto,db,directory,whatsapp}`,
`sis/infrastructure/{db,parsers,repositories,estate}`). One adapter proves why: the
embedder in `indexing/embedding.py` is used by the turn side and the ingest side alike, so
it cannot live inside either context's directory.

Implements the Protocols in `application/ports/`. Nothing here is imported by `domain` or
`application` - only `bootstrap` knows these modules exist.

Empty in step 6A. Step 6E moves today's `infra/`, `indexing/`, `assets/` and the loose
llm_*.py and provider_*.py modules in.
"""
