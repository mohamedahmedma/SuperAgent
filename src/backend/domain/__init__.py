"""Pure rules. The school's own logic, and nothing that talks to anything.

Nothing here may import FastAPI, Starlette, SQLAlchemy, Alembic, LangChain, LangGraph,
Redis, pymilvus, httpx, requests, boto3 or openai. `.importlinter` enforces that rather
than trusting this docstring. Pydantic is allowed, because the sibling services' domains
already use it (`sis/domain/value_objects.py`, `sis/domain/grades.py`).

The test for whether a module belongs here: the school is the authority for it, and it can
be wrong while the process runs perfectly. A chunk-size limit is configuration. Which
guardian may read which child's marks is a rule.

Two contexts, and they are named after what the code already calls them:

  `agent`   one turn, from a parent's message to the answer served
  `corpus`  the documents, chunks, assets and the index built from them

`shared` is the kernel both may use. It is a leaf: it knows neither of them.

Empty in step 6A on purpose. The packages exist so contracts can be written against them
and so later slices have somewhere to arrive; the ~7,600 lines under agent/chat and
agent/rag that already import no framework move in step 6F, one aggregate at a time.
"""
