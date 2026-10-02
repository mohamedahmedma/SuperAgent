"""The agent: everything that decides what a turn does and composes its answer.

The planner and the turn pipeline (`chat`), the retrieval graph and its nodes (`rag`),
the tools the model may call (`tools`), the prompt templates (`prompts`), the profile
that configures all of it (`profiles`) and the types they exchange (`schemas`).

What stays outside is what serves and stores rather than decides: the HTTP surface
(`api`), the database and its repositories (`db`, `infra`, `application`), document
ingestion (`indexing`, `assets`) and background jobs (`jobs`).

## Why this sits under the backend rather than beside it

The agent is backend scope. `sis`, `records` and `identity` do not import it and have no
reason to: the backend reaches them over HTTP, as a client, and they do not know it
exists. A peer package at `src/agent/` would say the estate has a fifth service, and the
import graph would contradict that on the first read. Nesting it here says what is true —
this is the part of ONE service that decides, split from the part that serves.
"""
