"""What the HTTP API accepts and answers, as Pydantic models - validated on the way in and out.

Where `sis`, `records` and `identity` keep theirs, `api/schemas/`. A request body declared as one
of these is validated by FastAPI before a route runs, and a `response_model` holds the answer to
its declared shape. That is the place to add a constraint when a field needs one.

The chat and session models are still in `backend/agent/schemas/chat.py`, because the turn
pipeline shares types with them - the rag trace and the HITL state travel inside a message.
"""
