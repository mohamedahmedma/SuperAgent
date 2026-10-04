"""Application layer: the backend's use cases, and the ports they reach the world through.

  `ports/`     Protocols for what the use cases need - repositories, the unit of work, and
               the collaborators a service is handed.
  `services/`  The use cases themselves: plain classes, each built with its collaborators
               by `backend/composition.py`, one method per thing a route can ask for.

The same shape as `sis.application`, and deliberately not a command bus. A route calls a
service method; a test calls the same method. Nothing here imports FastAPI, Starlette,
SQLAlchemy or the graph framework - `.importlinter` holds that line.
"""
