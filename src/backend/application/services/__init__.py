"""The backend's use cases, as plain classes built with their collaborators.

Each service takes what it needs in its constructor, typed by the Protocols in
`application/ports/`, and `backend/composition.py` builds it with the real ones. A route calls
a method; a test calls the same method with fakes. The same shape as `sis.application.services`.

Services raise the errors in `backend/domain/errors.py` and never an HTTP status: which status
each error becomes is decided once, in `backend/api/errors.py`.
"""

from backend.application.services.sessions import SessionPage, SessionService

__all__ = ["SessionPage", "SessionService"]
