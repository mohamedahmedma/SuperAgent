"""The use cases, one module per feature.

Sliced by what a person is trying to do — sign in with a password, sign in over WhatsApp,
administer an account — rather than by technical kind. That is deliberate: "what happens
when a parent's number is not registered" is answered by reading one file top to bottom,
and a change to the WhatsApp flow touches one module rather than a handler, a manager and
a helper in three different packages.

**Nothing here imports `identity.config`, FastAPI, SQLAlchemy or `httpx`.** Everything a
service needs arrives through its constructor as a port from `application/ports/`, which
is what makes each one testable with plain classes and no fixtures. `api/deps.py` is the
only place that knows an environment and a database exist.
"""
from identity.application.services.administration import AdministrationService
from identity.application.services.parent_sessions import ParentSessionService
from identity.application.services.sessions import SessionService
from identity.application.services.whatsapp_login import WhatsAppLoginService

__all__ = [
    "AdministrationService",
    "ParentSessionService",
    "SessionService",
    "WhatsAppLoginService",
]
