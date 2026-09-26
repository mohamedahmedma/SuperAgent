"""How a route receives a service.

One base dependency reads the container off the application, and one small provider per
service unwraps it. Routes then declare what they need in their signature —
`conversations: ConversationStorage = Depends(conversation_storage)` — instead of
importing a module-level instance, which is what makes a route testable with
`app.dependency_overrides` and what keeps `backend/composition.py` the only file that
knows how anything is built.

The providers are deliberately thin. Anything with logic in it belongs in a service.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Depends, Request

from backend.composition import Services, default_services

if TYPE_CHECKING:
    from backend.chat.storage import ConversationStorage
    from backend.indexing.pair_store import DocumentPairService
    from backend.jobs.upload_jobs import IngestJobTracker


def get_services(request: Request) -> Services:
    """The container `create_app()` built for this application.

    Falls back to the process default for an application assembled without one — a test
    that builds a bare `FastAPI()` around a router, most usually. A request served by
    `create_app()`'s application always gets that application's own container.
    """
    services = getattr(request.app.state, "services", None)
    return services if services is not None else default_services()


def conversation_storage(services: Services = Depends(get_services)) -> ConversationStorage:
    return services.conversations


def document_pair_service(services: Services = Depends(get_services)) -> DocumentPairService:
    return services.document_pairs


def upload_job_tracker(services: Services = Depends(get_services)) -> IngestJobTracker:
    return services.upload_jobs


def delete_job_tracker(services: Services = Depends(get_services)) -> IngestJobTracker:
    return services.delete_jobs


__all__ = [
    "conversation_storage",
    "delete_job_tracker",
    "document_pair_service",
    "get_services",
    "upload_job_tracker",
]
