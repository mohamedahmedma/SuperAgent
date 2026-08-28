"""The HTTP surface: the driving side of the hexagon.

    records   the parent-facing reads, under /v1/guardians/{guardian_id}/...
    admin     three routes that answer 410 and name where they went
"""
from records.api.routers import admin, records

__all__ = ["admin", "records"]
