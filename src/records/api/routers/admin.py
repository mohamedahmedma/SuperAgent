"""Guardian administration — moved, not removed.

These three routes used to write this service's own guardian tables. Those tables are no
longer read: who a child's parents are is the registrar's fact and lives in SIS, and this
service asks rather than remembers.

They answer **410** instead of being deleted outright. A removed route is a 404, which a
caller reads as "wrong URL" and retries; a 410 says the thing itself is gone and names
where it went. What they must never do is what they would do if left alone — accept the
write, return 201, and change nothing, so that a registrar believes a parent has been
granted access to their child's records when nobody has.

They take no credential. They hold no data and reveal nothing a reader of this repo does
not already know, and a caller who cannot authenticate is exactly the one most in need of
being told the route moved.
"""
from fastapi import APIRouter, HTTPException, status

router = APIRouter(prefix="/v1/admin", tags=["admin"])

_MOVED_TO_SIS = {
    "code": "moved",
    "message": (
        "Guardians are managed in the school's SIS. Upload them there "
        "(POST /v1/imports/guardians/preview) and change records access with "
        "PATCH /v1/students/{student_number}/guardians/{phone}."
    ),
}


@router.post("/guardians", status_code=status.HTTP_410_GONE)
def create_guardian_moved() -> dict:
    """Gone. Guardians are created by the SIS spreadsheet import."""
    raise HTTPException(status_code=status.HTTP_410_GONE, detail=_MOVED_TO_SIS)


@router.post("/guardians/{guardian_id}/students", status_code=status.HTTP_410_GONE)
def link_student_moved(guardian_id: str) -> dict:
    """Gone. Links are created by the SIS import and amended on the SIS access route."""
    raise HTTPException(status_code=status.HTTP_410_GONE, detail=_MOVED_TO_SIS)


@router.delete(
    "/guardians/{guardian_id}/students/{student_id}", status_code=status.HTTP_410_GONE
)
def unlink_student_moved(guardian_id: str, student_id: str) -> dict:
    """Gone. Unlinking is a SIS operation."""
    raise HTTPException(status_code=status.HTTP_410_GONE, detail=_MOVED_TO_SIS)


__all__ = ["router"]
