"""The school information system — the service that *owns* academic structure.

The fourth process in this repo, on :8300, and the only one that writes school data.
`records/` reads a system of record and decides who may see it; `sis/` **is** a system
of record. That is why it is a separate service rather than tables added to `records/`:
a registrar uploading a roster and a parent asking about a grade have different uptime
requirements, different credentials and different blast radii, and putting the write
path inside the read facade means an import bug can take parent reads down.

Phase 1 was grades only; guardians followed, because a parent asking about her own child
is the point of the system and it cannot be answered without knowing who she is.
Attendance, report cards, timetables and fees remain out of scope by decision, not by
omission — see README.md.
"""

SERVICE_NAME = "sis"
__version__ = "0.1.0"
