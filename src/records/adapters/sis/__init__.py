"""The school's Student Information Service, over HTTP.

Three clients to one service — marks, guardian links, calendar — sharing one pooled
transport in `http.py` so they cannot disagree about the pool size, the timeout policy,
or the two rules that must never differ: no retries, and no redirects.
"""
from records.adapters.sis.calendar import SisSchoolCalendar
from records.adapters.sis.directory import SisGuardianDirectory
from records.adapters.sis.grades import SisAdapter

__all__ = ["SisAdapter", "SisGuardianDirectory", "SisSchoolCalendar"]
