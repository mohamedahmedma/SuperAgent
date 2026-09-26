"""In-memory stand-ins, and what an unconfigured deployment falls back to."""
from records.adapters.fake.calendar import FakeSchoolCalendar
from records.adapters.fake.directory import FakeGuardianDirectory
from records.adapters.fake.lms import FakeLms

__all__ = ["FakeGuardianDirectory", "FakeLms", "FakeSchoolCalendar"]
