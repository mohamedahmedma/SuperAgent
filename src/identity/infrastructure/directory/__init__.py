"""The school's system of record, as this service reaches it.

`sis.py` is the real one, over HTTP. `fake.py` is a dict, and is what an unconfigured
deployment falls back to — see its docstring for why that is the safe direction.
"""
from identity.infrastructure.directory.fake import FakeGuardianDirectory
from identity.infrastructure.directory.sis import SisGuardianDirectory

__all__ = ["FakeGuardianDirectory", "SisGuardianDirectory"]
