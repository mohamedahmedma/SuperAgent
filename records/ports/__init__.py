"""What this facade needs from the outside world, stated as Protocols.

The **driven** side of the hexagon. Every one of these is declared by the layer that uses
it and implemented under `adapters/`, so `application/` can be exercised with plain
classes and no network.

`Protocol` rather than an abstract base class, because structural typing means a fake is
a plain class with the right methods: it does not import this module, inherits from
nothing, and cannot be broken by a base gaining a method it does not use. The type checker
still catches an implementation that drifts.
"""
from records.ports.calendar import SchoolCalendar
from records.ports.directory import GuardianDirectory
from records.ports.lms import LmsAdapter

__all__ = ["GuardianDirectory", "LmsAdapter", "SchoolCalendar"]
