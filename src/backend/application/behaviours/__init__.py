"""Behaviours: the cross-cutting concerns that wrap a handler instead of living inside it.

Two here, both of which need nothing but a clock and a logger. The ones that need
collaborators arrive with the slices that bring them: a transaction behaviour in step 6B,
once the unit of work is reachable from this layer, and turn admission in 6D.
"""

from backend.application.behaviours.logging_behaviour import Logging
from backend.application.behaviours.timing import Timing

__all__ = ["Logging", "Timing"]
