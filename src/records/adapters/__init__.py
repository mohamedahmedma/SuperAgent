"""The driven side of the hexagon: everything that talks to another system.

    sis/    the school's Student Information Service, over HTTP
    fake/   deterministic in-memory stand-ins, and the default when no SIS is configured

The fakes ship in the service rather than in the test suite on purpose. They are what an
unconfigured deployment falls back to — telling every parent they have no children on
file, rather than authorising them against nothing — and they are the reference for what
a correct adapter returns.
"""
