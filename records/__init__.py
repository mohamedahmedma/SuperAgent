"""Academic records facade.

A small, AI-free service that sits between the school assistant agent and whatever
system of record the school actually runs (Moodle today). It owns guardian
authorisation, terms, report card snapshots and the access audit; it owns no grades.

Runs and deploys independently of `backend/`. Nothing in this package imports from
there, and nothing there imports from here — the only coupling is the HTTP contract.
"""

__version__ = "0.1.0"
