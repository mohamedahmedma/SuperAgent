"""The rules, with nothing that does I/O.

Nothing in this package imports SQLAlchemy, FastAPI, `httpx`, or `identity.config`. That
is the whole constraint, and it is what makes the rules here testable by calling them:
"does eight bad passwords lock the account" is a function of a count and a threshold, and
"does a nonce survive a parent typing hello in front of it" is a function of a string.

The layer above — `identity/application/` — orchestrates these rules over ports it
declares. The layer below — `identity/infrastructure/` — implements those ports. Neither
direction of that dependency ever points inward at a database or a framework.
"""
