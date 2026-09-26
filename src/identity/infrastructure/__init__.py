"""Everything that touches the outside world.

The database, the signing keys, the WhatsApp Cloud API, and the HTTP client that asks the
school's system of record who a phone number belongs to. Each module here implements a
port declared in `application/ports/`, and nothing in `domain/` or `application/` imports
any of it.
"""
