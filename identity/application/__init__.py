"""The use cases, and the ports they need to carry them out.

This layer orchestrates: it reads and writes through `ports/`, applies the rules in
`domain/`, and returns the plain dataclasses in `dto.py`. It knows nothing about HTTP
status codes, SQL, or the environment.

The direction of every dependency is inward. `application/` imports `domain/`;
`infrastructure/` and `api/` import `application/`; nothing in `domain/` or `application/`
imports either of those.
"""
