"""Identity service.

The one place in the system that decides *who someone is*. Every other service —
records, the chat backend — verifies a signed token and reads the answer. None of
them resolve identity themselves, and none of them can mint a token.

Runs and deploys independently. Nothing here imports `backend`, `records`, or
`frontend`, and nothing imports this.
"""

__version__ = "0.1.0"
