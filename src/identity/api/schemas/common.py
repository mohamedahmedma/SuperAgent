"""Shapes shared by every router."""
from pydantic import BaseModel, Field


class ErrorOut(BaseModel):
    """The body behind every refusal this service makes.

    One envelope for all of them, whichever layer raised. A client parses one shape and
    branches on `code`, never on how the error happened to be produced — see
    `identity/api/errors.py`, which is the only place a domain error becomes a status.
    """

    code: str = Field(
        description="not_authorized | locked | not_found | conflict | not_configured"
    )
    message: str = ""


__all__ = ["ErrorOut"]
