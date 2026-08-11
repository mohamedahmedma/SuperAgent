"""The identity application.

    uvicorn identity.app:app --port 8200

Owns authentication for the whole system and nothing else. It stores no grades, no
students, and no chat history — a service holding credentials should be the least
interesting target in the estate, not the most.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from identity.db import init_db
from identity.routes import admin_router, public_router, wellknown_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Force key load at startup so a misconfigured signing key fails the deploy,
    # rather than the first parent's login.
    from identity import keys

    keys.kid()
    yield


app = FastAPI(
    title="School Identity Service",
    version="0.1.0",
    description=(
        "Authentication for the school assistant. The only service that decides who "
        "someone is, and the only one that can mint a token.\n\n"
        "**Tokens are RS256.** Other services verify against `/.well-known/jwks.json` "
        "with a public key and hold nothing that could forge one. That is what keeps "
        "identity resolution out of the chat backend and the records facade.\n\n"
        "**The `guardian_id` claim is the whole integration.** It is set only by an "
        "administrator through `PUT /v1/admin/accounts/{username}/guardian-binding`, "
        "never at self-registration and never from a request body on a public route. "
        "A token without the claim can read no student records at all."
    ),
    lifespan=lifespan,
)

app.include_router(wellknown_router)
app.include_router(public_router)
app.include_router(admin_router)


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok", "service": "identity"}
