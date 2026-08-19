---
noteId: "3e5c416095c711f1a6d4fb6c6accc4db"
tags: []

---

# Identity Service

The one place in the system that decides **who someone is**. Every other service
verifies a signed token and reads the answer; none of them resolve identity, and none
of them can mint a token.

Runs on its own port, against its own database, with six dependencies. Nothing here
imports `backend/`, `records/`, or `frontend/`, and nothing imports this.

## Why it is separate

Before it existed, the guardian id reached the records facade as a path parameter that
the chat backend filled in. That made the chat process — the one running a language
model, parsing untrusted input, and calling out to third parties — the thing standing
between a parent and every family's records.

Now the chat backend relays a token it cannot forge and cannot alter. It has no idea
how identity is established and no ability to change the answer.

## RS256, not a shared secret

With HS256 every service that *verifies* a token also holds the key that *mints* one.
A leaked config file on the chat backend would forge parent identities.

With an asymmetric pair only this service signs. Everyone else fetches
`/.well-known/jwks.json`, holds a public key, and can do nothing with it but check a
signature. That is what makes "authentication is handled at the authentication layer"
a structural fact rather than a convention someone has to remember.

Verification elsewhere is **offline** — no service calls this one per request. Identity
being down does not take records down, and it is not in the latency path of a parent's
question.

## Running it

```bash
pip install -r identity/requirements.txt
IDENTITY_ADMIN_KEY=... uvicorn identity.app:app --port 8200
pytest identity/tests -q
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `IDENTITY_DATABASE_URL` | `sqlite:///./identity.db` | Point at Postgres for real data. |
| `IDENTITY_PRIVATE_KEY_PEM` | — | **Required in production.** Without it a dev key is generated and a warning logged. |
| `IDENTITY_ADMIN_KEY` | — | Guards account creation and guardian binding. |
| `IDENTITY_ISSUER` / `IDENTITY_AUDIENCE` | `school-identity` / `school-services` | Must match the verifier's settings. |
| `IDENTITY_ACCESS_TTL_MINUTES` | `30` | Bounds the revocation window. |
| `IDENTITY_MAX_FAILED_ATTEMPTS` | `8` | Then locked for `IDENTITY_LOCKOUT_MINUTES`. |

## The claim that matters

```json
{ "iss": "school-identity", "aud": "school-services",
  "sub": "0501234567", "role": "parent", "guardian_id": "G-1", "exp": 1234567890 }
```

`guardian_id` is the whole integration. It is set **only** by an administrator through
`PUT /v1/admin/accounts/{username}/guardian-binding` — never at self-registration,
never from a request body on a public route, never inferred. An account that could name
its own guardian id could read any family's records.

It is **omitted entirely** when there is no binding, rather than sent as null. A staff
token and an unbound parent arrive at the records facade as the same absence, and both
read nothing.

Creation and binding are two separate calls on purpose: a bulk parent import that runs
only the first produces accounts that can log in and read nothing, which is the safe
half-finished state.

## Endpoints

```
GET  /.well-known/jwks.json                              public key, for verifiers
POST /v1/auth/login                                      credentials -> tokens
POST /v1/auth/refresh                                    re-reads the binding
POST /v1/auth/logout                                     revokes a refresh token
GET  /v1/auth/me                                         decode your own token

POST   /v1/admin/accounts                                admin key
PUT    /v1/admin/accounts/{username}/guardian-binding    admin key
DELETE /v1/admin/accounts/{username}/guardian-binding    admin key; revokes sessions
```

## Decisions worth knowing

- **Refresh re-reads the binding** rather than copying it from the old token. A custody
  change takes effect within one access-token lifetime instead of persisting until the
  parent happens to log out.
- **Unbinding revokes refresh tokens.** The urgent custody path: the session dies.
- **Wrong password and unknown user are indistinguishable**, and the timing is
  equalised. Otherwise this endpoint confirms which parents are registered at the
  school.
- **Lockout is per account, not per IP.** The threat is credential stuffing against a
  known parent, and an attacker has more IPs than the school has parents. The cost is
  that a parent can be locked out deliberately — the better failure, since a locked-out
  parent phones the school and a breached one does not know to.
- **Access tokens cannot be revoked**, because verification is offline. Keeping them
  short is what bounds that window; refresh revocation is the real control.

## Not built yet

Phone OTP login, which is what parents will actually want; password reset; and per-IP
rate limiting in front of `/v1/auth/login`. The lockout policy limits damage per
account but does nothing about a broad sweep across many accounts.
