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

POST /v1/auth/whatsapp/start                             begin a parent verification
GET  /v1/auth/whatsapp/webhook                           Meta's subscription handshake
POST /v1/auth/whatsapp/webhook                           inbound messages, signed
POST /v1/auth/whatsapp/status                            poll a verification
POST /v1/auth/whatsapp/verify                            code -> tokens

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

## Parent login by WhatsApp

Parents have no password and are never given one. The school already holds their phone
number, entered by a registrar from paperwork, and WhatsApp proves control of that number
at no cost.

```
browser  POST /v1/auth/whatsapp/start
         <- { poll_secret, link, message, business_number, expires_at }

parent   taps the link; WhatsApp opens with the message already typed; parent taps send
         (WhatsApp never sends it for them -- say so on the page)

Meta     POST /v1/auth/whatsapp/webhook   signed with X-Hub-Signature-256
         identity asks sis: POST /v1/guardians/resolve { phone }
         known   -> reply over WhatsApp with a six-digit code
         unknown -> reply "contact the school office", challenge rejected

browser  POST /v1/auth/whatsapp/status  { poll_secret }   -> pending | code_sent | ...
browser  POST /v1/auth/whatsapp/verify  { poll_secret, code }  -> the same TokenOut
                                                                  as a password login
```

### Two secrets, and why

`nonce` travels out in the link and back over WhatsApp. `poll_secret` never leaves the
browser. Holding one without the other is worth nothing:

- A nonce lifted from a screenshot and sent from the attacker's own phone delivers the code
  to **their** WhatsApp — but they have no poll secret, so they cannot finish. The parent
  whose nonce it was is merely blocked, not impersonated.
- A parent tricked into sending an attacker's nonce delivers the code to **the parent's**
  WhatsApp, which the attacker cannot read.

### Why it costs nothing

The parent messages first, which opens a 24-hour customer service window. Meta made
service conversations free on 1 November 2024, and under the per-message pricing that
started on 1 July 2025 "All non-template messages are free". We never send a template, so
we are never billed. That is a policy rather than a contract — `WhatsAppUnavailable` is a
handled outcome, and `identity/whatsapp.py` is the only file that would have to change.

### It never creates a guardian

An account never names its own guardian — the invariant `identity/models.py` has always
stated. This is a second authority for that column, not an exception to it: the binding
comes from the school's own records, keyed on a number WhatsApp proved, and a number `sis/`
does not hold is refused. The binding is re-asserted on every sign-in, so a registrar's
correction takes effect without an administrator touching this service.

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `IDENTITY_WHATSAPP_NUMBER` | — | The school's number, **E.164 with a leading `+`**, e.g. `+201288339613`. Refused at startup otherwise; see below. |
| `IDENTITY_WHATSAPP_PHONE_NUMBER_ID` | — | Meta's own id for that number. Not the number. |
| `IDENTITY_WHATSAPP_TOKEN` | — | A **System User** token. The dashboard's token expires in under 24 hours. |
| `IDENTITY_WHATSAPP_APP_SECRET` | — | Signs every inbound webhook. Unset means every message is rejected. |
| `IDENTITY_WHATSAPP_VERIFY_TOKEN` | — | Any string; must match what you type into the App Dashboard. |
| `IDENTITY_SIS_BASE_URL` | — | Where `sis/` lives, e.g. `http://localhost:8300`. Unset means every parent is refused. |
| `IDENTITY_SIS_API_KEY` | — | Sent as `X-API-Key`. `sis/` does not currently check it. |
| `IDENTITY_VERIFICATION_TTL_MINUTES` | `10` | How long a challenge lives. |

**The number must carry its `+`.** `01288339613` produces a link to `wa.me/01288339613`,
which is a different number that does not exist — the link opens, the chat is empty, no
message ever arrives, and nothing logs an error. `e164_or_raise` refuses it at startup so
that a silent estate-wide outage becomes a deploy that does not come up.

With no credentials the flow still runs end to end against a recording gateway and an empty
directory: developable with no Meta account, and an unconfigured production refuses every
parent rather than authenticating them against nothing.

### Going live

1. Meta business portfolio -> an app -> a WhatsApp Business Account.
2. Register the number. **It must not be active on WhatsApp Messenger or the WhatsApp
   Business app** — delete it there first, which destroys that number's message history and
   cannot be undone while it is on Cloud API. If staff currently chat to parents on it, use
   a different number.
3. Set a 6-digit two-step PIN during registration and keep it; re-registering needs it.
4. Create a **System User**, assign the app and the WABA, and generate a never-expiring
   token with `whatsapp_business_messaging`.
5. Point the webhook at `https://<host>/v1/auth/whatsapp/webhook` with your verify token,
   and **subscribe to the `messages` field** — nothing arrives otherwise. Public HTTPS on
   443 with a real certificate; `ngrok` for local work.
6. New portfolios start at a 250-recipient tier until business verification. Replies inside
   an open service window should not count against it — worth confirming before a rollout.

### Operational notes

- Meta retries an unacknowledged webhook for up to **seven days**, so duplicates are
  guaranteed. Deduplicated on the message id; without that, one parent tap sends several
  conflicting codes.
- The webhook answers 200 for anything it cannot use, and 403 only for a bad signature.
- WhatsApp throttles replies to one user to roughly one every six seconds.
- A parent sending from a number the school does not hold — dad's work phone — is refused
  by design. The fix is a registrar adding that number, not a looser rule here.

## Not built yet

Password reset, and per-IP rate limiting in front of `/v1/auth/login`. The lockout policy
limits damage per account but does nothing about a broad sweep across many accounts.
(Parent login by WhatsApp, formerly listed here, is above.)
