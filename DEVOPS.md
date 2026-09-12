# SuperAgent operations

The stack has a FastAPI backend plus identity, records and SIS services, and a Vue
frontend served by Nginx. SIS serves its registrar console at `/ui`. Infrastructure is
PostgreSQL, Redis, etcd, MinIO, Milvus standalone, and Attu.

`SUPERAGENT.bat` is the Windows entry point. It creates the ignored `.runtime.env`
credentials, starts Docker Desktop, validates Compose, checks for host port
conflicts, waits for health checks, and never removes named volumes.

Day-to-day URLs and the full command reference live in `PORTS_AND_COMMANDS.txt`.

## Required GitHub repository secrets

| Secret | Purpose |
| --- | --- |
| `DEPLOY_HOST` | Ubuntu Docker server hostname or IP. |
| `DEPLOY_PORT` | **Optional.** SSH port; defaults to `22` when unset. Set it only for a non-standard port. |
| `DEPLOY_USER` | Restricted deployment user. |
| `DEPLOY_PASSWORD` | SSH password, stored only as an Actions secret. |
| `DEPLOY_PATH` | Absolute deployment directory on the server. |
| `PROD_ENV_FILE` | The complete production environment file, as a multiline secret. |

Deployment is automatic on `main`. Missing deployment secrets fail the pipeline rather
than allowing an apparently successful run that never updated production.

`GITHUB_TOKEN` is provided automatically by Actions and publishes images to GHCR; it
does not need to be created. Production credentials stay in GitHub Secrets and in the
server-side `.env` that `PROD_ENV_FILE` writes. No secret is ever committed, and
deployment never deletes production volumes.

## Changing `.env` on a live server

**Never `scp` a `.env` and then run `docker compose up` yourself.** Use the script, which
is the only supported way to apply the file and is safe to re-run:

```
scp .env root@HOST:$DEPLOY_PATH/.env       # from the laptop, NOT from inside ssh
ssh root@HOST
cd $DEPLOY_PATH && chmod 600 .env
bash deploy/scripts/apply-env.sh backend records identity sis frontend
```

`$DEPLOY_PATH` is `/opt/superagent`, the directory the pipeline refreshes on every
release. Run nothing from any other checkout on the server. The compose project name is
pinned to `superagent`, so compose run from a stale copy still recreates the live
containers — with that copy's own `.env` and whatever `:stable` image is on disk, and
without the `.release-image-tags` manifest that says which image each service should be
running. That is how production went down on 2026-09-11.

The pipeline calls the same script (`deploy.yml`, before its release), so the manual and
automated routes cannot drift. It validates the file, reconciles Postgres, proves the
credential authenticates, and only then recreates anything — a wrong value fails while
the previous containers are still serving.

Two things about this estate make the manual route dangerous without it:

**`POSTGRES_PASSWORD` has two independent homes, and writing `.env` changes one.** The
postgres image reads it only while initialising an *empty* data directory. Once
`postgres_data` exists, the role keeps the password it was created with, so editing
`.env` rotates what the backend **sends** and never what the role **accepts**. `pg_isready`
does not authenticate, so the container stays `(healthy)` and `depends_on:
service_healthy` goes green while every connection is refused. The backend is the only
service that speaks to Postgres, so it is the only one that breaks — which reads like a
broken image rather than a credential. `apply-env.sh` reconciles the role with the file
(`ALTER USER CURRENT_USER`, over the container's trusted local socket, so it works even
while the password is wrong) and then proves it over TCP, the way the application
connects. Rotating the password is therefore just: edit `.env`, run the script.

**An edited `.env` reaches nothing on its own.** Compose hashes a service's own
configuration to decide what to recreate, and the *contents* of an `env_file` are not part
of that hash — so a new file sits unread behind containers Compose considers up to date.
`restart` does nothing. `--force-recreate` is mandatory, and the script always passes it.

If Compose reports `required variable ... is missing a value`, it stops interpolating
*everything* — `ps` and `logs` included. The running containers still hold what they were
created with, so recover from them rather than guessing:

```
docker inspect superagent-postgres --format '{{range .Config.Env}}{{println .}}{{end}}' | grep ^POSTGRES_
docker inspect superagent-minio    --format '{{range .Config.Env}}{{println .}}{{end}}' | grep ^MINIO_ROOT_
docker inspect superagent-records  --format '{{range .Config.Env}}{{println .}}{{end}}' | grep ^RECORDS_API_KEY=
```

`RECORDS_API_KEY` is the value of `LOCAL_SERVICE_KEY`; Compose hands that one secret to
five services under different names.

`docker compose config` renders **every** secret in plaintext. Use `config --quiet`, which
prints nothing and still fails on an unresolvable variable.

## What the backend says at boot

One line each for the things a deployment can silently believe wrongly. After a release:

```
docker logs superagent-backend 2>&1 | grep -E "LLM provider:|LangSmith:|Database:|Vision"
```

| Line | Answers |
| --- | --- |
| `LLM provider: …` | which provider and model every call in the request path goes to |
| `LangSmith: …` | whether runs are traced, to which project and endpoint, under which key |
| `Database: …` | which database and user — never the password |
| `Vision: …` | whether figure extraction has usable credentials |

Tracing is worth the line because nothing in the codebase turns it on; the SDK reads the
environment. A switch that is off, a project nobody is watching and a key the endpoint
rejects are otherwise indistinguishable — the runs just never arrive. The key is shown as
`sha256 | cut -c1-8`, the same shorthand the `.env` comparison above uses, so the line can
be matched against the file a deployment believes it is using without either being read
aloud.

Comparing that fingerprint against `.env` takes the newline off first, or a key that is
correct fingerprints two different ways and reads as a mismatch that is not there:

```
cd /opt/superagent
printf %s "$(grep '^LANGSMITH_API_KEY=' .env | cut -d= -f2- | tr -d '\r"')" | sha256sum | cut -c1-8
docker exec superagent-backend sh -c 'printf %s "$LANGSMITH_API_KEY" | sha256sum | cut -c1-8'
```

`sha256sum` reading a pipeline hashes the trailing newline `grep` emits; `$( )` strips it,
and `printf %s` adds none. Both sides then agree, and the boot line's fingerprint — taken
from the value in memory — agrees with them.

## Pipeline order

`deploy.yml` runs `ci.yml` as its first job and everything else depends on it, so a push
to `main` goes **CI → CD (publish images) → deploy** in one run. Nothing is pushed to GHCR and nothing
reaches the server until the full suite — backend tests, both frontends, Compose
validation and the image builds — has passed.

CI is *called* rather than triggered on its completion. Under a `workflow_run` trigger
`github.sha` is the default branch head rather than the commit that was pushed, so every
image would be tagged and deployed for the wrong commit.

`main` is therefore not in `ci.yml`'s own push triggers: it would race a second, redundant
CI run against the deployment it is meant to gate. Pull requests to `main`, and pushes to
`develop`, still run CI on their own.

## Current automatic production target

Every push to `main` runs the ordered `CI -> CD -> deploy` pipeline. The production
defaults are host `13.140.153.131`, user `root`, port `22`, and path
`/opt/superagent`; matching repository secrets can override them. `DEPLOY_PASSWORD`
and `PROD_ENV_FILE` are mandatory GitHub Actions secrets. A missing value fails the run
instead of silently skipping deployment.

The `deploy` job targets the GitHub Environment named `production`. Configure that
environment with both maintainers as required reviewers; GitHub then pauses after CD and
either reviewer can select **Approve and deploy**. Reviewers are GitHub users or teams,
not arbitrary email addresses, and each account receives the request according to its
GitHub notification settings.

## Knowledge-base upload size

An admin uploading a KB document crosses up to three proxies, and the SMALLEST ceiling
among them is the one that answers. All three are now in this repository.

| Hop | Where | Ceiling |
| --- | --- | --- |
| Host nginx, `superagent.aurexis.cc` -> `127.0.0.1:3000` | `deploy/nginx/superagent.aurexis.cc.conf` | `512m` |
| Frontend container nginx | `frontend/nginx.conf` | `512m` |
| Host nginx, `api.aurexis.cc` -> `127.0.0.1:8000` | `deploy/nginx/api.aurexis.cc.conf` | `512m` |

Which hops apply depends on how the image was built. `frontend/Dockerfile` accepts only
`VITE_IDENTITY_BASE_URL`, so `VITE_API_BASE_URL` is empty in the bundle and `utils/api.ts`
posts to the page's own origin — across the first two rows. Set that build arg and uploads
go to `api.aurexis.cc` instead. All three are kept at the same number so the route cannot
change the outcome; `tests/general/test_upload_size_limits.py` fails if they drift or if
any goes missing.

Nothing behind nginx imposes a limit: `save_upload_file` streams the body a megabyte at a
time, and Starlette's 1 MB `max_part_size` applies to form FIELDS, not to file parts.

### Why the UI vhost is repository-managed

It was hand-written, and it was the reason a 2 MB upload still answered 413 after the other
two were raised: it carried nginx's 1m default. The 413 page named `nginx/1.24.0 (Ubuntu)`
while the container runs `1.27.3-alpine`, and nginx relays an upstream's error body
unchanged — so the page identified its own author, and the request was never reaching the
container at all.

**That vhost also fronts SIS**, on `127.0.0.1:8300`: `/v1/`, `/health`, `/docs`,
`/openapi.json` and `/ui/` are all the registrar console, and only `/` is the chat UI.
`configure-public-domains.sh` DELETES the vhost it takes over, so those locations are
reproduced verbatim in `deploy/nginx/superagent.aurexis.cc.conf` and must stay there. A
rewrite that dropped `/ui/` or `/health` would pass `nginx -t` and take the console down
silently. `verify_public` checks both halves after every deployment for that reason.

The takeover also means the deploy pipeline now owns SIS's public routing on this host, and
that the TLS block no longer includes certbot's `options-ssl-nginx.conf` — it states
`ssl_protocols` directly, as the other two vhosts do, so the vhost cannot become unloadable
on a host whose certbot has no nginx plugin. The certificate itself is untouched: the
script only issues one when none exists.

### Verifying

Against the RUNNING configuration, not the files — an edit in a vhost that is not enabled
changes nothing, and the symptom is identical to not having made it. The host's `nginx -T`
shows only the host's own two vhosts; the frontend container runs a separate nginx, so it
is asked separately:

```bash
sudo nginx -T | grep -c 'client_max_body_size 512m'   # expect 2
docker exec superagent-frontend nginx -T | grep -c 'client_max_body_size 512m'   # expect 1
```

Then prove the path end to end, because `nginx -T` still does not show which hop a real
request meets. No token is needed: the body clears every proxy before auth rejects it, so
the status alone distinguishes a size limit from an auth failure.

```bash
head -c 2000000 /dev/urandom > /tmp/probe.bin
curl -s -o /dev/null -w '%{http_code}\n' -X POST \
  -F 'file=@/tmp/probe.bin;filename=probe.pdf' \
  https://superagent.aurexis.cc/documents/upload/async
```

`401` is the pass: the 2 MB body crossed both proxies and the backend refused it for auth.
`413` means a hop is still at its default — read the error page's footer to see which nginx
wrote it.


## Release and rollback

`deploy.yml` builds every service image, tags it with both the commit SHA and
`stable`, and pushes to GHCR. On the server it records the tag currently serving
traffic before rolling out, then pulls and starts the new tag.

Every release then brings **all** application services onto the server's `.env`, not
only the ones it rebuilt. When `.env` has changed since the last successful release,
every service that reads it is recreated; whether or not it has, any service that is not
running is started. The health gate runs after that, so a release counts as healthy
only with the `.env` actually applied — a bad value fails the release instead of
surfacing after a green run. The file that is applied is `/opt/superagent/.env` on the
server. The pipeline never overwrites it from the `PROD_ENV_FILE` secret, which is used
only to create the file when it is missing.

A release counts as healthy only when the backend `/health` endpoint and the frontend
both answer within 300 seconds — the timeout covers Milvus's slow cold start. If that
check fails, the workflow automatically redeploys the previously recorded tag and
fails the run, so a bad release does not stay live.

**Schema migrations are forward-only.** A rollback restores images, never the
database. Keep each migration backwards-compatible with the release before it, or an
image rollback will meet a schema it cannot read.

## Kubernetes evaluation — not adopted, and why

Kubernetes was evaluated and deliberately **not** adopted. Docker Compose is the
correct tool for this system today. Three properties of the current architecture
decide it:

1. **The deployment target is a single Ubuntu host.** `deploy.yml` deploys over SSH to
   one server. Kubernetes would add a control plane, a CNI, storage classes, and
   ingress machinery to run one node's worth of containers — cost with no benefit.

2. **`identity` and `sis` are backed by SQLite** (`sqlite:////app/data/*.db` on a
   mounted volume). SQLite is single-writer. Horizontal pod autoscaling — the main
   reason to reach for Kubernetes — is not merely useless here, it is *unsafe*:
   pointing two replicas at one shared `ReadWriteMany` volume risks database
   corruption. Kubernetes cannot be adopted meaningfully until these two services move
   to PostgreSQL.

3. **Milvus runs standalone.** The supported Kubernetes path is the Milvus Operator or
   the official Helm chart with a clustered topology. Hand-written StatefulSets for
   standalone Milvus, etcd, and MinIO would reimplement that operator, worse.

Writing manifests now would produce configuration nobody runs, that drifts from the
Compose files that are actually deployed, and that invites an unsafe `replicas: 2` on
a SQLite-backed service.

### Revisit Kubernetes when any of these becomes true

- More than one node is needed for HA or genuinely zero-downtime rolling updates.
- Backend traffic requires horizontal scaling — **after** `identity` and `sis` are
  migrated off SQLite onto PostgreSQL.
- Several strongly isolated environments are needed beyond the current single
  production target.
- A platform team already operates a cluster this can live in.

The migration path, when that day comes: move `identity` and `sis` to PostgreSQL
first, adopt the Milvus Operator for the vector tier rather than porting the Compose
service, and only then translate the five stateless application services into
Deployments. Reach for Helm or Kustomize at that point, not before.
