# SuperAgent operations

For the current three-environment pipeline, credentials and production-domain migration,
see [deploy/ENVIRONMENTS.md](deploy/ENVIRONMENTS.md). The complete access directory is
[deploy/ENVIRONMENT_LINKS.md](deploy/ENVIRONMENT_LINKS.md). Production troubleshooting
below applies to `/opt/superagent` only; dev and test have separate estates.

The stack has a FastAPI backend plus identity, records and SIS services, and a Vue
frontend served by Nginx. SIS serves its registrar console at `/ui`. Infrastructure is
PostgreSQL, Redis, etcd, MinIO, Milvus standalone, and Attu.

`SUPERAGENT.bat` is the Windows entry point. It creates the ignored `.runtime.env`
credentials, starts Docker Desktop, validates Compose, checks for host port
conflicts, waits for health checks, and never removes named volumes.

Day-to-day URLs and the full command reference live in `PORTS_AND_COMMANDS.txt`.

## Required deployment configuration

The existing production secrets remain in use. Dev and test need their own
GitHub Environments, bootstrap files and administration logins. The full table
and environment-variable overrides are documented in
[deploy/ENVIRONMENTS.md](deploy/ENVIRONMENTS.md).

The server's existing production `.env` remains persistent. No secret is committed,
and deployments do not delete volumes.

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

## When a release brings a NEW setting

The server's `.env` is persistent state — `deploy.yml` writes it only to bootstrap an
empty estate, and never again. That is deliberate (an operator-validated file must
survive a release), and it has a consequence worth stating: **a feature whose setting is
absent from the live `.env` ships disabled, and the release still reports success.**
Nothing in the pipeline compares `.env` against what the new code reads.

So when a release adds a setting, it has to be added to `/opt/superagent/.env` by hand:

```
ssh root@HOST
cd /opt/superagent
grep -E '^(LLM_PROVIDER|.*TRANSCRIPTION_MODEL)=' .env    # what is set now
# add the missing line, then apply it to the services that read it
bash deploy/scripts/apply-env.sh backend
```

`.env.example` is the list of what the code reads; a setting missing from `.env` takes
whatever default the code has, which for a feature-gating value means "off".

This bit the estate on 2026-09-15: voice notes shipped reading `TRANSCRIPTION_MODEL`
(or `<PROVIDER>_TRANSCRIPTION_MODEL` for the live `LLM_PROVIDER` block), production's
`.env` predated the feature and named none, so every recording was stored, transcribed
into nothing, and the parent was told to type their question instead. The backend now
says which at boot, and it is the first thing to check when a model-backed feature is
inert in production but works locally:

```
docker logs superagent-backend 2>&1 | grep 'LLM provider:'
# ... model=..., key from ..., transcription=<model|unset>
```

## Bringing a service onto the release it missed

`.release-image-tags` in `$DEPLOY_PATH` records, per service, the image tag the last
successful release put it on, and `apply-env.sh` recreates from that manifest. So if a
service is running an older image than `main` (check with
`docker inspect --format '{{.Config.Image}}' superagent-frontend`, and compare with the
`compose ps` at the end of the last "Stage 3" job), recreating it as-is only re-applies
the stale tag. Point the manifest at the release first, then recreate:

```
ssh root@HOST
cd /opt/superagent
tag=<full commit sha of the main merge>          # the tag CD published, e.g. 6b67dea191b6...
for s in FRONTEND IDENTITY; do sed -i "/^${s}_IMAGE_TAG=/d" .release-image-tags; echo "${s}_IMAGE_TAG=$tag" >> .release-image-tags; done
bash deploy/scripts/apply-env.sh --pull always frontend identity
```

Every service in a release used to be able to miss it: the pipeline handed the services
list to the server over ssh, which re-splits the command, and only the first service was
released while the run reported success (2026-09-15). The release now checks each named
service's running image against the tag and fails otherwise, so this section is for a
service that fell behind before that, or for a release rolled back by hand.

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

## Pipeline order

`deploy.yml` runs exactly three stages: **CI -> CD (publish the five images) -> Deploy**.
Push `develop` for dev, `test` for test, and `main` for production. A manual run selects
its environment. The production ref must be `main`, and the existing `Production`
GitHub Environment reviewers still approve the final stage.

CI is called directly at the pushed commit. Pull requests run CI without deploying;
branch pushes do not race an additional standalone CI run. Each environment gets its
own credentials, project name, containers, ports, networks and data volumes.

The default server remains `13.140.153.131`. Production retains `/opt/superagent`;
dev and test deploy to `/opt/superagent-dev` and `/opt/superagent-test`. A repository
administrator must provision the environments and independent nonproduction secrets
before their first deployment. See `deploy/ENVIRONMENTS.md` for the configuration.

## Knowledge-base upload size

An admin uploading a KB document crosses up to three proxies, and the SMALLEST ceiling
among them is the one that answers. All three are now in this repository.

| Hop | Where | Ceiling |
| --- | --- | --- |
| Host nginx, `superagent.aurexis.cc` -> `127.0.0.1:3000` | `deploy/nginx/production/superagent.aurexis.cc.conf` | `512m` |
| Frontend container nginx | `src/frontend/nginx.conf` | `512m` |
| Dev/test API and frontend hosts | `deploy/nginx/dev/` and `deploy/nginx/test/` | `512m` |

The deployed frontend uses its own origin for authentication and document uploads.
Dev/test also publish direct API hosts. Every upload route uses the same ceiling;
`tests/general/test_upload_size_limits.py` checks all three environments for missing
or inconsistent limits.

Nothing behind nginx imposes a limit: `save_upload_file` streams the body a megabyte at a
time, and Starlette's 1 MB `max_part_size` applies to form FIELDS, not to file parts.

### Why the UI vhost is repository-managed

It was hand-written, and it was the reason a 2 MB upload still answered 413 after the other
two were raised: it carried nginx's 1m default. The 413 page named `nginx/1.24.0 (Ubuntu)`
while the container runs `1.27.3-alpine`, and nginx relays an upstream's error body
unchanged — so the page identified its own author, and the request was never reaching the
container at all.

SIS now has its own production origin, `sis.aurexis.cc`. The SuperAgent host forwards
`/v1/` to Identity through the frontend proxy, while SIS owns its own `/v1/`, `/health`
and `/ui/` routes. Production hides `/docs`, `/redoc` and `/openapi.json`; dev/test
retain their independent API documentation. Environment verification checks each
selected host after deployment. Unrelated virtual hosts are left alone.

### Verifying

Against the RUNNING configuration, not the files — an edit in a vhost that is not enabled
changes nothing, and the symptom is identical to not having made it. The host's `nginx -T`
shows only the host's own two vhosts; the frontend container runs a separate nginx, so it
is asked separately:

```bash
sudo nginx -T | grep -c 'client_max_body_size 512m'   # 2 production + enabled dev/test hosts
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

The backend and sis each own their schema through Alembic (`src/backend/alembic.ini`,
`src/sis/alembic.ini`). Each container runs `alembic upgrade head` before its server starts,
the release runs it once more after the health gate, and each service refuses to start
on a database that is not at its own head revision. The backend records its revision in
`backend_alembic_version` rather than the default table, so both histories can live in
one Postgres database without reading each other's.

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
