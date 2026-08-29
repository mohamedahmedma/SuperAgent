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

## Release and rollback

`deploy.yml` builds every service image, tags it with both the commit SHA and
`stable`, and pushes to GHCR. On the server it records the tag currently serving
traffic before rolling out, then pulls and starts the new tag.

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
