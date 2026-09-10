#!/usr/bin/env bash
#
# Apply the server's .env to the running estate. This is the ONLY supported way to do it,
# by hand or from CI, and it is safe to re-run as often as you like.
#
# It exists because .env carries one secret that has two independent homes, and writing
# the file only changes one of them:
#
#   POSTGRES_PASSWORD is read by the postgres image ONLY while it initialises an empty
#   data directory. On an estate whose postgres_data volume already exists, editing
#   .env rotates what the backend SENDS and never what the role ACCEPTS. The backend is
#   the only service that speaks to postgres, so it is the only one that breaks, and it
#   breaks authenticating at boot while `pg_isready` - which does not authenticate -
#   still reports the container healthy. `depends_on: service_healthy` goes green, and
#   the failure surfaces as a hundred and fifty lines of SQLAlchemy traceback that read
#   like a broken image rather than a credential.
#
# So: reconcile the role with the file, PROVE the credential works over TCP the way the
# application will use it, and only then recreate anything. A wrong password fails here,
# with the previous containers still serving, instead of halfway through a restart.
#
#   ssh root@HOST
#   cd /opt/SuperAgent-main
#   bash deploy/scripts/apply-env.sh backend records identity sis frontend
#
# With no services named it reconciles and verifies only, changing nothing else - which
# is how .github/workflows/deploy.yml calls it, before running its own release.
#
# Options:
#   --pull never|always   registry policy for the recreate. Default `never`: compose's
#                         prod overlay sets `pull_policy: always`, so stale ghcr.io
#                         credentials on the server turn a recreate into "denied", then
#                         a failed BUILD, and nothing is recreated. `never` uses the
#                         image already on disk. CI, which has just pushed, passes
#                         `always`.
#   --path DIR            estate directory. Default: the repository this script is in.
#   --no-health           skip the /health gate, for a services list that excludes the
#                         backend.

set -uo pipefail

PULL_POLICY=never
DEPLOY_PATH=""
RUN_HEALTH=true
SERVICES=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --pull)       PULL_POLICY="${2:?--pull needs never or always}"; shift 2 ;;
    --path)       DEPLOY_PATH="${2:?--path needs a directory}"; shift 2 ;;
    --no-health)  RUN_HEALTH=false; shift ;;
    -h|--help)    sed -n '2,37p' "$0" | cut -c3-; exit 0 ;;
    -*)           printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
    *)            SERVICES+=("$1"); shift ;;
  esac
done

if [ "$PULL_POLICY" != never ] && [ "$PULL_POLICY" != always ]; then
  printf 'pull policy must be never or always, not %s\n' "$PULL_POLICY" >&2
  exit 2
fi

if [ -z "$DEPLOY_PATH" ]; then
  DEPLOY_PATH="$(cd "$(dirname "$0")/../.." && pwd)"
fi
cd "$DEPLOY_PATH" || { printf 'no such directory: %s\n' "$DEPLOY_PATH" >&2; exit 1; }

ok()    { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
warn()  { printf '  \033[33mWARN\033[0m  %s\n' "$*"; }
bad()   { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; }
head_() { printf '\n=== %s ===\n' "$*"; }
die()   { bad "$*"; exit 1; }

COMPOSE=(docker compose -f docker-compose.yml -f docker-compose.prod.yml)
# CI writes the per-service image tags it just published into this manifest and layers it
# over .env. It is absent on a manual run, where .env's own IMAGE_TAG stands.
if [ -f .release-image-tags ]; then
  COMPOSE=(docker compose --env-file .env --env-file .release-image-tags
           -f docker-compose.yml -f docker-compose.prod.yml)
fi

# --------------------------------------------------------------------------------------
head_ "1. .env is present and compose can interpolate it"

[ -f .env ] || die ".env is absent at $DEPLOY_PATH - nothing to apply"

perms="$(stat -c %a .env 2>/dev/null || echo unknown)"
if [ "$perms" = "600" ]; then
  ok ".env mode 600"
else
  warn ".env mode $perms - run: chmod 600 .env"
fi

# Every compose command interpolates EVERY service, so one missing required secret makes
# even `ps` and `logs` fail. Catching that here is the difference between a clear message
# and a deployment that cannot report its own state.
config_err="$(mktemp)"
if "${COMPOSE[@]}" config --quiet 2>"$config_err"; then
  ok "compose config resolves"
else
  bad "compose cannot interpolate .env:"
  sed 's/^/        /' "$config_err" >&2
  rm -f "$config_err"
  die "recover the missing secret from a running container first (see DEVOPS.md)"
fi
rm -f "$config_err"

# --------------------------------------------------------------------------------------
head_ "2. reconcile the postgres role with .env"

# Read straight from the file rather than from `compose config`, which would print every
# other secret too. PROD_ENV_FILE is commonly authored on Windows and compose strips the
# CR each value carries; reading the file does not, so without `tr` the role would be set
# to the password plus a carriage return - matching nothing the backend ever sends, and
# failing in exactly the way this step repairs.
pg_user="$(sed -n 's/^POSTGRES_USER=//p' .env | head -1 | tr -d '\r')"
pg_pass="$(sed -n 's/^POSTGRES_PASSWORD=//p' .env | head -1 | tr -d '\r')"
pg_db="$(sed -n 's/^POSTGRES_DB=//p' .env | head -1 | tr -d '\r')"
: "${pg_user:=postgres}"
: "${pg_db:=langchain_app}"

[ -n "$pg_pass" ] || die "POSTGRES_PASSWORD is empty in .env - compose declares it required"

"${COMPOSE[@]}" up -d postgres </dev/null >/dev/null 2>&1 || die "could not start postgres"

waited=0
until "${COMPOSE[@]}" exec -T postgres pg_isready -U "$pg_user" -q </dev/null; do
  waited=$((waited + 2))
  [ "$waited" -gt 120 ] && die "postgres did not accept connections in ${waited}s"
  sleep 2
done
ok "postgres accepting connections as '$pg_user'"

# ALTER USER CURRENT_USER, not a named role: this connects over the container's local
# socket, which pg_hba trusts, so it works even though the password is currently wrong -
# which is the whole point, since a rotated .env is exactly when it IS wrong.
#
# psql quotes :'pw' itself, so a password containing shell or SQL syntax is carried
# through as data rather than parsed. It only interpolates input it lexes, though - never
# -c, which goes to the server verbatim and fails on the colon - so the statement has to
# arrive on stdin. Output is discarded rather than logged.
if printf '%s\n' "ALTER USER CURRENT_USER WITH PASSWORD :'pw';" \
    | "${COMPOSE[@]}" exec -T postgres psql -v ON_ERROR_STOP=1 -U "$pg_user" \
        -d postgres -v pw="$pg_pass" >/dev/null 2>&1; then
  ok "role '$pg_user' password reconciled with .env"
else
  die "ALTER USER failed - postgres is up but would not accept the statement"
fi

# --------------------------------------------------------------------------------------
head_ "3. prove the credential works the way the application uses it"

# The reconcile above went over the trusted local socket. The backend connects over TCP
# and is challenged by pg_hba's scram-sha-256 line, so the only check worth anything is
# one that takes the same path. This is the assertion the 2026-09-10 incident was
# missing: pg_isready had already reported the container healthy while every
# authenticated connection to it was being refused.
if "${COMPOSE[@]}" exec -T -e PGPASSWORD="$pg_pass" postgres \
     psql -h 127.0.0.1 -U "$pg_user" -d "$pg_db" -c 'SELECT 1' </dev/null >/dev/null 2>&1; then
  ok "password authentication over TCP succeeds for '$pg_user' on '$pg_db'"
else
  die "the reconciled password is still rejected over TCP - check pg_hba and: docker logs superagent-postgres"
fi

# --------------------------------------------------------------------------------------
if [ "${#SERVICES[@]}" -eq 0 ]; then
  head_ "done (reconcile and verify only; no services named)"
  exit 0
fi

head_ "4. recreate ${SERVICES[*]}"

# --force-recreate is mandatory, not defensive. Compose hashes a service's own
# configuration to decide what to restart, and the CONTENTS of an env_file are not part
# of that hash - so an edited .env sits unread behind containers compose considers up to
# date, and `restart` does nothing at all here.
if [ "$PULL_POLICY" = always ]; then
  "${COMPOSE[@]}" pull "${SERVICES[@]}" </dev/null || die "pull failed"
fi
# Recreate with `--pull never` in both modes. The image is on disk by now either way:
# it was already there, or the explicit pull above just fetched it — and pulling first
# means a registry that fails cannot take a running container down with it. The flag has
# to be passed because the prod overlay sets `pull_policy: always`, which would otherwise
# reach for ghcr.io again at the moment the container is being replaced.
"${COMPOSE[@]}" up -d --no-deps --force-recreate --pull never "${SERVICES[@]}" </dev/null \
  || die "recreate failed"
ok "recreated against the current .env"

if [ "$RUN_HEALTH" = true ]; then
  head_ "5. health gate"
  if timeout 300 sh -c 'until wget -qO- http://127.0.0.1:8000/health >/dev/null 2>&1; do sleep 5; done'; then
    ok "backend /health answers"
  else
    bad "backend did not become healthy in 300s - its own diagnosis:"
    "${COMPOSE[@]}" logs --no-color --tail 40 backend 2>&1 | sed 's/^/        /'
    exit 1
  fi
fi

head_ "done"
"${COMPOSE[@]}" ps
