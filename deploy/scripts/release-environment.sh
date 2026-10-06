#!/usr/bin/env bash
# Promote the frozen image manifest into exactly one isolated estate.
set -euo pipefail
deploy_path="${1:?deployment path required}"
image_tag="${2:?image tag required}"
image_prefix="${3:?registry prefix required}"
environment="${4:?environment required}"
case "$environment" in dev|test|production) ;; *) exit 2 ;; esac
[[ "$image_tag" =~ ^[0-9a-f]{40}$ ]] || { echo 'Expected a commit SHA'; exit 2; }
cd "$deploy_path"
source deploy/scripts/load-deployment-profile.sh
[ "$DEPLOY_ENVIRONMENT" = "$environment" ] || { echo 'Environment mismatch'; exit 2; }
if [ "$environment" != production ]; then
  [ "$deploy_path" != /opt/superagent ] || { echo 'Nonproduction cannot use the production directory'; exit 2; }
fi
[ -s .env ] || { echo 'Environment credentials absent'; exit 1; }
candidate_tags=".candidate-image-tags-$image_tag"
[ -s "$candidate_tags" ] || { echo 'Approved release manifest absent'; exit 1; }
declare -A candidate_values=()
while IFS='=' read -r key value; do
  case "$key" in
    REGISTRY_IMAGE_PREFIX) [ "$value" = "$image_prefix" ] || exit 2 ;;
    BACKEND_IMAGE_TAG|FRONTEND_IMAGE_TAG|IDENTITY_IMAGE_TAG|RECORDS_IMAGE_TAG|SIS_IMAGE_TAG)
      [[ "$value" =~ ^[0-9a-f]{40}$ ]] || exit 2 ;;
    *) echo 'Unknown manifest setting'; exit 2 ;;
  esac
  [ -z "${candidate_values[$key]+x}" ] || { echo 'Duplicate manifest setting'; exit 2; }
  candidate_values["$key"]="$value"
done < "$candidate_tags"
[ "${#candidate_values[@]}" -eq 6 ] || { echo 'Incomplete manifest'; exit 2; }
# Infrastructure, application data and keys belong to this stack's named volumes.
compose=(docker compose --env-file .env --env-file .deployment.env
  --env-file .release-image-tags -f docker-compose.yml -f docker-compose.prod.yml)
had_previous=false
if [ -f .release-image-tags ]; then
  cp .release-image-tags .release-image-tags.previous
  had_previous=true
fi
umask 077
{
  printf 'REGISTRY_IMAGE_PREFIX=%s\n' "$image_prefix"
  for service in BACKEND FRONTEND IDENTITY RECORDS SIS; do
    printf '%s_IMAGE_TAG=%s\n' "$service" "${candidate_values[${service}_IMAGE_TAG]}"
  done
} > .release-image-tags.next
mv .release-image-tags.next .release-image-tags
apps=(backend frontend identity records sis)
current_env_hash="$(cat .env .deployment.env | sha256sum | awk '{print $1}')"
previous_env_hash=""
if [ -f .current-env-sha256 ]; then previous_env_hash="$(cat .current-env-sha256)"; fi

healthy() {
  local service container state url
  for service in "${apps[@]}"; do
    container="$("${compose[@]}" ps -q "$service")"
    [ -n "$container" ] || return 1
    state="$(docker inspect -f '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}' "$container")"
    case "$state" in 'running healthy') ;; *) return 1 ;; esac
  done
  for url in $LOCAL_CHECK_URLS; do curl -fsS --max-time 5 "$url" >/dev/null || return 1; done
  curl -fsS --max-time 10 "http://127.0.0.1:$BACKEND_HOST_PORT/ready" >/dev/null || return 1
}
wait_healthy() {
  local i
  for i in $(seq 1 120); do
    if healthy; then return 0; fi
    sleep 5
  done
  return 1
}
rollback() {
  echo 'Release failed; collecting status and restoring the previous image manifest' >&2
  "${compose[@]}" ps || true
  # Application logs can contain personal information; do not copy them into public CI logs.
  if [ "$had_previous" = true ]; then
    cp .release-image-tags.previous .release-image-tags
    "${compose[@]}" up -d --no-deps --no-build --pull never "${apps[@]}" || true
    wait_healthy || true
  fi
}
attempt() {
  "${compose[@]}" pull "${apps[@]}" || return 1
  "${compose[@]}" up -d --no-build --pull missing --wait --wait-timeout 300 \
    postgres redis etcd minio standalone attu || return 1
  bash deploy/scripts/apply-env.sh --path "$deploy_path" </dev/null || return 1
  if [ "$current_env_hash" != "$previous_env_hash" ]; then
    "${compose[@]}" up -d --no-deps --no-build --pull never --force-recreate \
      backend identity records sis || return 1
  fi
  # Preserve incremental deployment: unchanged images remain running.
  "${compose[@]}" up -d --no-deps --no-build --pull never "${apps[@]}" || return 1
  wait_healthy || return 1
  local service actual key
  for service in "${apps[@]}"; do
    actual="$(docker inspect -f '{{.Config.Image}}' "$STACK_NAME-$service")" || return 1
    key="$(printf '%s' "$service" | tr '[:lower:]' '[:upper:]')_IMAGE_TAG"
    [ "$actual" = "$image_prefix/$service:${candidate_values[$key]}" ] || return 1
  done
}
if ! attempt; then rollback; exit 1; fi
# Nginx failures also fail the release, and restore the previous application images.
if ! bash deploy/scripts/configure-public-domains.sh "${5:?public IP required}" </dev/null; then
  rollback
  exit 1
fi
printf '%s\n' "$image_tag" > .current-image-tag
printf '%s\n' "$current_env_hash" > .current-env-sha256
"${compose[@]}" ps
