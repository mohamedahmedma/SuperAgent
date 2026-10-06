#!/usr/bin/env bash
# Release a complete immutable image set into exactly one isolated estate.
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
    printf '%s_IMAGE_TAG=%s\n' "$service" "$image_tag"
  done
} > .release-image-tags.next
mv .release-image-tags.next .release-image-tags
apps=(backend frontend identity records sis)

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
  "${compose[@]}" up -d --no-deps --no-build --pull never --force-recreate "${apps[@]}" || return 1
  wait_healthy || return 1
  local service actual
  for service in "${apps[@]}"; do
    actual="$(docker inspect -f '{{.Config.Image}}' "$STACK_NAME-$service")" || return 1
    [ "$actual" = "$image_prefix/$service:$image_tag" ] || return 1
  done
}
if ! attempt; then rollback; exit 1; fi
# Nginx failures also fail the release, and restore the previous application images.
if ! bash deploy/scripts/configure-public-domains.sh "${5:?public IP required}" </dev/null; then
  rollback
  exit 1
fi
printf '%s\n' "$image_tag" > .current-image-tag
"${compose[@]}" ps
