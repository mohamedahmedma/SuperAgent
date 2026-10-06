#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/load-deployment-profile.sh"
check() {
  local url="$1" expected="$2" code
  code="$(curl -sS -o /dev/null -w '%{http_code}' --retry 5 --retry-delay 5 \
    --retry-all-errors --max-time 20 "$url")" || return 1
  [ "$code" = "$expected" ] || { echo "$url returned $code, expected $expected" >&2; return 1; }
  printf 'OK %s -> %s\n' "$url" "$code"
}
for url in $PUBLIC_CHECK_URLS; do check "$url" 200; done
# Unauthenticated administration must be refused, not silently exposed.
for url in $PROTECTED_CHECK_URLS; do check "$url" 401; done
if [ "$DEPLOY_ENVIRONMENT" = production ]; then
  for host in "$SUPERAGENT_DOMAIN" "$SIS_DOMAIN"; do
    for path in docs redoc openapi.json; do check "https://$host/$path" 404; done
  done
fi
