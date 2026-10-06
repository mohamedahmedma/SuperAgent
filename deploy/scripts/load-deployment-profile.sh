#!/usr/bin/env bash
# Non-secret, repository-controlled topology. Never source the credentials in .env.
PROFILE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -f "$PROFILE_ROOT/.deployment.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROFILE_ROOT/.deployment.env"
  set +a
fi
DEPLOY_ENVIRONMENT="${DEPLOY_ENVIRONMENT:-production}"
case "$DEPLOY_ENVIRONMENT" in
  dev|test) expected_stack="superagent-$DEPLOY_ENVIRONMENT" ;;
  production) expected_stack=superagent ;;
  *) echo "Invalid deployment environment" >&2; return 1 ;;
esac
if [ "${STACK_NAME:-superagent}" != "$expected_stack" ]; then
  echo "Stack does not match deployment environment; refusing shared data" >&2
  return 1
fi
export STACK_NAME="$expected_stack"
export COMPOSE_PROJECT_NAME="$expected_stack"
# Direct manual calls default to the checked-in production topology.
if [ ! -f "$PROFILE_ROOT/.deployment.env" ]; then
  set -a
  source "$PROFILE_ROOT/deploy/environments/$DEPLOY_ENVIRONMENT.env"
  set +a
fi
read -r -a DOMAINS <<< "$PUBLIC_DOMAINS"
read -r -a RETIRED <<< "$RETIRED_DOMAINS"

is_managed_domain() {
  local candidate="$1" owned
  for owned in "${DOMAINS[@]}" "${RETIRED[@]}"; do
    [ "$candidate" = "$owned" ] && return 0
  done
  return 1
}
