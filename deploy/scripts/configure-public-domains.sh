#!/usr/bin/env bash
#
# Publish the identity and backend services on their public domains.
#
# Runs on the production VPS, from the deployment directory, after the compose stack is
# up. It is deliberately re-runnable: every step checks the state it wants before acting,
# so a second run installs nothing, requests no certificate, and reloads nothing.
#
# The two services listen on loopback only. Host nginx is the sole public surface, and
# this script owns exactly two virtual hosts — auth.aurexis.cc and api.aurexis.cc. Any
# other vhost on the box, the SIS and Super Agent hosts included, is left alone.
#
# Order matters, and the reason is that nginx refuses to load a server block naming a
# certificate file that does not exist. So a domain without a certificate is first
# published over HTTP alone, which is enough for the ACME challenge; the TLS vhost is
# installed only once the certificate is on disk.

set -euo pipefail

WEBROOT=/var/www/certbot
BACKUP_ROOT=/var/backups/superagent-nginx
RENEWAL_HOOK=/etc/letsencrypt/renewal-hooks/deploy/00-reload-nginx.sh
# Treat a certificate with less than this left as "renewing soon"; certbot's timer owns
# the renewal, this script only declines to reissue.
RENEW_WINDOW_DAYS=30

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGINX_SRC="$(cd "$SCRIPT_DIR/../nginx" && pwd)"
DEPLOY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# The pipeline connects as root; keep working if someone runs this by hand as a non-root
# user with sudo rights.
if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi

DOMAINS=(auth.aurexis.cc api.aurexis.cc)

LAYOUT=""
BACKUP_DIR=""

log()  { printf '[domains] %s\n' "$*"; }
fail() { printf '[domains] ERROR: %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------------------
# Prerequisites
# --------------------------------------------------------------------------------------

require_tools() {
  command -v nginx >/dev/null 2>&1 || fail "nginx is not installed on this host"
  command -v curl  >/dev/null 2>&1 || fail "curl is not installed on this host"
  if ! command -v certbot >/dev/null 2>&1; then
    log "certbot missing; installing"
    $SUDO apt-get update -qq
    $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq certbot
  fi
  command -v certbot >/dev/null 2>&1 || fail "certbot could not be installed"
  # dig lets the DNS check query a public resolver directly. Without it the check falls
  # back to getent, which answers from this host's own resolver and would happily confirm
  # a record that the rest of the internet cannot see.
  if ! command -v dig >/dev/null 2>&1; then
    log "dig missing; installing dnsutils"
    $SUDO apt-get update -qq
    $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq dnsutils \
      || log "WARN could not install dnsutils; falling back to the local resolver"
  fi
  $SUDO install -d -m 755 "$WEBROOT"
  $SUDO install -d -m 700 "$BACKUP_ROOT"
}

# Ubuntu keeps sites-available/sites-enabled; some images only have conf.d. Support both
# rather than assume, and never invent a layout the host does not already use.
detect_layout() {
  if [ -d /etc/nginx/sites-available ] && [ -d /etc/nginx/sites-enabled ]; then
    LAYOUT=sites
  elif [ -d /etc/nginx/conf.d ]; then
    LAYOUT=confd
  else
    fail "neither /etc/nginx/sites-available nor /etc/nginx/conf.d exists"
  fi
  log "nginx layout: $LAYOUT"
}

conf_path() {
  if [ "$LAYOUT" = sites ]; then
    echo "/etc/nginx/sites-available/$1.conf"
  else
    echo "/etc/nginx/conf.d/$1.conf"
  fi
}

enabled_path() {
  if [ "$LAYOUT" = sites ]; then
    echo "/etc/nginx/sites-enabled/$1.conf"
  else
    echo "/etc/nginx/conf.d/$1.conf"
  fi
}

# --------------------------------------------------------------------------------------
# Local service checks — before touching nginx, prove there is something to publish
# --------------------------------------------------------------------------------------

http_status() {
  # curl already prints 000 to stdout when it never got a response, so a `|| echo 000`
  # here would append a second one and report the status as 000000.
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$1" 2>/dev/null)"
  printf '%s' "${code:-000}"
}

wait_for_local() {
  local url="$1" want="$2" tries="$3" i=1 code
  while :; do
    code="$(http_status "$url")"
    if [ "$code" = "$want" ]; then
      log "OK  $url -> $code"
      return 0
    fi
    if [ "$i" -ge "$tries" ]; then
      break
    fi
    i=$((i + 1))
    sleep 5
  done
  log "FAILED $url -> $code (wanted $want)"
  return 1
}

check_local_services() {
  log "checking services on loopback"
  wait_for_local "http://127.0.0.1:8200/docs" 200 24 \
    || fail "identity is not serving /docs on 127.0.0.1:8200"
  wait_for_local "http://127.0.0.1:8000/health" 200 24 \
    || fail "backend is not serving /health on 127.0.0.1:8000"
  # /ready reports 503 until the embedding model is built, which on a cold hf_cache is a
  # download rather than a fault. Wait it out, then print the body, which names the
  # dependency that stayed down.
  if ! wait_for_local "http://127.0.0.1:8000/ready" 200 60; then
    log "readiness body:"
    curl -s --max-time 10 "http://127.0.0.1:8000/ready" || true
    echo
    fail "backend did not become ready on 127.0.0.1:8000/ready"
  fi
}

# --------------------------------------------------------------------------------------
# DNS — verify only. Namecheap is authoritative and this script never edits it.
# --------------------------------------------------------------------------------------

resolve_a() {
  local domain="$1"
  if command -v dig >/dev/null 2>&1; then
    dig +short @1.1.1.1 "$domain" A 2>/dev/null | grep -E '^[0-9]+\.' || true
  else
    getent ahostsv4 "$domain" 2>/dev/null | awk '{print $1}' | sort -u || true
  fi
}

check_dns() {
  local expected="$1" domain got
  log "verifying DNS against a public resolver (expecting $expected)"
  for domain in "${DOMAINS[@]}"; do
    got="$(resolve_a "$domain")"
    if [ -z "$got" ]; then
      fail "$domain does not resolve to any A record"
    fi
    if ! printf '%s\n' "$got" | grep -qx "$expected"; then
      fail "$domain resolves to [$(printf '%s' "$got" | tr '\n' ' ')], not $expected"
    fi
    log "OK  $domain -> $expected"
  done
}

# --------------------------------------------------------------------------------------
# Certificates
# --------------------------------------------------------------------------------------

cert_live_dir() { echo "/etc/letsencrypt/live/$1"; }

cert_exists_at_all() { [ -s "$(cert_live_dir "$1")/fullchain.pem" ]; }

# Present, readable, and not inside the renewal window.
cert_is_usable() {
  local domain="$1" dir
  dir="$(cert_live_dir "$domain")"
  [ -s "$dir/fullchain.pem" ] || return 1
  [ -s "$dir/privkey.pem" ] || return 1
  openssl x509 -in "$dir/fullchain.pem" -noout \
    -checkend $((RENEW_WINDOW_DAYS * 86400)) >/dev/null 2>&1
}

issue_cert() {
  local domain="$1"
  if [ -z "${LETSENCRYPT_EMAIL:-}" ]; then
    fail "$domain has no certificate and LETSENCRYPT_EMAIL is unset; add the secret before deploying"
  fi
  log "requesting a certificate for $domain (webroot)"
  # --keep-until-expiring makes a re-run a no-op instead of a duplicate order, and there
  # is deliberately no --force-renewal anywhere in this script.
  $SUDO certbot certonly \
    --webroot -w "$WEBROOT" \
    -d "$domain" \
    --email "$LETSENCRYPT_EMAIL" \
    --agree-tos --no-eff-email \
    --non-interactive --keep-until-expiring \
    || fail "certbot could not issue a certificate for $domain"
  cert_exists_at_all "$domain" \
    || fail "certbot reported success but no certificate exists for $domain"
}

ensure_renewal() {
  # certbot's own timer does the renewing; all this adds is the reload that makes a
  # renewed certificate take effect without waiting for the next deployment.
  $SUDO install -d -m 755 /etc/letsencrypt/renewal-hooks/deploy
  printf '#!/bin/sh\n# Installed by SuperAgent deploy. Reload nginx after a renewal.\nnginx -t && nginx -s reload\n' \
    | $SUDO tee "$RENEWAL_HOOK" >/dev/null
  $SUDO chmod 0755 "$RENEWAL_HOOK"
  if command -v systemctl >/dev/null 2>&1; then
    if systemctl list-unit-files 2>/dev/null | grep -q '^certbot\.timer'; then
      $SUDO systemctl enable --now certbot.timer >/dev/null 2>&1 \
        || log "WARN could not enable certbot.timer"
      log "certbot.timer enabled"
    else
      log "WARN no certbot.timer on this host; renewal must be scheduled another way"
    fi
  fi
}

# --------------------------------------------------------------------------------------
# nginx configuration
# --------------------------------------------------------------------------------------

# A host with IPv6 switched off cannot bind [::], and nginx would then fail to reload for
# every site on the box, not just ours. Drop those directives rather than take the estate
# down over an address family this deployment does not require.
ipv6_available() { [ -s /proc/net/if_inet6 ]; }

render_conf() {
  local src="$1" dest="$2"
  if ipv6_available; then
    cat "$src" > "$dest"
  else
    grep -v 'listen \[::\]' "$src" > "$dest"
  fi
}

write_bootstrap_conf() {
  local domain="$1" dest="$2"
  {
    echo "# Temporary HTTP-only vhost for $domain, written by configure-public-domains.sh."
    echo "# It exists only so the ACME challenge can be answered before a certificate is"
    echo "# on disk; the TLS vhost from deploy/nginx/ replaces it in the same run."
    echo "server {"
    echo "    listen 80;"
    if ipv6_available; then
      echo "    listen [::]:80;"
    fi
    echo "    server_name $domain;"
    echo "    location ^~ /.well-known/acme-challenge/ {"
    echo "        root $WEBROOT;"
    echo "        default_type \"text/plain\";"
    echo "    }"
    echo "    location / { return 503; }"
    echo "}"
  } > "$dest"
}

# Refuse to fight another vhost for the same name. A file that serves only our domains is
# ours to replace and is backed up first; one that also serves something else belongs to
# somebody else and stops the deployment instead of being edited.
disable_conflicts() {
  local domain="$1" ours_conf ours_enabled f names
  ours_conf="$(conf_path "$domain")"
  ours_enabled="$(enabled_path "$domain")"
  for f in /etc/nginx/sites-enabled/* /etc/nginx/conf.d/*.conf; do
    [ -e "$f" ] || continue
    [ "$f" = "$ours_conf" ] && continue
    [ "$f" = "$ours_enabled" ] && continue
    # Compare parsed names rather than pattern-match the line: a regex built from a
    # domain treats its dots as wildcards, so api.aurexis.cc would also claim a file
    # serving apiXaurexisYcc, and a leading-dot allowance would claim any subdomain.
    names="$(grep -hoE '^[[:space:]]*server_name[^;]*' "$f" 2>/dev/null \
      | sed 's/^[[:space:]]*server_name[[:space:]]*//' \
      | tr -s '[:space:]' '\n' | grep -v '^$' | sort -u)"
    printf '%s\n' "$names" | grep -Fqx "$domain" || continue
    if printf '%s\n' "$names" | grep -qvE '^(auth|api)\.aurexis\.cc$'; then
      fail "$f also serves [$(printf '%s' "$names" | tr '\n' ' ')] and claims $domain; resolve this by hand rather than have a deployment disable someone else's vhost"
    fi
    log "disabling conflicting vhost $f (backed up)"
    $SUDO cp -a "$f" "$BACKUP_DIR/$(basename "$f").conflict"
    $SUDO rm -f "$f"
  done
}

backup_existing() {
  local domain="$1" c
  c="$(conf_path "$domain")"
  if [ -e "$c" ]; then
    $SUDO cp -a "$c" "$BACKUP_DIR/$(basename "$c")"
  fi
}

install_conf() {
  local domain="$1" src="$2" c e
  c="$(conf_path "$domain")"
  e="$(enabled_path "$domain")"
  $SUDO install -m 644 "$src" "$c"
  if [ "$LAYOUT" = sites ] && [ ! -e "$e" ]; then
    $SUDO ln -sfn "$c" "$e"
  fi
}

restore_backup() {
  log "restoring the previous nginx configuration"
  local domain c e b
  for domain in "${DOMAINS[@]}"; do
    c="$(conf_path "$domain")"
    e="$(enabled_path "$domain")"
    b="$BACKUP_DIR/$(basename "$c")"
    if [ -e "$b" ]; then
      $SUDO install -m 644 "$b" "$c"
    else
      $SUDO rm -f "$c"
      if [ "$LAYOUT" = sites ]; then
        $SUDO rm -f "$e"
      fi
    fi
  done
}

# Validate, then reload. A configuration that does not pass `nginx -t` is never reloaded,
# and a failure puts the previous files back before returning non-zero.
nginx_apply() {
  if ! $SUDO nginx -t; then
    log "nginx -t rejected the new configuration"
    restore_backup
    $SUDO nginx -t \
      || fail "nginx is invalid even after restoring the backup — manual repair needed"
    fail "nginx -t failed; previous configuration restored and left running"
  fi
  $SUDO nginx -s reload || $SUDO systemctl reload nginx || fail "nginx reload failed"
  log "nginx reloaded"
}

# --------------------------------------------------------------------------------------
# Public verification
# --------------------------------------------------------------------------------------

check_public() {
  local url="$1" want="$2" tries="${3:-12}" i=1 code
  while :; do
    # No -k anywhere: an unverifiable certificate is a failed deployment, not a warning.
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$url" 2>/dev/null)"
    code="${code:-000}"
    if [ "$code" = "$want" ]; then
      log "OK  $url -> $code"
      return 0
    fi
    if [ "$i" -ge "$tries" ]; then
      break
    fi
    i=$((i + 1))
    sleep 5
  done
  log "FAILED $url -> $code (wanted $want)"
  log "diagnostics:"
  curl -sS -o /dev/null \
    -w '  http=%{http_code} tls_verify=%{ssl_verify_result} time=%{time_total}s\n' \
    --max-time 20 "$url" 2>&1 || true
  return 1
}

verify_public() {
  log "verifying public endpoints"
  check_public "https://auth.aurexis.cc/docs" 200  || fail "https://auth.aurexis.cc/docs is not serving 200"
  check_public "https://api.aurexis.cc/docs" 200   || fail "https://api.aurexis.cc/docs is not serving 200"
  check_public "https://api.aurexis.cc/health" 200 || fail "https://api.aurexis.cc/health is not serving 200"
  check_public "https://api.aurexis.cc/ready" 200  || fail "https://api.aurexis.cc/ready is not serving 200"
}

# --------------------------------------------------------------------------------------

main() {
  local expected_ip="${1:-}"
  if [ -z "$expected_ip" ]; then
    fail "usage: $0 <expected-public-ip>"
  fi

  # An email is only needed when something has to be issued, so a run that reuses
  # existing certificates does not depend on the secret being set.
  if [ -z "${LETSENCRYPT_EMAIL:-}" ] && [ -r "$DEPLOY_ROOT/.letsencrypt-email" ]; then
    LETSENCRYPT_EMAIL="$(tr -d '\r\n' < "$DEPLOY_ROOT/.letsencrypt-email")"
    export LETSENCRYPT_EMAIL
  fi

  require_tools
  detect_layout

  BACKUP_DIR="$BACKUP_ROOT/$(date -u +%Y%m%dT%H%M%SZ)"
  $SUDO install -d -m 700 "$BACKUP_DIR"
  log "backups for this run: $BACKUP_DIR"

  check_local_services
  check_dns "$expected_ip"

  if ! ipv6_available; then
    log "WARN host has no IPv6; [::] listeners omitted"
  fi

  local domain needs_reload=0
  DOMAIN_CONFIG_TMP="$(mktemp -d)"
  trap 'rm -rf "$DOMAIN_CONFIG_TMP"' EXIT

  # Pass 1 — make sure every domain has a certificate, publishing HTTP-only where one is
  # missing so the challenge can be answered.
  for domain in "${DOMAINS[@]}"; do
    backup_existing "$domain"
    disable_conflicts "$domain"
    if cert_is_usable "$domain"; then
      log "$domain: certificate present and outside the ${RENEW_WINDOW_DAYS}-day window; reusing"
      continue
    fi
    if cert_exists_at_all "$domain"; then
      log "$domain: certificate exists but is near expiry; certbot.timer owns the renewal, reusing it now"
      continue
    fi
    log "$domain: no certificate; installing HTTP-only bootstrap"
    write_bootstrap_conf "$domain" "$DOMAIN_CONFIG_TMP/$domain.bootstrap"
    install_conf "$domain" "$DOMAIN_CONFIG_TMP/$domain.bootstrap"
    needs_reload=1
  done

  if [ "$needs_reload" = 1 ]; then
    nginx_apply
    for domain in "${DOMAINS[@]}"; do
      if ! cert_exists_at_all "$domain"; then
        issue_cert "$domain"
      fi
    done
  fi

  # Pass 2 — every certificate now exists, so the TLS vhosts can be installed safely.
  for domain in "${DOMAINS[@]}"; do
    cert_exists_at_all "$domain" \
      || fail "$domain still has no certificate; refusing to install a TLS vhost that would break nginx"
    render_conf "$NGINX_SRC/$domain.conf" "$DOMAIN_CONFIG_TMP/$domain.conf"
    if [ -e "$(conf_path "$domain")" ] && cmp -s "$DOMAIN_CONFIG_TMP/$domain.conf" "$(conf_path "$domain")"; then
      log "$domain: configuration already current"
      continue
    fi
    log "$domain: installing configuration"
    install_conf "$domain" "$DOMAIN_CONFIG_TMP/$domain.conf"
    needs_reload=1
  done

  if [ "$needs_reload" = 1 ]; then
    nginx_apply
  else
    # Still prove the running configuration is valid, even on a no-op run.
    $SUDO nginx -t
    log "no changes; nginx left as it is"
  fi

  ensure_renewal
  verify_public
  log "done"
}

main "$@"
