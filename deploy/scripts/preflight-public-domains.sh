#!/usr/bin/env bash
#
# Read-only pre-flight for configure-public-domains.sh.
#
# Answers, before a deployment does it the expensive way, every question that script
# depends on: does nginx exist and in which layout, does anything already claim these
# names, is there a certificate, can the ACME challenge actually reach port 80, do the
# services answer on loopback, and does DNS say what we think it says.
#
# It changes nothing. No file is written, no service reloaded, no certificate requested.
# Run it as often as you like.
#
# The deployment runs this itself, immediately before configure-public-domains.sh, so its
# report is in the workflow log of every release and nobody has to open a session on the
# server to find out why a domain did not come up. It can also be run by hand:
#
#   ssh root@HOST bash /opt/superagent/deploy/scripts/preflight-public-domains.sh

set -uo pipefail

DOMAINS=(auth.aurexis.cc api.aurexis.cc)
EXPECTED_IP="${1:-13.140.153.131}"

PASS=0
WARN=0
FAIL=0

ok()   { printf '  \033[32mOK\033[0m    %s\n' "$*"; PASS=$((PASS + 1)); }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$*"; WARN=$((WARN + 1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; FAIL=$((FAIL + 1)); }
head_() { printf '\n=== %s ===\n' "$*"; }

# --------------------------------------------------------------------------------------
head_ "1. nginx"
if command -v nginx >/dev/null 2>&1; then
  ok "nginx present: $(nginx -v 2>&1 | sed 's/nginx version: //')"
  if nginx -t >/dev/null 2>&1; then
    ok "the running configuration currently passes nginx -t"
  else
    bad "nginx -t already fails BEFORE any change — fix this first, the deploy will refuse to reload"
    nginx -t 2>&1 | sed 's/^/        /'
  fi
  if [ -d /etc/nginx/sites-available ] && [ -d /etc/nginx/sites-enabled ]; then
    ok "layout: sites-available + sites-enabled"
  elif [ -d /etc/nginx/conf.d ]; then
    ok "layout: conf.d"
  else
    bad "no sites-available or conf.d directory — the script cannot install a vhost"
  fi
else
  bad "nginx is NOT installed — install it before deploying"
fi

# --------------------------------------------------------------------------------------
head_ "2. vhosts already claiming these names   <-- the most likely blocker"
for domain in "${DOMAINS[@]}"; do
  found=0
  for f in /etc/nginx/sites-enabled/* /etc/nginx/sites-available/* /etc/nginx/conf.d/*.conf; do
    [ -e "$f" ] || continue
    names="$(grep -hoE '^[[:space:]]*server_name[^;]*' "$f" 2>/dev/null \
      | sed 's/^[[:space:]]*server_name[[:space:]]*//' \
      | tr -s '[:space:]' '\n' | grep -v '^$' | sort -u)"
    printf '%s\n' "$names" | grep -Fqx "$domain" || continue
    found=1
    other="$(printf '%s\n' "$names" | grep -vE '^(auth|api)\.aurexis\.cc$' || true)"
    if [ -n "$other" ]; then
      bad "$domain is served by $f, which ALSO serves: $(printf '%s' "$other" | tr '\n' ' ')"
      printf '        The deployment will stop here rather than disable a shared vhost.\n'
      printf '        Fix: move %s into its own file, then re-run this check.\n' "$domain"
    else
      warn "$domain is served by $f (only these domains) — it will be backed up and replaced"
    fi
  done
  [ "$found" -eq 0 ] && ok "$domain has no existing vhost; a clean install"
done

# --------------------------------------------------------------------------------------
head_ "3. certificates"
if command -v certbot >/dev/null 2>&1; then
  ok "certbot present: $(certbot --version 2>&1)"
else
  warn "certbot missing — the deploy will apt-get install it (needs working apt)"
fi
for domain in "${DOMAINS[@]}"; do
  full="/etc/letsencrypt/live/$domain/fullchain.pem"
  if [ -s "$full" ]; then
    if openssl x509 -in "$full" -noout -checkend $((30 * 86400)) >/dev/null 2>&1; then
      ok "$domain: certificate exists, >30 days left — will be reused, nothing requested"
    else
      warn "$domain: certificate exists but expires within 30 days (certbot.timer renews it)"
    fi
    printf '        expires: %s\n' "$(openssl x509 -in "$full" -noout -enddate 2>/dev/null | cut -d= -f2)"
  else
    warn "$domain: NO certificate — one will be requested. Port 80 and DNS must be correct."
  fi
done
if command -v systemctl >/dev/null 2>&1; then
  if systemctl list-unit-files 2>/dev/null | grep -q '^certbot\.timer'; then
    ok "certbot.timer exists (auto-renewal available)"
  else
    warn "no certbot.timer — renewal would need another scheduler"
  fi
fi

# --------------------------------------------------------------------------------------
head_ "4. port 80 reachable   <-- required for the ACME challenge"
# Pick whichever socket tool the host has. Reporting "nothing is listening" because the
# tool is missing would fail a deployment over a diagnostic, so an absent tool is a
# warning and never a failure.
listeners=""
if command -v ss >/dev/null 2>&1; then
  listeners="$(ss -lnt 2>/dev/null)"
elif command -v netstat >/dev/null 2>&1; then
  listeners="$(netstat -lnt 2>/dev/null)"
fi
if [ -z "$listeners" ]; then
  warn "neither ss nor netstat is available — cannot confirm port 80 is listening"
elif printf '%s\n' "$listeners" | grep -qE '[:.]80[[:space:]]'; then
  ok "something is listening on port 80"
  printf '%s\n' "$listeners" | grep -E '[:.]80[[:space:]]' | sed 's/^/        /'
else
  bad "nothing is listening on port 80 — the ACME http-01 challenge cannot be answered"
fi
if command -v ufw >/dev/null 2>&1 && ufw status 2>/dev/null | grep -q '^Status: active'; then
  if ufw status 2>/dev/null | grep -qE '(^80|Nginx|80/tcp)'; then
    ok "ufw is active and appears to allow port 80"
  else
    bad "ufw is active and does not obviously allow port 80 — Let's Encrypt will not reach this host"
    ufw status 2>/dev/null | sed 's/^/        /'
  fi
else
  ok "no active ufw rule set to block port 80"
fi

# --------------------------------------------------------------------------------------
head_ "5. DNS"
if command -v dig >/dev/null 2>&1; then
  ok "dig present (public resolver will be queried directly)"
  resolver="dig"
else
  warn "dig missing — the deploy installs dnsutils; falling back to the local resolver here"
  resolver="getent"
fi
for domain in "${DOMAINS[@]}"; do
  if [ "$resolver" = dig ]; then
    got="$(dig +short @1.1.1.1 "$domain" A 2>/dev/null | grep -E '^[0-9]+\.' | tr '\n' ' ')"
  else
    got="$(getent ahostsv4 "$domain" 2>/dev/null | awk '{print $1}' | sort -u | tr '\n' ' ')"
  fi
  got="$(printf '%s' "$got" | sed 's/[[:space:]]*$//')"
  if [ -z "$got" ]; then
    bad "$domain does not resolve at all"
  elif [ "$got" = "$EXPECTED_IP" ]; then
    ok "$domain -> $got"
  else
    bad "$domain -> $got (expected $EXPECTED_IP)"
  fi
done

# --------------------------------------------------------------------------------------
head_ "6. services on loopback"
probe() {
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$1" 2>/dev/null)"
  code="${code:-000}"
  if [ "$code" = "$2" ]; then
    ok "$1 -> $code"
  else
    bad "$1 -> $code (expected $2)"
  fi
}
probe "http://127.0.0.1:8200/docs" 200
probe "http://127.0.0.1:8000/health" 200
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://127.0.0.1:8000/ready 2>/dev/null)"
code="${code:-000}"
if [ "$code" = "200" ]; then
  ok "http://127.0.0.1:8000/ready -> 200"
else
  warn "http://127.0.0.1:8000/ready -> $code (503 while the embedder warms up is normal; the deploy waits 5 minutes)"
  curl -s --max-time 10 http://127.0.0.1:8000/ready 2>/dev/null | sed 's/^/        /'
  echo
fi

# --------------------------------------------------------------------------------------
head_ "7. IPv6"
if [ -s /proc/net/if_inet6 ]; then
  ok "IPv6 available — [::] listeners will be kept"
else
  warn "no IPv6 — the script drops [::] listeners automatically (not a failure)"
fi

# --------------------------------------------------------------------------------------
printf '\n=== verdict ===\n'
printf '  passed: %s   warnings: %s   failures: %s\n' "$PASS" "$WARN" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
  printf '\n  Fix the FAIL lines above before merging. The deployment would stop on them.\n'
  exit 1
fi
printf '\n  Nothing blocking. Warnings are expected on a first run.\n'
exit 0
