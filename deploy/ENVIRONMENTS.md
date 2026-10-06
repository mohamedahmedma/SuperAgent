# Deploying the three environments

The configured addresses and SSH tunnels are in [ENVIRONMENT_LINKS.md](ENVIRONMENT_LINKS.md).
They become usable only after DNS, credentials and deployment have been configured.
The source of truth is `deploy/environments.json`. Regenerate profiles, Nginx routes
and links with `python deploy/scripts/render-environments.py`.

## Pipeline

The original workflow stages and granular changed-service build jobs are retained:
CI, changed-service detection/CD, deployment, and recording the last successful
production release. Pushes to `main` (or manual runs on `main`) produce a candidate.
`develop` continues running standalone CI and does not deploy a separate release.

The deployment part is now **dev approval -> dev deploy/verify -> test approval ->
test deploy/verify -> production approval -> production deploy/verify**. Each
approval is optional: the reviewer can leave the release waiting rather than pull
it into that environment. Test is never offered before dev succeeds, and production
is never offered before test succeeds. There is no target selector that skips an
upstream environment.

The tester selects **Review deployments / Approve and deploy** on the same workflow
run to pull the accepted development candidate. The production reviewer does the
same after successful testing. No image is rebuilt when moving between estates.
A frozen manifest combines new images for changed services and the exact image
versions from the last successful production manifest for unchanged services.
All three estates download the same immutable workflow artifact. Missing or expired
baseline artifacts rebuild all services at CD, never substitute a mutable stable tag.
The original `production-deployed` tag advances only after production succeeds.

The frontend uses same-origin `/v1/` authentication in every environment, so a dev
user is never sent to production Identity.

By default, all estates use the existing server `13.140.153.131`. Separate servers
are supported with environment variables `DEPLOY_HOST`, `DEPLOY_PUBLIC_IP`,
`DEPLOY_PORT` and `DEPLOY_USER`; update the corresponding DNS records too.
Each directory contains its own persistent `.env` and release manifest:

| Environment | Server directory | Compose project | Branch |
| --- | --- | --- | --- |
| dev | `/opt/superagent-dev` | `superagent-dev` | `main` candidate, dev approval |
| test | `/opt/superagent-test` | `superagent-test` | Same accepted dev candidate, test approval |
| production | `/opt/superagent` | `superagent` | Same accepted test candidate, production approval |

Production retains the previous project, container, network and volume names.
Nonproduction uses new names and different loopback ports. No production database
is copied into a nonproduction environment. Deployment never deletes volumes.

## GitHub configuration required before enabling deployment

A repository administrator creates `dev` and `test` environments and preserves the
existing `Production` environment and its reviewers. GitHub environment names are
case-insensitive. Dev and test must have the SAME reviewers, self-review setting,
wait timer and administrator-bypass setting as Production. All estates accept the
`main` release ref, rather than requiring separate dev/test branches.

An administrator can copy and verify the policy without changing Production:

```sh
python deploy/scripts/approval-policy.py --repo mohamedahmedma/SuperAgent --plan
python deploy/scripts/approval-policy.py --repo mohamedahmedma/SuperAgent --apply
```

With an existing selected-branch policy, copy the corresponding branch rules in
GitHub Settings too; the helper declines to invent those rules. The workflow checks
that all three protected environments exist and match BEFORE publishing or deploying.
A missing or unprotected dev/test environment fails closed, rather than being implicitly
created without reviewers. Do not disable any approval to make a first run pass.

| Secret | Scope and purpose |
| --- | --- |
| `DEPLOY_PASSWORD` | SSH credential; existing repository secret can be retained. |
| `DEV_ENV_FILE` | Complete independent dev `.env`, preferably a dev environment secret. |
| `TEST_ENV_FILE` | Complete independent test `.env`, preferably a test environment secret. |
| `PROD_ENV_FILE` | Existing production bootstrap, used only if the server file is absent. |
| `GATEWAY_HTPASSWD` | A different htpasswd value in each nonproduction environment. |
| `LETSENCRYPT_EMAIL` | Existing certificate-registration secret. |
| `DEPLOY_KNOWN_HOSTS` | Verified OpenSSH server key(s); otherwise the existing keyscan bootstrap is used. |

Generate an administration password hash with `htpasswd -nB USERNAME`, then put the
entire `username:hash` line into that environment's `GATEWAY_HTPASSWD` secret.
Attu, etcd and Milvus HTTP require this login. MinIO Console uses its native login;
S3 requests use MinIO access keys, since an additional Basic Auth layer would break
S3 request signatures. Application APIs keep their own authentication and API keys.
S3's unauthenticated health endpoint is intentionally public; bucket operations
still require their native access policy.

Give each nonproduction `.env` independent database passwords, service keys, JWT
keys and administrator passwords. Configure test WhatsApp endpoints separately
instead of reusing production phone-number/webhook credentials. Provider API
credentials may be chosen independently. The pipeline never falls back from a
missing dev/test bootstrap file to `PROD_ENV_FILE` or overwrites a present `.env`.

## DNS and production migration

`dns-records.csv` lists all 22 A records for Namecheap. Update only these records;
preserve MX/TXT and unrelated hosts. Existing A/AAAA records for a selected host
must agree with its server, and IPv6 records require that server's working IPv6.
The selected environment's DNS is checked before issuing certificates.

Production publishes only `superagent.aurexis.cc` and `sis.aurexis.cc`.
SIS moves from `superagent.aurexis.cc/ui/` to `sis.aurexis.cc/ui/`; it has its own
API origin. SuperAgent `/v1/` becomes the Identity route rather than the SIS route.
Update bookmarks and any external integration using the old routing. Update the
Meta WhatsApp webhook URL to the Identity webhook path at the SuperAgent origin
before retiring `auth.aurexis.cc` if the current integration uses that hostname.

After the new production UI and SIS pass verification, the deployment replaces
old `api` and `auth` vhosts with refusal responses. Their TLS certificates are kept
for rollback; remove their DNS A/AAAA records in Namecheap after confirming the
new integration routes work. No other host is removed.

Production hides Swagger, ReDoc and OpenAPI at both public origins. Necessary
application APIs remain behind the two origins. PostgreSQL, Redis, etcd, MinIO,
Milvus and Attu have no production public hostname in this configuration.

## Verification and rollback

CI checks generated topology and shell syntax alongside the existing application
suite. Deploy pulls the frozen manifest's application images before recreating them, reconciles and
authenticates PostgreSQL using the existing `apply-env.sh`, waits for application
health and backend readiness, then proves the running images match each service version in the accepted manifest.
It installs certificates and Nginx only after application health succeeds.
The server and the GitHub runner both check public URLs; management hosts must
refuse unauthenticated requests. These checks do not substitute for an operator
testing the complete authenticated administration and WhatsApp flows.

On an application release failure, the previous image manifest is restored and
the prior applications are recreated. Database migrations are not automatically
reversed: keep database backups before releases with incompatible schema changes.
The previous Nginx configuration is backed up under `/var/backups/superagent-nginx`.
Nginx changes from concurrent environment releases are serialized on the server.
An outside-server verification failure marks the workflow failed and requires
operator diagnosis; it does not automatically rewind database state.

All configuration in this change is prepared locally. It does not confirm that
Namecheap records, GitHub environments, bootstrap secrets, or live deployments
have been installed.
