# Environment access directory

These are configured addresses, not confirmation that DNS or deployment is live.

The original CI, changed-service CD and release-record stages are preserved.
A main release promotes dev -> test -> production, with approval at each environment.
After dev succeeds, the tester may approve test to pull the SAME frozen images; production follows only after successful test and its own approval.

Environment data, keys, networks, container names and ports are isolated.
Management endpoints require the environment-specific Nginx password.

## dev

| Service | Address | Access |
| --- | --- | --- |
| superagent | [https://dev-superagent-b43d422afead4b89c57c5034.aurexis.cc/](https://dev-superagent-b43d422afead4b89c57c5034.aurexis.cc/) | Application login / API credentials |
| sis | [https://dev-sis-b43d422afead4b89c57c5034.aurexis.cc/ui/](https://dev-sis-b43d422afead4b89c57c5034.aurexis.cc/ui/) | Application login / API credentials |
| api | [https://dev-api-b43d422afead4b89c57c5034.aurexis.cc/docs](https://dev-api-b43d422afead4b89c57c5034.aurexis.cc/docs) | Application login / API credentials |
| auth | [https://dev-auth-b43d422afead4b89c57c5034.aurexis.cc/docs](https://dev-auth-b43d422afead4b89c57c5034.aurexis.cc/docs) | Application login / API credentials |
| records | [https://dev-records-b43d422afead4b89c57c5034.aurexis.cc/docs](https://dev-records-b43d422afead4b89c57c5034.aurexis.cc/docs) | Application login / API credentials |
| attu | [https://dev-attu-b43d422afead4b89c57c5034.aurexis.cc/](https://dev-attu-b43d422afead4b89c57c5034.aurexis.cc/) | Management login |
| minio | [https://dev-minio-b43d422afead4b89c57c5034.aurexis.cc/](https://dev-minio-b43d422afead4b89c57c5034.aurexis.cc/) | Application login / API credentials |
| s3 | [https://dev-s3-b43d422afead4b89c57c5034.aurexis.cc/minio/health/live](https://dev-s3-b43d422afead4b89c57c5034.aurexis.cc/minio/health/live) | Application login / API credentials |
| milvus | [https://dev-milvus-b43d422afead4b89c57c5034.aurexis.cc/healthz](https://dev-milvus-b43d422afead4b89c57c5034.aurexis.cc/healthz) | Management login |
| etcd | [https://dev-etcd-b43d422afead4b89c57c5034.aurexis.cc/health](https://dev-etcd-b43d422afead4b89c57c5034.aurexis.cc/health) | Management login |

TCP services use SSH tunnels; their host ports bind to loopback only.

| Service | Server port | Local tunnel address |
| --- | --- | --- |
| POSTGRES | 15432 | `127.0.0.1:15432` |
| REDIS | 16379 | `127.0.0.1:16379` |
| MILVUS | 29530 | `127.0.0.1:29530` |

```sh
ssh -N -L 15432:127.0.0.1:15432 -L 16379:127.0.0.1:16379 -L 29530:127.0.0.1:29530 root@13.140.153.131
```
Use that environment's database credentials. PostgreSQL, Redis and Milvus TCP are not web pages.

## test

| Service | Address | Access |
| --- | --- | --- |
| superagent | [https://test-superagent-2c70f5816cf684210e323910.aurexis.cc/](https://test-superagent-2c70f5816cf684210e323910.aurexis.cc/) | Application login / API credentials |
| sis | [https://test-sis-2c70f5816cf684210e323910.aurexis.cc/ui/](https://test-sis-2c70f5816cf684210e323910.aurexis.cc/ui/) | Application login / API credentials |
| api | [https://test-api-2c70f5816cf684210e323910.aurexis.cc/docs](https://test-api-2c70f5816cf684210e323910.aurexis.cc/docs) | Application login / API credentials |
| auth | [https://test-auth-2c70f5816cf684210e323910.aurexis.cc/docs](https://test-auth-2c70f5816cf684210e323910.aurexis.cc/docs) | Application login / API credentials |
| records | [https://test-records-2c70f5816cf684210e323910.aurexis.cc/docs](https://test-records-2c70f5816cf684210e323910.aurexis.cc/docs) | Application login / API credentials |
| attu | [https://test-attu-2c70f5816cf684210e323910.aurexis.cc/](https://test-attu-2c70f5816cf684210e323910.aurexis.cc/) | Management login |
| minio | [https://test-minio-2c70f5816cf684210e323910.aurexis.cc/](https://test-minio-2c70f5816cf684210e323910.aurexis.cc/) | Application login / API credentials |
| s3 | [https://test-s3-2c70f5816cf684210e323910.aurexis.cc/minio/health/live](https://test-s3-2c70f5816cf684210e323910.aurexis.cc/minio/health/live) | Application login / API credentials |
| milvus | [https://test-milvus-2c70f5816cf684210e323910.aurexis.cc/healthz](https://test-milvus-2c70f5816cf684210e323910.aurexis.cc/healthz) | Management login |
| etcd | [https://test-etcd-2c70f5816cf684210e323910.aurexis.cc/health](https://test-etcd-2c70f5816cf684210e323910.aurexis.cc/health) | Management login |

TCP services use SSH tunnels; their host ports bind to loopback only.

| Service | Server port | Local tunnel address |
| --- | --- | --- |
| POSTGRES | 25432 | `127.0.0.1:25432` |
| REDIS | 26379 | `127.0.0.1:26379` |
| MILVUS | 39530 | `127.0.0.1:39530` |

```sh
ssh -N -L 25432:127.0.0.1:25432 -L 26379:127.0.0.1:26379 -L 39530:127.0.0.1:39530 root@13.140.153.131
```
Use that environment's database credentials. PostgreSQL, Redis and Milvus TCP are not web pages.

## production

| Service | Address | Access |
| --- | --- | --- |
| superagent | [https://superagent.aurexis.cc/](https://superagent.aurexis.cc/) | Application login / API credentials |
| sis | [https://sis.aurexis.cc/ui/](https://sis.aurexis.cc/ui/) | Application login / API credentials |

Only these two application hosts are published. Swagger routes return 404.
Authentication remains available at the SuperAgent origin under /v1/.
Old api/auth hosts return 404 after migration; remove their DNS A/AAAA records after verification.
