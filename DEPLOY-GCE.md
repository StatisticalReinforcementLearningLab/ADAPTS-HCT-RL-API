# Deploy to a Google Compute Engine VM

The production stack consists of one Gunicorn worker, PostgreSQL, and the
independent monitoring loop. Runtime state is stored in Docker volumes. The API
binds to `127.0.0.1:8000` by default because the API contract has no
application-layer authentication.

## VM prerequisites

Use an Ubuntu 24.04 LTS VM with at least 2 vCPU, 4 GB RAM, and a 30 GB balanced
persistent disk. Keep the VM and the calling host server on the same VPC. Do not
create a public ingress firewall rule for port 8000.

Install Docker from Docker's official Ubuntu repository, including the Compose
plugin, and clone this repository into `/opt/adapts/ADAPTS-HCT-RL-API`.

## Configure and start

```sh
cd /opt/adapts/ADAPTS-HCT-RL-API
cp .env.production.example .env.production
openssl rand -hex 32
# Put the generated value in POSTGRES_PASSWORD in .env.production.
docker compose --env-file .env.production build
docker compose --env-file .env.production up -d
docker compose --env-file .env.production ps
curl --fail http://127.0.0.1:8000/healthz
docker compose --env-file .env.production exec api flask --app run:app hyperparams show
docker compose --env-file .env.production exec monitor flask --app run:app monitor status
```

For temporary operator access from a workstation, use an SSH tunnel:

```sh
gcloud compute ssh VM_NAME --zone ZONE -- -N -L 8000:127.0.0.1:8000
```

Then call `http://127.0.0.1:8000/api/v1/...` locally.

If the production host server calls over the VPC, set `ADAPTS_BIND_ADDRESS` to
the VM's private IP and add a narrowly scoped ingress firewall rule allowing
TCP 8000 only from the host server's service account or private IP range.

## Update and rollback

Before an update, snapshot the VM disk or back up all Docker volumes. Then:

```sh
git pull --ff-only
docker compose --env-file .env.production build
docker compose --env-file .env.production up -d
curl --fail http://127.0.0.1:8000/healthz
```

Do not scale the API above one Gunicorn worker: the deterministic sample cursor
is process-local and is intentionally serialized with `--workers=1 --threads=1`.
