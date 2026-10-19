# Bruno smoke collection (`main/`)

A minimal collection that fires every public endpoint once, in dependency
order, and asserts the HTTP status. Use it to confirm a running API answers
each route — not to exercise the learner.

## Requests (run in `seq` order)

| seq | request           | endpoint                  | expect |
|-----|-------------------|---------------------------|--------|
| 1   | Health Check      | `GET /healthz`                    | 200 |
| 2   | Register Group    | `POST /api/v1/register_group`     | 201 |
| 3   | Upload Data       | `POST /api/v1/upload_data`        | 201 |
| 4   | Get AYA Decision  | `POST /api/v1/action`              | 201 |
| 5   | Get CP Decision   | `POST /api/v1/action`              | 201 |
| 6   | Get Game Decision | `POST /api/v1/action`              | 201 |
| 7   | Run Update        | `POST /api/v1/update`              | 202 |
| 8   | Get Alarms        | `GET /api/v1/monitor/alarms`      | 200 |

Order matters: `register_group` registers the dyad, `upload_data` seeds the snapshot
that `/action` reads (without it `/action` returns 409), and `update` runs last.

Register Group's pre-request script sets a fresh `group_id`, Detroit-local
timestamp, consent window, and study week into run variables that every later
request reuses, so the whole collection is re-runnable without resetting the
database.

With fewer than 5 registered dyads the API serves warm-up (randomized) actions;
the requests still return 201, which is all this collection checks.

## Environments

`environments/local.bru` → `base_url = http://127.0.0.1:5001`
`environments/gcp-iap.bru` → `base_url = http://127.0.0.1:8000` through the
Google IAP SSH tunnel below.

Select one in the Bruno top-right environment dropdown (or `--env` on the CLI).

## Run it

GUI: open the `bruno/main` collection, pick an environment, then **Run** the
collection (or send requests top-to-bottom).

For the deployed GCP VM, start this tunnel before opening Bruno:

```sh
gcloud compute ssh development-rl \
  --project adapts-hct-e1f6 \
  --zone us-east1-b \
  --tunnel-through-iap \
  -- -N -L 8000:127.0.0.1:8000
```

Keep that terminal open and select the `gcp-iap` environment in Bruno.

CLI (`npm i -g @usebruno/cli`):

```sh
# start the API first (from the repo root)
conda activate justin_rl_api && flask run --port 5001

# then, from this folder
cd bruno/main
bru run --env local
```
