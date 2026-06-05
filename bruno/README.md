# Bruno smoke collection (`main/`)

A minimal collection that fires every public endpoint once, in dependency
order, and asserts the HTTP status. Use it to confirm a running API answers
each route — not to exercise the learner.

## Requests (run in `seq` order)

| seq | request           | endpoint                  | expect |
|-----|-------------------|---------------------------|--------|
| 1   | Add Group         | `POST /api/v1/add_group`  | 201    |
| 2   | Upload Data       | `POST /api/v1/upload_data`| 201    |
| 3   | Get AYA Decision  | `POST /api/v1/action`     | 201    |
| 4   | Get CP Decision   | `POST /api/v1/action`     | 201    |
| 5   | Get Game Decision | `POST /api/v1/action`     | 201    |
| 6   | Run Update        | `POST /api/v1/update`     | 202    |

Order matters: `add_group` registers the dyad, `upload_data` seeds the snapshot
that `/action` reads (without it `/action` returns 409), and `update` runs last.

Add Group's pre-request script sets a fresh `group_id` (`dyad_<timestamp>`) into
a run variable that every later request reuses, so the whole collection is
re-runnable without `flask reset-db` (a duplicate `group_id` would 400).

With fewer than 5 registered dyads the API serves warm-up (randomized) actions;
the requests still return 201, which is all this collection checks.

## Environments

`environments/local.yml`  → `base_url = http://127.0.0.1:5001`
`environments/tunnel.yml` → `base_url = <cloudflare tunnel>`

Select one in the Bruno top-right environment dropdown (or `--env` on the CLI).

## Run it

GUI: open the `bruno/main` collection, pick an environment, then **Run** the
collection (or send requests top-to-bottom).

CLI (`npm i -g @usebruno/cli`):

```sh
# start the API first (from the repo root)
conda activate justin_rl_api && flask run --port 5001

# then, from this folder
cd bruno/main
bru run --env local
```
