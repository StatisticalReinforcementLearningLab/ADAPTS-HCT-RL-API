# ADAPTS-HCT RL API — Error Codes

This is the document for maintaining the list of error codes.

Branch on `(status, title)`; `message` is human-readable and may change wording. `rid` is always present, never `null`.

## `POST /api/v1/register_group` (alias `/api/v1/add_group`)

| `status` | `title` | `message` | Cause | Host action |
|---|---|---|---|---|
| `201` | `Success` | `Group registered successfully.` | new dyad registered | — |
| `201` | `Success` | `Group consent window updated.` | re-registration with matching `member_list`; consent window overwritten | — |
| `400` | `Invalid Parameter` | `Request body is required.` | empty or non-JSON body | correct and re-send |
| `400` | `Invalid Parameter` | `group_id is required.` | field missing | correct and re-send |
| `400` | `Invalid Parameter` | `member_list is required.` | field missing | correct and re-send |
| `400` | `Invalid Parameter` | `consent_start_date is required.` | field missing | correct and re-send |
| `400` | `Invalid Parameter` | `consent_end_date is required.` | field missing | correct and re-send |
| `400` | `Invalid Parameter` | `consent_start_date must be a date in YYYY-MM-DD format; got <repr>.` | unparseable date (weekday **not** enforced) | correct and re-send |
| `400` | `Invalid Parameter` | `consent_end_date must be a date in YYYY-MM-DD format; got <repr>.` | unparseable date | correct and re-send |
| `409` | `Member Mismatch` | `group_id <id> is already registered with members [...]; membership cannot be changed via /register_group.` | re-registration with a differing `member_list` | do not retry; escalate to the RL team |
| `500` | `Internal Error` | `Internal server error.` | unhandled server exception | retry (upsert is idempotent); escalate if persistent |
| `503` | `Service Unavailable` | `Database write failed; retry with backoff.` | DB commit failed; no row written | retry with backoff |

Null discipline: the single offending field is `null`; all other parsed fields echo. On `409` the *submitted* (rejected) `member_list` is echoed.

## `POST /api/v1/action`

| `status` | `title` | `message` | Cause | Host action |
|---|---|---|---|---|
| `201` | `Success` | `Action requested successfully.` | decision minted | use `action` |
| `200` | `Duplicate Decision` | `Returning the existing action for (<group>, <type>, <idx>).` | `(group_id, decision_type, decision_idx)` already committed; original outputs replayed verbatim | **not a failure**; use `action`. `rid` is the original call's |
| `400` | `Invalid Parameter` | `Request body is required.` | empty or non-JSON body | fallback; correct and re-send |
| `400` | `Invalid Parameter` | `group_id is required.` / `group_id must be a string.` | missing / wrong type | fallback; correct and re-send |
| `400` | `Invalid Parameter` | `timestamp is required.` / `timestamp must be a string or datetime object.` | missing / wrong type | fallback; correct and re-send |
| `400` | `Invalid Parameter` | `decision_idx is required.` / `decision_idx must be an integer.` | missing / wrong type (JSON `bool` rejected) | fallback; correct and re-send |
| `400` | `Invalid Parameter` | `decision_type is required.` / `decision_type must be a string.` | missing / wrong type | fallback; correct and re-send |
| `400` | `Invalid Parameter` | `Invalid decision_type. Must be 'aya_message', 'cp_message', or 'dyad_game'.` | unrecognized agent | fallback; correct and re-send |
| `404` | `Unknown Group` | `group_id <id> is not registered.` | dyad not registered | fallback; register before the next decision |
| `409` | `No State Available` | `No /upload_data has been received for <id>; send a snapshot first.` | zero `data_uploads` rows for the dyad | fallback; send the first snapshot. **Open: `API-Spec.md:357` proposes serving a decision instead** |
| `500` | `Internal Error` | `Learner could not build the state: <detail>` | `make_state` failed on the server-projected context | fallback; infrastructure alert |
| `500` | `Internal Error` | `Failed to persist the action.` | `actions` commit failed, rolled back; sampler cursor already advanced in-memory | fallback; flag to the RL team (reproducibility) |
| `500` | `Internal Error` | `Learner raised an exception while sampling the action.` | catch-all, incl. sample buffer unavailable | fallback; infrastructure alert |
| `503` | `Model Not Initialized` | `The learner for <type> is not loaded; the deployment is not ready to serve decisions.` | learner not loaded / deployment not ready | fallback; **deployment alert**, not host-fixable |

On **any** `/action` failure the host applies the same local fallback: draw `Bernoulli(0.5)` and record `action_prob = 0.5`. Retry is always safe — a committed decision replays as `200`. Null on failure: `action`, `action_prob`, `warmup`, `state`, `model_param`, plus the offending request field.

## `POST /api/v1/upload_data`

| `status` | `title` | `message` | Cause | Host action |
|---|---|---|---|---|
| `201` | `Success` | `Snapshot accepted.` | row appended | — |
| `400` | `Invalid Parameter` | `group_id is required.` / `group_id must be a string.` | envelope field missing / wrong type | correct and re-send |
| `400` | `Invalid Parameter` | `timestamp is required.` | envelope field missing | correct and re-send |
| `400` | `Invalid Parameter` | `data is required.` / `data must be a dictionary.` | snapshot missing / not an object | correct and re-send |
| `400` | `Invalid Parameter` | `unknown field '<name>'.` | key not in `API-Spec.md` §4.1 (typos rejected, not ignored) | correct and re-send |
| `400` | `Invalid Parameter` | `<field> is required (full snapshot).` | §4.1 key omitted; send `"miss"` instead of omitting | correct and re-send |
| `400` | `Invalid Parameter` | `<field>: <type error>` | value failed its declared type (suffixes below) | correct and re-send |
| `404` | `Unknown Group` | `group_id <id> is not registered.` | dyad not registered | register, then re-send |
| `500` | `Internal Error` | `Internal server error.` | unhandled server exception | retry (append-only) |
| `503` | `Service Unavailable` | `Database write failed; retry.` | DB commit failed; no row written | retry (append-only) |

Validation is fail-fast — only the first offending field is reported. `<type error>` suffixes:

| Field type | Suffix |
|---|---|
| `slot` | `must be 'am' or 'pm'.` |
| `positive_int` | `must be a positive integer.` |
| non-negative int | `must be a non-negative integer.` |
| `nonneg_float` | `must be a non-negative number.` |
| `unit_interval` | `must be a number between 0 and 1.` |
| `binary` | `must be 0 or 1.` |
| `binary_or_miss` | `must be 0, 1, or 'miss'.` |
| `float_or_miss` | `must be numeric or 'miss'.` |
| `engagement` | `must be one of 1, 2, 3, or 4.` |
| `engagement_or_miss` | `must be one of 1, 2, 3, 4, or 'miss'.` |
| `unit_interval_or_miss` | `must be a number between 0 and 1, or 'miss'.` |
| diary object | `must be a dictionary with diary items.` / `missing diary item '<item>'.` / `'<item>' must be numeric or 'miss'.` |
| `bool` | `must be true or false.` |

## `POST /api/v1/update`

| `status` | `title` | `message` | Cause | Action |
|---|---|---|---|---|
| `202` | `Update Accepted` | `Model update started.` | fit enqueued, running in a background thread | poll `model_update_requests` by `rid` |
| `400` | `Invalid Parameter` | `timestamp is required.` | field missing | correct and re-send |
| `503` | `Service Unavailable` | `Could not start the update; retry.` | enqueue failed; **no `model_update_requests` row** | retry; do not poll the returned `rid` |

Post-`202` fit outcomes are reported only in `model_update_requests` (`API-Spec.md` §5.6), never over HTTP:

| `status` | `completed_at` | `error_message` | Cause | Effect |
|---|---|---|---|---|
| `processing` | `NULL` | `NULL` | fit running, **or** background thread crashed / timed out | previous parameters stay active; apply a timeout and re-trigger |
| `completed` | set | `NULL` | fit finished, or batch was empty | new parameters published (previous retained if empty) |
| `failed` | set | set | fit raised, or sanity gate rejected the fit | fit not published; previous parameters stay active |

Per-action reward-derivation errors do not fail the fit: that action is skipped (`reward = NULL`) and the rest are derived.

## `GET /api/v1/monitor/alarms`

Read-only alarm pull (`API-Spec.md` §2.5). Query parameters, no request body.
No `404` — the endpoint is not dyad-scoped, so there is no identifier to fail
to resolve.

| `status` | `title` | `message` | Cause | Host action |
|---|---|---|---|---|
| `200` | `Success` | `Returning <n> alarms.` | window served; `n = 0` is normal and healthy | record the `alarms`, keyed by `event_id` |
| `400` | `Invalid Parameter` | `since must be an ISO-8601 timestamp; got <repr>.` | unparseable `since` | correct and re-send |
| `500` | `Internal Error` | `Internal server error.` | unhandled server exception | retry (read-only); escalate if persistent |
| `503` | `Service Unavailable` | `Monitoring store unavailable; retry with backoff.` | monitoring store unreadable | retry with backoff |

Retry is always safe — the pull is read-only and idempotent, and an overlapping
window is harmless because the host keys on `event_id`. Null on failure:
`count`, `alarms`.

A pull failure is **not** an `/action`-style incident: no decision depends on
it and no alarm is lost (the feed is a window over persisted state, not a
draining queue). The condition worth alerting on is a *missing* successful
pull, not an empty one.

## Any path — responses that are **not** the envelope

Body is `{"error": "Internal server error"}` with no `rid`, `title`, or `server_timestamp` (global `@app.errorhandler(Exception)`, `app/__init__.py:140`).

| `status` | Cause | Host action |
|---|---|---|
| `500` | unknown route (e.g. `GET /api/v1/nope`) — the catch-all intercepts Werkzeug's `404` | check the URL |
| `500` | wrong HTTP method (e.g. `GET /api/v1/action`) — intercepts Werkzeug's `405` | check the method |
| — | API unreachable / process down | connection refused, DNS failure, or client timeout; `/action` falls back locally |
| — | request times out mid-flight | server may still have committed; on `/action` retry and read the `200` replay |

## Spec / implementation gaps

| Gap | Detail |
|---|---|
| `500 Internal Error` on `/register_group` | `app/routes/group.py:165`; absent from `API-Spec.md` §2.1 error table |
| `500 Internal Error` on `/upload_data` | `app/routes/data.py:92`; absent from §2.3 error table |
| `400 Invalid Parameter` on `/update` | `app/routes/update.py:211`; §2.4 table lists only `503` |
| Monday rule not enforced | §2.1 requires `consent_start_date` to be a Monday; `app/routes/group.py:check_fields` checks format only. Matches the `API-Spec.md:163` TODO |
| Routing errors return `500` | not `404` / `405`, and not the envelope; undocumented in the spec |
| `/action` `409` may be retired | `API-Spec.md:357` TODO proposes serving a decision with no upload history |
