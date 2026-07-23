# ADAPTS-HCT RL API — Specification

The contract from the host's perspective:

1. **`/upload_data` is a flat "latest values" snapshot.** No context/outcome
   distinction; the host sends the latest value of every variable in §4.1. No
   `decision_type` or `decision_idx`. See §2.3 and §4.
2. **`/action` is context-free.** The API reads the dyad's most recent
   uploaded snapshot and projects out the subset the requested `decision_type`
   needs. See §2.2.
3. **`/update` issues no callback.** The monitoring algorithm schedules updates
   and watches for completion via the `model_update_requests` table; the API
   does not POST back. See §2.4.
4. **Every response uses one common envelope.** A given endpoint returns the
   same set of fields regardless of success, failure, or failure type. Every
   response carries `status`, `title`, `message`, and `server_timestamp`; the
   endpoint-specific fields are added on top. On failure a field is `null` only
   when the failure makes its value unknown — everything the API still knows
   (valid identifiers, per-call ids like `rid`) is echoed. See the §2 preamble.

---

## 1. Overview

The RL API is a Flask REST service. The study **host** (app backend + scheduler, owned by the Michigan team) calls it at each decision time; the API returns a randomized action, logs the latest values of every variable the host posts, and re-fits the learner on a periodic schedule triggered by
the monitoring algorithm.

- **The API is the system of record for all model state.** The host relays raw values and does not compute model parameters. The algorithm API takes inputs the context and generates a sampling probability and the action for each decision call.
- **Three decision types (agents)**, all served by one learner with cross-dyad pooling:
  `aya_message` (twice daily), `cp_message` (daily), `dyad_game` (weekly).
- **Reproducibility:** every action and update is deterministic given (i) a pre-sampled random-primitive buffer (`.npz`) and (ii) the ordered event log.

---

## 2. Endpoints

### Common response format

Every endpoint returns the **same envelope on every call** — success or
failure, regardless of failure type. Five fields are always present:

| Field | Type | Notes |
|---|---|---|
| `status` | int | HTTP status code, echoed in the body (equals the response's HTTP status, so a client may branch on either). Success: `201` (`202` for `/update`; `200` for an idempotent `/action` replay, §2.2). Client errors: `400`, `404`, `409`. Server errors: `500`, `503`. |
| `title` | string | stable outcome label, e.g. `"Success"`, `"Invalid Parameter"`, `"Unknown Group"`, `"No State Available"`, `"Duplicate Decision"`, `"Internal Error"`, `"Service Unavailable"`; distinguishes failures that share one HTTP code |
| `message` | string | human-readable detail, e.g. `"decision_type must be one of aya_message / cp_message / dyad_game."` |
| `server_timestamp` | ISO-8601 string | server time the response was produced. Named distinctly from the request-body `timestamp` (the host's decision/upload time) to avoid the request/response key collision. |
| `rid` | string | per-call request id the API assigns to **every** call on **every** endpoint; **always present, even on failure** (assigned on receipt, before any write). Where the call writes a primary row, the same `rid` is persisted on it — `groups.rid`, `actions.rid`, `data_uploads.rid`, `model_update_requests.rid` — so the host can reconcile a response with server state. It additionally serves as the `/action` idempotency-replay handle (§2.2) and the key the monitoring algorithm polls in `model_update_requests` for `/update` (§2.4, §5.6). |

Each endpoint adds its own fields on top (listed per endpoint below), so the
response shape is the same whether the call succeeds or fails. **On failure, a
field is `null` only when the failure makes its value unknown or meaningless;
every value the API still knows is echoed as its real value.** Concretely:

- the per-call `rid` the API always assigns is **never** `null`, on success or
  failure and on every endpoint;
- request fields that parsed and validated (e.g. a `group_id` that is a real
  registered dyad) are echoed as sent;
- only fields the failure prevents from being produced are `null` — e.g.
  `action` / `action_prob` when no decision could be made, or the one field that
  was malformed.

The host must not read the `null` fields; each endpoint's error table gives the
recommended action. Example failure envelope (`/action`, `409` — the request is
well-formed and the dyad is valid, so the identifiers and per-call `rid` are
echoed; only the un-producible decision outputs are `null`):

```json
{
  "status": 409,
  "title": "No State Available",
  "message": "No /upload_data has been received for dyad_007; send a snapshot first.",
  "server_timestamp": "2026-06-10T07:30:01",
  "group_id": "dyad_007", "decision_type": "aya_message", "decision_idx": 29,
  "rid": "a1b2c3d4",
  "action": null, "action_prob": null, "warmup": null,
  "state": null, "model_param": null
}
```

### 2.1 `POST /api/v1/register_group` — register a dyad

Registers a dyad (a group of two participants) at recruitment.

Request:

| Field | Type | Notes |
|---|---|---|
| `group_id` | string | unique dyad identifier |
| `member_list` | list | participant identifiers, e.g. `[cp_id, aya_id]` |
| `consent_start_date` | `YYYY-MM-DD` | onboarding/consent complete; must be a **Monday** — it starts the study clock (`day_in_study = 1`) |
| `consent_end_date` | `YYYY-MM-DD` | active window end (≈ start + 100 days) |


Request body (example):

```json
{
  "group_id": "dyad_007",
  "member_list": ["cp_007", "aya_007"],
  "consent_start_date": "2026-05-25",
  "consent_end_date": "2026-09-02"
}
```

Response — common envelope plus:

| Field | Type | Notes |
|---|---|---|
| `group_id` | string | echo of the registered dyad |
| `member_list` | list | members recorded at first registration |
| `consent_start_date` | `YYYY-MM-DD` | echo (may have been updated on re-registration) |
| `consent_end_date` | `YYYY-MM-DD` | echo |

(The always-present per-call `rid` — persisted on the `groups` row, §5.1 — is
part of the common envelope; see the §2 preamble.)

```json
{
  "status": 201,
  "title": "Success",
  "message": "Group registered successfully.",
  "server_timestamp": "2026-05-25T14:03:12",
  "group_id": "dyad_007",
  "member_list": ["cp_007", "aya_007"],
  "consent_start_date": "2026-05-25",
  "consent_end_date": "2026-09-02",
  "rid": "xxxx"
}
```

**Path alias.** The canonical path is `/api/v1/register_group`; the former
`/api/v1/add_group` is retained as a **deprecated alias** to the same handler.
New callers should use `/register_group`.

**Re-registration is an idempotent upsert of the consent window only.**
Registering an existing `group_id` overwrites that dyad's
`consent_start_date` / `consent_end_date` from the request body and returns
`201` with `message: "Group consent window updated."`. This is the supported
way to correct a dyad's active window.

A re-registration whose `member_list` **matches** the first registration (or
repeats it verbatim) is accepted; the first-registration members stand. A
re-registration whose `member_list` **differs** from the recorded members is
**rejected** with `409 Member Mismatch` (see the error table) rather than
silently ignored — changing a dyad's membership after enrollment is not
supported through this endpoint, and a silent no-op would let the host believe
a correction took effect when it did not.

**Errors** (guiding principle — never lose a recruitable dyad; registration is
off the real-time path, so a transient outage never affects an in-flight
intervention):

| `status` | `title` | Cause | Host action |
|---|---|---|---|
| `400` | `Invalid Parameter` | required field missing or malformed | no partial row written; correct and re-submit |
| `409` | `Member Mismatch` | re-registration with a `member_list` differing from the recorded members | membership is fixed at first registration; do not attempt to change it here |
| `503` | `Service Unavailable` | database write failure | safe to retry with backoff (writes exactly one row, no other side effects) |

Error examples. `rid` is always present; parsed fields are echoed; only
a malformed field is `null`.

```json
// 400 — consent_start_date malformed (the other fields parsed, so they echo)
{
  "status": 400,
  "title": "Invalid Parameter",
  "message": "consent_start_date must be a Monday in YYYY-MM-DD format; got \"2026-05-26\" (Tuesday).",
  "server_timestamp": "2026-05-25T14:03:12",
  "group_id": "dyad_007",
  "member_list": ["cp_007", "aya_007"],
  "consent_start_date": null,
  "consent_end_date": "2026-09-02",
  "rid": "reg_000042"
}
```

```json
// 409 — re-registration changed the member_list (recorded members stand; the submitted list is echoed for diagnosis)
{
  "status": 409,
  "title": "Member Mismatch",
  "message": "group_id dyad_007 is already registered with members [cp_007, aya_007]; membership cannot be changed via /register_group.",
  "server_timestamp": "2026-05-25T14:03:12",
  "group_id": "dyad_007",
  "member_list": ["cp_007", "aya_999"],
  "consent_start_date": "2026-05-25",
  "consent_end_date": "2026-09-02",
  "rid": "reg_000042"
}
```

```json
// 503 — database write failure (all fields parsed; nothing is null)
{
  "status": 503,
  "title": "Service Unavailable",
  "message": "Database write failed; retry with backoff.",
  "server_timestamp": "2026-05-25T14:03:12",
  "group_id": "dyad_007",
  "member_list": ["cp_007", "aya_007"],
  "consent_start_date": "2026-05-25",
  "consent_end_date": "2026-09-02",
  "rid": "reg_000042"
}
```

### 2.2 `POST /api/v1/action` — request an action

Called by the host shortly before each decision window for a dyad (see §3).

**Context is not sent on `/action`.** The host calls
`/upload_data` before each `/action` (and may call it any number of times in
between). At action time the API reads the dyad's most recent uploaded values
from the `data_uploads` log and projects out the subset of fields the
requested `decision_type` needs. The host does not tell the API which
fields are "context" — the learner decides per decision type.

Request:

| Field | Type | Notes |
|---|---|---|
| `group_id` | string | must be a registered dyad |
| `timestamp` | ISO-8601 | time of the decision |
| `decision_idx` | int | host's per-dyad decision index; used for idempotency/validation, not by the learner |
| `decision_type` | string | `aya_message` / `cp_message` / `dyad_game` |

Request body (example):

```json
{
  "group_id": "dyad_007",
  "timestamp": "2026-06-10T07:30:00",
  "decision_idx": 29,
  "decision_type": "aya_message"
}
```

Response — common envelope plus:

| Field | Type | Notes |
|---|---|---|
| `group_id` | string | echo of the requested dyad |
| `decision_type` | string | echo |
| `decision_idx` | int | echo |
| `action` | int | chosen action: `0` = do not send / game off, `1` = send / game on. `null` on failure (no decision made). |
| `action_prob` | float | **Pr(action = 1)** — the probability the learner assigned to `action = 1`, regardless of which action was chosen (no conversion needed). Always `0.5` during warm-up; `null` on failure. |
| `warmup` | bool | `true` if a pure `Bernoulli(0.5)` draw (learner bypassed); surfaced for logging only — the host need not act on it. `null` on failure. |
| `state` | list[float] | the state/feature vector the learner scored, recorded alongside the decision (echoes `actions.state`, §5.2). `null` on warm-up and on failure. |
| `model_param` | list[float] | the flat model-parameter vector the learner scored to produce `action_prob`. **Opaque to the host** — the host records/echoes it for later reproducibility and analysis but does not interpret its contents or need to know which learner is active. `null` on warm-up (no learner was used) and on failure. |

`action_prob` is always `Pr(action = 1)`, even when the chosen `action` is `0`;
the host uses it as-is and never reconstructs it. `state` and `model_param` are
opaque logging fields — they carry enough for the RL/analysis side to replay a
decision offline, but the host treats them as blobs to store and hand back.

```json
{
  "status": 201,
  "title": "Success",
  "message": "Action requested successfully.",
  "server_timestamp": "2026-06-10T07:30:01",
  "group_id": "dyad_007",
  "decision_type": "aya_message",
  "decision_idx": 29,
  "action": 1,
  "action_prob": 0.65,
  "rid": "a1b2c3d4",
  "warmup": false,
  "state": [1.0, 0.0, 0.6, 1.0, 0.4],
  "model_param": [0.12, 0.34, -0.05, 0.21, 0.05, 0.0, 0.0, 0.05, 1.0]
}
```

(`state` and `model_param` are abbreviated here; their real lengths depend on
the active learner and are not part of the host contract.)

**Idempotency key:** `(group_id, decision_type, decision_idx)`. Each agent has
its own per-dyad counter, so the same `decision_idx` may appear once per
`decision_type` for a dyad.

**A repeated triple replays the original decision — it is not an error.** If
the host re-sends an `/action` for a triple that already has a committed
decision (e.g. the first response was lost to a timeout and the host retried),
the API returns `200` with `title: "Duplicate Decision"` and the **originally
minted** `action`, `action_prob`, `rid`, `warmup`, `state`, and `model_param`
— never a freshly drawn action. This makes `/action` safe to retry: a lost
response can never cause the server and the host to disagree about which action
was taken, and no second action is minted. The host should treat this `200`
exactly like a `201` success and use the returned action.

```json
// 200 — replay of an already-committed decision (original outputs echoed verbatim)
{
  "status": 200,
  "title": "Duplicate Decision",
  "message": "Returning the existing action for (dyad_007, aya_message, 29).",
  "server_timestamp": "2026-06-10T07:30:05",
  "group_id": "dyad_007", "decision_type": "aya_message", "decision_idx": 29,
  "action": 1,
  "action_prob": 0.65,
  "rid": "a1b2c3d4",
  "warmup": false,
  "state": [1.0, 0.0, 0.6, 1.0, 0.4],
  "model_param": [0.12, 0.34, -0.05, 0.21, 0.05, 0.0, 0.0, 0.05, 1.0]
}
```

**Errors.** `/action` is the only hard-real-time endpoint, so a decision must
never be lost. On **any** failure the host applies the same **local fallback**:
draw `Bernoulli(0.5)` itself and record `action_prob = 0.5` (it does not read the
`null` action fields).

(A repeat of the idempotency triple is **not** listed here — it is a `200`
replay of the original decision, described above, not a failure.)

| `status` | `title` | Cause | Beyond the local fallback |
|---|---|---|---|
| `400` | `Invalid Parameter` | missing / type-invalid field (named in `message`) | correct and re-send |
| `404` | `Unknown Group` | `group_id` not registered | fix the registration before the next decision |
| `409` | `No State Available` | no `/upload_data` yet for this dyad | send the first snapshot before the next `/action` |
| `500` | `Internal Error` | learner or other internal failure (incl. sample buffer unavailable) | infrastructure alert raised |
| `503` | `Model Not Initialized` | the learner/model is not yet loaded or deployed (ops/deployment issue, not a host input error) | raise a deployment alert; not host-fixable |

Error examples. `rid` is always present; valid request identifiers are echoed;
only the un-producible decision outputs (`action`, `action_prob`, `warmup`) and
any field that is itself the cause of failure are `null`.

```json
// 400 — malformed field (decision_type is the offender, so it is null; the rest echo)
{
  "status": 400,
  "title": "Invalid Parameter",
  "message": "decision_type must be one of aya_message / cp_message / dyad_game; got \"aya_msg\".",
  "server_timestamp": "2026-06-10T07:30:01",
  "group_id": "dyad_007", "decision_type": null, "decision_idx": 29,
  "rid": "a1b2c3d4",
  "action": null, "action_prob": null, "warmup": null,
  "state": null, "model_param": null
}
```

```json
// 404 — group_id not a registered dyad (so it is null; the rest of the request echoes)
{
  "status": 404,
  "title": "Unknown Group",
  "message": "group_id dyad_999 is not registered.",
  "server_timestamp": "2026-06-10T07:30:01",
  "group_id": null, "decision_type": "aya_message", "decision_idx": 29,
  "rid": "a1b2c3d4",
  "action": null, "action_prob": null, "warmup": null,
  "state": null, "model_param": null
}
```

```json
// 409 — no upload history yet (request valid; only the decision can't be produced)
{
  "status": 409,
  "title": "No State Available",
  "message": "No /upload_data has been received for dyad_007; send a snapshot first.",
  "server_timestamp": "2026-06-10T07:30:01",
  "group_id": "dyad_007", "decision_type": "aya_message", "decision_idx": 29,
  "rid": "a1b2c3d4",
  "action": null, "action_prob": null, "warmup": null,
  "state": null, "model_param": null
}
```

```json
// 500 — internal / learner failure (request fully valid; failure is server-side)
{
  "status": 500,
  "title": "Internal Error",
  "message": "Learner raised an exception while sampling the action.",
  "server_timestamp": "2026-06-10T07:30:01",
  "group_id": "dyad_007", "decision_type": "aya_message", "decision_idx": 29,
  "rid": "a1b2c3d4",
  "action": null, "action_prob": null, "warmup": null,
  "state": null, "model_param": null
}
```

```json
// 503 — model not yet initialized/deployed (request valid; ops-side failure, not host-fixable)
{
  "status": 503,
  "title": "Model Not Initialized",
  "message": "The learner for aya_message is not loaded; the deployment is not ready to serve decisions.",
  "server_timestamp": "2026-06-10T07:30:01",
  "group_id": "dyad_007", "decision_type": "aya_message", "decision_idx": 29,
  "rid": "a1b2c3d4",
  "action": null, "action_prob": null, "warmup": null,
  "state": null, "model_param": null
}
```

### 2.3 `POST /api/v1/upload_data` — provide a full snapshot of dyad data

The host posts a **full snapshot** of every variable listed
in §4.1. There is **no** context/outcome distinction — the host does not need
to know which variables the learner uses as context and which as outcome.
There is also **no** `decision_type` or `decision_idx` — each upload is a
flat "current state of the dyad" snapshot, not tied to a particular decision.

<a id="upload-schedule"></a>**Upload schedule.** The host calls `/upload_data`
**exactly once immediately before every `/action`**, in decision order. Each
day's uploads (each paired with the `/action` it precedes):

| Day | Uploads, in order | Count |
|---|---|---|
| Monday | pre-`dyad_game` → pre-AYA-AM → pre-CP → pre-AYA-PM | 4 |
| Tue–Sat | pre-AYA-AM → pre-CP → pre-AYA-PM | 3 |
| Sunday | none | 0 |

Within a single morning most fields carry the same value (measurements don't
change in minutes); the exceptions are `current_game_on` and
`prior_game_action`, which depend on whether the Monday `dyad_game` /action has
run yet (§4.1). §4.1 therefore states an **AM upload value** (any morning
upload, with Monday exceptions noted) and a **PM upload value** (the pre-AYA-PM
upload) per field.

Request:

| Field | Type | Notes |
|---|---|---|
| `group_id` | string | must be a registered dyad |
| `timestamp` | ISO-8601 | wall-clock time of this snapshot |
| `data` | object | flat dict; every key listed in §4.1 must be present. Use the literal `"miss"` (or JSON `null`) to mark an unobservable value. |

Request body (example — the pre-AYA-AM snapshot on Wednesday of study week 3):

```json
{
  "group_id": "dyad_007",
  "timestamp": "2026-06-10T07:25:00",
  "data": {
    "day_in_study": 17,
    "week_in_study": 3,
    "slot": "am",
    "aya_diary_mood": 0.6,
    "aya_diary_physical": 0.4,
    "aya_app_engagement": 3,
    "aya_app_burden": 1.5,
    "aya_missing_rate_7d": 0.14,
    "previous_med_adherence": 1,
    "prompted_by_message": true,
    "cp_diary_mood": 0.5,
    "cp_app_engagement": 2,
    "cp_app_burden": 1.2,
    "cp_missing_rate_7d": 0.1,
    "daily_diary_completed": true,
    "daily_diary_score": 0.5,
    "relationship_quality_aya": 0.8,
    "relationship_quality_cp": 0.7,
    "current_game_on": 1,
    "prior_game_action": 0,
    "aya_diary_summary": 0.55,
    "cp_diary_summary": 0.5,
    "weekly_survey_completed": true,
    "weekly_relationship_score": 0.75
  }
}
```

Response — the common envelope only (no endpoint-specific fields; the
always-present `rid` is persisted on the `data_uploads` row, §5.3):

```json
{
  "status": 201,
  "title": "Success",
  "message": "Snapshot accepted.",
  "server_timestamp": "2026-06-10T07:25:01",
  "rid": "u7f3c9a1"
}
```

**Semantics.** Every upload is all-or-nothing: a field with no measurement is
sent as `"miss"` (or JSON `null`), never omitted, and the learner masks
`"miss"` via the shared missing-indicator mechanism. The host never tags an
upload's purpose — the learner does all matching server-side, at `/action` time
(latest-snapshot lookup) and `/update` time (timeline-based reward derivation).

**Errors** (guiding principle — uploads are the sole inputs to reward
derivation, so never silently drop; on any error, correct and re-send):

| `status` | `title` | Cause | Host action |
|---|---|---|---|
| `400` | `Invalid Parameter` | missing / unknown / type-invalid field (see §4.1 for the accepted set) | send the full snapshot with `"miss"` where needed; correct and re-send |
| `404` | `Unknown Group` | unknown `group_id` | register the dyad, then re-send |
| `503` | `Service Unavailable` | database write failure | safe to retry (append-only; a duplicate row is harmless) |

Error examples (envelope only — `/upload_data` has no endpoint-specific fields):

```json
// 400 — a field failed validation
{
  "status": 400,
  "title": "Invalid Parameter",
  "message": "Field aya_diary_mood must be a number or \"miss\"; got \"n/a\".",
  "server_timestamp": "2026-06-10T07:25:01",
  "rid": "u7f3c9a1"
}
```

```json
// 404 — unknown group_id
{
  "status": 404,
  "title": "Unknown Group",
  "message": "group_id dyad_999 is not registered.",
  "server_timestamp": "2026-06-10T07:25:01",
  "rid": "u7f3c9a1"
}
```

```json
// 503 — database write failure
{
  "status": 503,
  "title": "Service Unavailable",
  "message": "Database write failed; retry.",
  "server_timestamp": "2026-06-10T07:25:01",
  "rid": "u7f3c9a1"
}
```

> **[planned — `IMPLEMENTATION_TODO.md`]** rejected `400` payloads will be
> preserved verbatim for post-trial analysis.

### 2.4 `POST /api/v1/update` — re-fit the model

Asynchronous. The **monitoring algorithm** (a separate component, see
`Monitoring_Algorithm/`) is responsible for triggering this on its own
schedule (production target: weekly, Monday 3 AM, §3) and for watching for
completion. The API fits in a background thread.

**No callback.** The monitoring algorithm observes completion by reading the
`model_update_requests` table directly. The API does not POST anywhere on
completion.

Request:

| Field | Type | Notes |
|---|---|---|
| `timestamp` | ISO-8601 | time of the update request |

Request body (example):

```json
{
  "timestamp": "2026-06-01T03:00:00"
}
```

Immediate response — the common envelope only (`202`, accepted; the fit then
runs in a background thread). No endpoint-specific fields: the always-present
`rid` (§2 preamble) **is** the update handle — it keys the
`model_update_requests` row the monitoring algorithm polls (§5.6) and is
retained for post-study analysis.

```json
{
  "status": 202,
  "title": "Update Accepted",
  "message": "Model update started.",
  "server_timestamp": "2026-06-01T03:00:00",
  "rid": "e999a61c-fb5c-4f01-9942-cb7dbe501013"
}
```

This `202` only acknowledges that the fit *started*. A fit that fails *after*
acceptance is reported via `model_update_requests.status = "failed"` (§5.6, §7.1),
**not** in this response.

**Errors.**

| `status` | `title` | Cause | Host action |
|---|---|---|---|
| `503` | `Service Unavailable` | could not enqueue the update | retry |

Error example. `rid` is still present (the API assigns it on receipt), but on a
`503` the enqueue failed, so **no `model_update_requests` row was created** — this
`rid` keys no row and the monitoring algorithm should simply retry rather than
poll it:

```json
// 503 — could not start the update
{
  "status": 503,
  "title": "Service Unavailable",
  "message": "Could not start the update; retry.",
  "server_timestamp": "2026-06-01T03:00:00",
  "rid": "e999a61c-fb5c-4f01-9942-cb7dbe501013"
}
```

Behavior once accepted:
- Optionally backs up all tables to a timestamped zip before fitting (`BACKUP_DATABASE`).
- Writes a pre-update reproducibility snapshot (copy of `data_uploads`, `actions`, `groups`).
- For every `actions` row not yet paired, walks forward on the `data_uploads`
  timeline to derive that decision's outcome and writes a
  `study_data` row.
- Re-fits the learner over all `study_data` and writes new `ModelParameters`.
- On completion, sets `model_update_requests.status` to `completed` (and
  stamps `completed_at`) or `failed` (and stamps `error_message`).



---

## 3. Endpoint calling flow (worked example)

This section walks through a complete week of API calls for one active
dyad, using concrete clock times. It expands on
`ADAPTS-HCT-Interaction-Flow.md` (which only sketches the once-per-week
events) by laying out the full Mon–Sat pattern that the host runs against
every active dyad. All times are in the dyad's local study timezone.

**Assumed per-dyad clock (illustrative — each dyad picks its own MTW
windows during onboarding):**

| Window | Time | Notes |
|---|---|---|
| Morning MTW | 08:00–10:00 | AYA's chosen 2-hour AM medication-taking window |
| CP morning window | 08:00–10:00 | aligned with the AYA's AM MTW |
| Evening MTW | 20:00–22:00 | AYA's chosen 2-hour PM medication-taking window |

The host scheduler runs an `/upload_data` + `/action` pair shortly **before**
each window opens (so the API's response reaches the user via push
notification before they start medication), and collects the post-window
reports (adherence, diary) for the next upload. There is **no cohort-wide
decision clock**: each AYA and CP independently selects a 2-hour AM and PM
window, and `/action` fires relative to *their* windows. (The bundled
simulator approximates this with a single Sunday clock; production issues
per-window calls.)

**One-time, at recruitment.** Host POSTs `/register_group` once per dyad (any
time before the first decision; dyads enroll ≈1/week, non-sequential, with
overlapping active windows).

**Weekly cron — Monday 03:00.** Monitoring algorithm POSTs `/update`. The
fit runs in a background thread; the monitoring algorithm watches
`model_update_requests` for completion (§2.4). Must finish before 06:00,
when the first `/action` of the week is requested.

**Weekly cron — Monday 06:00** (`dyad_game` decision; once per week).

1. **`POST /upload_data`** (pre-`dyad_game`):
   - `current_game_on = "miss"` (this week's game action not yet decided)
   - `prior_game_action` = action returned by week $w - 1$'s `dyad_game`
     /action (or `"miss"` in week 1)
   - bookkeeping (`day_in_study`, `week_in_study`), all dyad-level
     weekly aggregates (`*_diary_summary`, `relationship_quality_*`,
     `weekly_survey_completed`, `weekly_relationship_score`), all
     AYA/CP-side fields per §4.1
2. **`POST /action`** with `decision_type = "dyad_game"`, `decision_idx =
   w`. API returns $a^{(g)}_w \in \{0, 1\}$. Host writes $a^{(g)}_w$ into
   its delivery log; this becomes `current_game_on` for every subsequent
   upload this week.

**Daily — Mon–Sat 07:30** (AYA-AM and CP decisions).

1. **`POST /upload_data`** (pre-AYA-AM):
   - `slot = "am"`
   - `current_game_on` = $a^{(g)}_w$ (binary; Mon onwards) or `"miss"`
     pre-week-1
   - `previous_med_adherence` = AYA's response to the **previous evening's**
     MTW (yesterday 20:00–22:00). `"miss"` if no response.
   - `prompted_by_message` = `true` iff the previous evening's AYA-PM
     message was delivered to the AYA's phone
   - `aya_diary_mood` / `aya_diary_physical` = yesterday's
     evening-MTW diary responses (`"miss"` if not submitted)
   - `cp_diary_mood` = yesterday's CP diary response
   - `aya_app_burden` = $\gamma_A \, b_{k-1} + a_{k-1}$ where $k - 1$
     indexes yesterday's AYA-PM decision
   - `cp_app_burden` = $\gamma_C \, b_{m-1} + a_{m-1}$ where $m - 1$
     indexes yesterday's CP decision
   - all remaining fields per §4.1
2. **`POST /action`** with `decision_type = "aya_message"`, `decision_idx`
   = host's per-dyad AYA-decision counter. API returns $a_k \in \{0, 1\}$.
   If $a_k = 1$, host queues the AYA-AM supportive message for delivery
   before the morning MTW opens.

**Daily — Mon–Sat 07:35** (CP decision, immediately after AYA-AM).

1. **`POST /upload_data`** (pre-CP): same full snapshot as the pre-AYA-AM
   upload, but issued as a separate call. (Field values are typically
   unchanged from the pre-AYA-AM upload — no new measurements arrive in 5
   minutes — but the host re-sends the full snapshot to satisfy the
   one-upload-per-`/action` rule.)
2. **`POST /action`** with `decision_type = "cp_message"`, `decision_idx`
   = host's per-dyad CP-decision counter. API returns $a_m \in \{0, 1\}$.
   If $a_m = 1$, host queues the CP message for delivery.

**During the morning MTW (08:00–10:00).** AYA takes (or does not take)
medication and responds to the post-MTW adherence question. The host
records the response (it becomes `previous_med_adherence` at the next
evening's upload). The host also stamps the actual delivery outcome of
the AYA-AM and CP messages (success / fail) for later use in
`prompted_by_message`.

**Daily — Mon–Sat 19:30** (AYA-PM decision).

1. **`POST /upload_data`** (pre-AYA-PM):
   - `slot = "pm"`
   - `previous_med_adherence` = AYA's response to **this morning's** MTW
   - `prompted_by_message` = `true` iff this morning's AYA-AM message was
     delivered
   - `aya_diary_mood` / `aya_diary_physical` = **still yesterday's**
     evening-MTW diary (today's evening-MTW diary has not happened yet)
   - `cp_diary_mood` = **still yesterday's** CP diary (CP diary is daily
     and hasn't been collected for today yet)
   - `aya_app_burden` = $\gamma_A \, b_k + a_k$ where $k$ indexes this
     morning's AYA-AM decision (the AM action has been delivered; the
     burden has advanced)
   - `cp_app_burden` = $\gamma_C \, b_m + a_m$ where $m$ indexes this
     morning's CP decision
   - `current_game_on` = $a^{(g)}_w$ (unchanged from this morning)
   - all remaining fields per §4.1
2. **`POST /action`** with `decision_type = "aya_message"`,
   `decision_idx` = host's AYA counter. API returns $a_{k+1}$. Host
   queues the AYA-PM message if $a_{k+1} = 1$.

**During the evening MTW (20:00–22:00).** AYA takes medication and
responds to the adherence question; AYA submits the daily diary (mood +
physical); CP submits the daily diary (mood). The host records all three
for use in tomorrow's AM upload.

**Sunday — no calls.** No `/upload_data` or `/action` calls all day. The
host uses Sunday to finalize Saturday-evening reports (adherence + AYA
diary + CP diary), administer the weekly relationship-quality survey
(typically Sunday evening), and pre-stage data for Monday 06:00.

**Cold-start (week 1, Monday).** The first call sequence of the dyad's
study window. At every field that summarizes prior history
(`previous_med_adherence`, `aya_diary_mood`, `aya_diary_physical`,
`cp_diary_mood`, `prior_game_action`, `*_diary_summary`,
`relationship_quality_*`), the host sends `"miss"`. `aya_app_burden`,
`cp_app_burden`, `*_missing_rate_7d`, `weekly_relationship_score` are
sent as `0.0`. `current_game_on` is `"miss"` until the Monday 06:00
`dyad_game` /action returns. `prompted_by_message` is `false`.

---

## 4. Data field dictionary

`/upload_data` carries a **full snapshot** of every variable listed in §4.1.
The host does not need to know which variables the learner uses as context
vs. outcome — it simply sends the latest measured value of each at every
upload. The learner consults a fixed subset of these fields at `/action` and
`/update` time; which subset is internal to the API.

The per-day upload schedule and the **AM upload value** / **PM upload value**
convention used below are defined once in [§2.3](#upload-schedule).

For variables measured on a weekly cadence (relationship-quality survey),
the host caches the most recent value and re-sends it on every upload until
the next administration. For the dose-trace burdens (`aya_app_burden`,
`cp_app_burden`), the host advances the recurrence at the action delivery
that follows each /action.

Field-type encodings:
- `binary` ∈ {0, 1}
- `binary_or_miss` ∈ {0, 1, "miss"}
- `bool` ∈ {true, false}
- `unit_interval` ∈ [0, 1]
- `nonneg_float` ≥ 0
- `float_or_miss` — real number or `"miss"`
- `engagement` ∈ {1, 2, 3, 4} — ordinal app-engagement bucket
- `engagement_or_miss` ∈ {1, 2, 3, 4, "miss"}
- `unit_interval_or_miss` — [0, 1] or `"miss"`
- `slot` ∈ {"am", "pm"}
- `positive_int` ≥ 1

Any field whose value is unavailable for this upload must be sent as the
literal `"miss"` (or JSON `null`); the learner masks `"miss"` values via the
shared missing-indicator mechanism.

### 4.1 Field dictionary

Every variable below must be present in every `/upload_data` (full
snapshot). For each variable, "AM value" and "PM value" specify what the
host should send at the AM upload and the PM upload respectively. For
summary statistics, the exact computation and the missingness policy are
stated.

#### Bookkeeping

**`day_in_study`** — `positive_int`. 1-indexed day since the dyad's
`consent_start_date` (day 1 = the consent-start Monday).
- *AM/PM value:* `day = (current_date - consent_start_date).days + 1`.
  Identical at AM and PM on the same calendar day.
- *Missingness:* never missing (always computable).

**`week_in_study`** — `positive_int`. 1-indexed week since
`consent_start_date`.
- *AM/PM value:* `week = floor((day_in_study - 1) / 7) + 1`.
- *Alignment:* `consent_start_date` is a Monday (§2.1), so study weeks run
  Monday–Sunday and `week_in_study` increments on Mondays, in step with the
  weekly `dyad_game` cadence (§3).
- *Missingness:* never missing.

**`slot`** — `slot`.
- *AM value:* `"am"`. *PM value:* `"pm"`.
- *Missingness:* never missing.

#### AYA-side

**`aya_diary_mood`** — `float_or_miss`. AYA's response to the mood question
on the daily AYA diary.
- *Measurement schedule:* the AYA is prompted to complete the diary (mood
  + physical-symptoms questions) **once per day, at the evening MTW
  only** — there is no morning diary prompt.
- *AM upload value:* the mood response submitted at the **previous
  evening's** MTW (i.e. yesterday's diary).
- *PM upload value:* identical to that morning's AM value — the evening
  MTW that produces today's diary has not yet occurred at PM-upload time,
  so the latest available response is still yesterday's.
- *Missingness:* `"miss"` if the AYA did not submit yesterday's diary
  (skipped the prompt, did not open the app, or submitted only physical).

**`aya_diary_physical`** — `float_or_miss`. AYA's response to the
physical-symptoms question on the daily AYA diary. Same measurement
schedule (once per day, evening MTW only) and upload-timing rules as
`aya_diary_mood` — both AM and PM uploads carry yesterday's response.
- *Missingness:* `"miss"` if not submitted yesterday.

**`aya_app_engagement`** — `engagement_or_miss`. Ordinal app-engagement
level for the AYA, computed by the host from its own app-open and
diary-completion analytics.
- *Measurement schedule:* recomputed at **every upload** from app-open and
  diary-completion analytics for calendar days $d-3$, $d-2$, $d-1$ relative
  to the upload's decision date $d$. There is no weekly caching.
- *Bucket rules:*
  - `1` — no app open on any of $d-1$, $d-2$, $d-3$.
  - `2` — no open on $d-1$, but at least one open on $d-2$ or $d-3$.
  - `3` — at least one open on $d-1$, but the daily diary for $d-1$ was
    **not** completed.
  - `4` — at least one open on $d-1$ and the daily diary for $d-1$ was
    completed.
- *AM/PM value:* both uploads on day $d$ use the same window
  $\{d-3, d-2, d-1\}$, so AM and PM values agree unless the $d-1$ diary
  record changes between uploads.
- *Missingness:* normally never `"miss"` — diary non-completion is already
  encoded in the 3-vs-4 split, and a participant who never opens the app is
  level 1, not missing. Send `"miss"` only for an operational analytics gap
  (e.g. the first uploads of the study, before analytics are computed); the
  learner substitutes level 1.

**`aya_app_burden`** — `nonneg_float`. Discount-weighted cumulative
"dose trace" of past AYA-message actions. **Not a survey** — computed by
the host from its own action-delivery log using the same discount as the
AYA learner.
- *Recurrence:* let $k$ index AYA decisions (twice daily; $k = 1, 2,
  \ldots$), let $a_k \in \{0, 1\}$ be the action returned by `/action`
  for AYA decision $k$, and let $\gamma_A = 13/14$. Define
  $$
  b_1 = 0, \qquad b_{k+1} = \gamma_A \, b_k + a_k.
  $$
  Then `aya_app_burden` at the time of decision $k$ equals $b_k$ (i.e. it
  incorporates every AYA action through decision $k - 1$, exponentially
  decayed by $\gamma_A$ per AYA decision step).
- *AM upload value:* $b_k$ for the upcoming AYA-AM decision $k$, i.e.
  $\gamma_A \, b_{k-1} + a_{k-1}$, where $k - 1$ is the previous evening's
  AYA-PM decision.
- *PM upload value:* $b_{k+1}$ for the upcoming AYA-PM decision $k + 1$,
  i.e. $\gamma_A \, b_k + a_k$, where $k$ is that morning's AYA-AM
  decision (whose action was delivered between the AM upload and the PM
  upload).
- *Missingness:* not nullable. For the very first AM upload (no prior
  AYA decision), send `0.0`.

**`aya_missing_rate_7d`** — `unit_interval`. Fraction of expected AYA
check-ins missed in the past 7 days. Expected check-ins per day are:
2 per-MTW medication-adherence responses (morning MTW + evening MTW) and
1 evening-MTW diary submission — 3 per day, 21 per 7-day window.
- *Formula:* let $E$ be the set of expected check-ins scheduled in the
  past 7 days (each MTW contributes one adherence check-in; each evening
  MTW additionally contributes one diary check-in). Let $M \subseteq E$
  be those the AYA missed (no `previous_med_adherence` response; or no
  diary submitted with at least one of `aya_diary_mood` /
  `aya_diary_physical` non-missing). Then
  $$
  \text{aya\_missing\_rate\_7d} = \frac{|M|}{|E|}.
  $$
- *AM/PM value:* recomputed at upload time over the rolling 7-day window
  ending at the upload timestamp. AM and PM values on the same day
  generally differ (the morning MTW's adherence response, if available,
  arrives between the two uploads).
- *Missingness:* if $|E| = 0$ (very early in the study, before any
  check-in has been scheduled), send `0.0`. Never `"miss"`.

**`previous_med_adherence`** — `binary_or_miss`. The AYA's
medication-adherence response for the most recently elapsed MTW.
- *Measurement schedule:* the AYA is prompted to answer "did you take your
  medication?" after each MTW; `1` = yes, `0` = no.
- *AM upload value:* response for the **previous evening's** MTW.
- *PM upload value:* response for **that morning's** MTW.
- *Missingness:* `"miss"` if the AYA did not respond within the post-MTW
  reporting window.

**`prompted_by_message`** — `bool`. Whether an AYA-supportive message was
actually delivered to the AYA during the same MTW that
`previous_med_adherence` refers to.
- *AM/PM value:* `true` iff the host's delivery log confirms that the
  `/action` for that MTW returned `action = 1` **and** the message was
  successfully delivered to the AYA's phone; `false` otherwise (no message
  sent, or send failed). Same MTW as `previous_med_adherence`.
- *Missingness:* not nullable. Send `false` if `previous_med_adherence` is
  `"miss"` or if there is no prior MTW (first AM upload of the study).

#### CP-side

**`cp_diary_mood`** — `float_or_miss`. CP's response to the mood question
on the daily CP diary. The CP diary has **only** a mood question (no
physical-symptoms question, unlike the AYA diary).
- *Measurement schedule:* the CP is prompted once daily (typically the
  evening) to complete the mood diary.
- *AM upload value:* the mood response submitted on the **previous
  calendar day**.
- *PM upload value:* identical to that morning's AM value (CP diary is
  daily, not per-MTW), so PM uploads re-send the same value.
- *Missingness:* `"miss"` if the CP did not submit yesterday's diary.

**`cp_app_engagement`** — `engagement_or_miss`. Same definition, measurement
schedule, upload-timing rules, and missingness policy as
`aya_app_engagement`, for the CP.

**`cp_app_burden`** — `nonneg_float`. Discount-weighted cumulative dose
trace of past CP-message actions. Same construction as `aya_app_burden`
with two differences: index $m$ runs over CP decisions (once daily, AM
only) and the discount is $\gamma_C = 6/7$.
- *Recurrence:* $b_1 = 0$, $b_{m+1} = \gamma_C \, b_m + a_m$ where
  $a_m \in \{0, 1\}$ is the action returned by `/action` for CP decision
  $m$.
- *AM upload value:* $b_m$ for the upcoming CP decision $m$, i.e.
  $\gamma_C \, b_{m-1} + a_{m-1}$, where $m - 1$ is the previous morning's
  CP decision.
- *PM upload value:* $b_{m+1}$ for the next CP decision (tomorrow
  morning), i.e. $\gamma_C \, b_m + a_m$, where $m$ is that morning's CP
  decision (whose action was delivered between the AM upload and the PM
  upload). Carries the same value through to the next AM upload (no CP
  decision happens between PM upload $D$ and AM upload $D + 1$).
- *Missingness:* not nullable. For the very first AM upload, send `0.0`.

**`cp_missing_rate_7d`** — `unit_interval`. Fraction of expected CP daily
diaries missed in the past 7 days.
- *Formula:* let $D$ be the set of past 7 calendar days (excluding the
  current day). Let $M \subseteq D$ be days on which the CP did not submit
  the daily diary (or `cp_diary_mood` was `"miss"`). Then
  $$
  \text{cp\_missing\_rate\_7d} = \frac{|M|}{|D|} = \frac{|M|}{7}.
  $$
- *AM/PM value:* recomputed at upload time. AM and PM on the same day are
  identical (since the diary is daily, not per-MTW).
- *Missingness:* if fewer than 7 days have elapsed in the study, use
  $|D|$ = days-elapsed-so-far in the denominator; if 0, send `0.0`. Never
  `"miss"`.

**`daily_diary_completed`** — `bool`. Whether the CP completed the most
recent CP daily diary. Used as the basis for the `cp_message` reward.
- *AM upload value:* `true` iff the CP submitted yesterday's daily diary.
- *PM upload value:* same as that morning's AM value.
- *Missingness:* not nullable; `false` indicates non-completion.

**`daily_diary_score`** — `nonneg_float`. Score of the most recent
completed CP daily diary. Since the CP diary contains only one question,
this equals `cp_diary_mood` whenever it is non-missing.
- *AM/PM value:* `cp_diary_mood` if `daily_diary_completed = true`,
  else `0.0`.
- *Missingness:* if `daily_diary_completed = false`, send `0.0` (the
  learner ignores it).

#### Dyad-level

**`relationship_quality_aya`** — `float_or_miss`. AYA's most recent
relationship-quality score with the CP.
- *Measurement schedule:* the AYA is prompted weekly (host-defined day;
  assume Sunday evening) to complete a relationship-quality survey.
  Between administrations the host caches the most recent score.
- *AM/PM value:* the cached most recent AYA relationship-quality response.
  Identical at all uploads within a week.
- *Missingness:* `"miss"` until the AYA has completed at least one
  relationship survey.

**`relationship_quality_cp`** — `float_or_miss`. Same definition,
measurement schedule, and upload-timing rules as `relationship_quality_aya`,
for the CP.

**`current_game_on`** — `binary_or_miss`. Whether the dyad game is on in
the current week — i.e. the action returned by the most recent
`dyad_game` /action.
- *Pre-`dyad_game` upload (Monday morning, before the weekly `dyad_game`
  /action):* `"miss"`. The current week's `dyad_game` action has **not
  yet been decided** at this upload, and the previous week's action is
  captured separately by `prior_game_action` — so `current_game_on` is
  not yet meaningful. The `dyad_game` learner does not read
  `current_game_on`, so this `"miss"` does not affect the
  decision.
- *Pre-AYA-AM upload (Monday morning, after the `dyad_game` /action):*
  equals the action $a^{(g)} \in \{0, 1\}$ just returned by the
  `dyad_game` /action. The `aya_message` and `cp_message` learners read
  this value as a state feature for that morning's decisions.
- *Pre-CP upload (Monday morning):* same as the pre-AYA-AM upload value.
- *Pre-AYA-PM upload (Monday evening) and every upload Tue–Sat:* equals
  Monday's $a^{(g)}$, carried forward unchanged through the week.
- *Missingness:* `"miss"` at the pre-`dyad_game` Monday upload by
  construction; never `"miss"` at any subsequent upload in the week (the
  action is always either `0` or `1`).

**`prior_game_action`** — `binary_or_miss`. The action chosen at the
**most-recently-completed** `dyad_game` decision — the feature read by
the `dyad_game` learner at this Monday's decision.
- *Pre-`dyad_game` upload (Monday morning of week $w \geq 2$):* equals
  the action returned by week $w - 1$'s `dyad_game` /action.
- *All subsequent uploads in week $w$ (pre-AYA-AM, pre-CP, pre-AYA-PM,
  through Saturday evening):* unchanged from the pre-`dyad_game` value —
  the host does **not** advance `prior_game_action` to the action just
  decided. Advancement happens only at week $w + 1$'s pre-`dyad_game`
  upload.
- *Missingness:* `"miss"` in study week 1 (no previous week's `dyad_game`
  decision exists).

**`aya_diary_summary`** — `unit_interval_or_miss`. Weekly aggregate of the AYA's
daily evening-MTW mood-diary entries.
- *Formula:* the AYA submits at most one diary per day (at the evening
  MTW), so the past 7 days yield at most 7 entries. Let $S$ be the subset
  of those daily diaries with a non-missing `aya_diary_mood` value, and
  let $\widetilde{m}_d \in [0, 1]$ be the AYA's mood response on day $d$
  normalized to $[0, 1]$ by the host-defined mood scale (e.g. for a 1–5
  Likert, $\widetilde{m}_d = (m_d - 1)/4$). Then
  $$
  \text{aya\_diary\_summary} = \frac{1}{|S|}\sum_{d \in S} \widetilde{m}_d, \qquad |S| \le 7.
  $$
- *AM/PM value:* recomputed at upload time over the rolling 7-day window.
  AM and PM values on the same day are identical (no new diary arrives
  between AM and PM uploads).
- *Missingness:* if $|S| = 0$ (no mood entries in the past 7 days), send
  `"miss"`.

**`cp_diary_summary`** — `unit_interval_or_miss`. Weekly aggregate of the CP's
daily mood-diary entries.
- *Formula:* let $S$ be the set of CP daily-diary submissions in the past
  7 days with a non-missing `cp_diary_mood`, and let $\widetilde{m}_d \in
  [0, 1]$ be the normalized mood response. Then
  $$
  \text{cp\_diary\_summary} = \frac{1}{|S|}\sum_{d \in S} \widetilde{m}_d.
  $$
- *AM/PM value:* recomputed at upload time.
- *Missingness:* if $|S| = 0$, send `"miss"`.

**`weekly_survey_completed`** — `bool`. Whether the dyad completed the
most recent weekly relationship survey.
- *AM/PM value:* `true` iff **both** `relationship_quality_aya` and
  `relationship_quality_cp` are non-missing for the most recent weekly
  administration; `false` otherwise.
- *Missingness:* not nullable.

**`weekly_relationship_score`** — `nonneg_float`. Composite weekly
relationship score, used as the basis for the `dyad_game` reward.
- *Formula:*
  $$
  \text{weekly\_relationship\_score} = \begin{cases}
  \tfrac{1}{2}\left(\text{relationship\_quality\_aya} + \text{relationship\_quality\_cp}\right) & \text{if } \texttt{weekly\_survey\_completed} = \text{true} \\
  0.0 & \text{otherwise.}
  \end{cases}
  $$
- *AM/PM value:* recomputed at upload time. Identical across uploads
  within a week unless a partial survey arrives mid-week.
- *Missingness:* `0.0` whenever `weekly_survey_completed = false` (the
  learner ignores it).

---

## 5. Persisted data model (API-internal)

Authoritative source: `app/models.py`. The host does not write these directly; all
mutations happen through the four endpoints in §2. `flask export-csv` dumps every
table to `exports/` for post-study analysis.

Every table has a synthetic `id` integer primary key (autoincrement) which is omitted
from the column listings below.

### 5.1 `groups`

One row per dyad. Written by `/register_group`.

| Column | Type | Notes |
|---|---|---|
| `group_id` | string (unique) | host-supplied dyad identifier |
| `rid` | string | per-call id returned by the most recent `/register_group` for this dyad (updated on idempotent re-registration); lets the host reconcile a registration response with the stored row |
| `group_info` | JSON | `{member_list, consent_start_date, consent_end_date}` |
| `created_at` | datetime | row creation timestamp |

### 5.2 `actions`

One row per `/action`. Records the realized decision and the random-state cursor used,
so the action can be replayed deterministically.

| Column | Type | Notes |
|---|---|---|
| `rid` | string (unique) | unique action id returned in the response |
| `group_id` | string | FK-by-value to `groups.group_id` |
| `decision_idx` | int | per-`(dyad, decision_type)` decision counter; component of the idempotency key |
| `decision_type` | string | `aya_message` / `cp_message` / `dyad_game` |
| `raw_context` | JSON | the per-agent context used at decision time — projected at action time from the dyad's latest values in `data_uploads`. Recorded explicitly so the decision is reproducible even if later uploads overwrite individual fields. |
| `state` | JSON | the state/feature vector scored at decision time; echoed to the `/action` response as `state` (§2.2) |
| `action` | int | chosen action ∈ {0, 1} |
| `action_prob` | float | Pr(action = 1), regardless of the chosen action; always `0.5` on warm-up rows |
| `is_warmup` | bool | `true` if this decision was a `Bernoulli(0.5)` warm-up draw (learner bypassed), per §2.2 |
| `warmup_reason` | string (nullable) | `cohort` / `week1` on warm-up rows; `NULL` otherwise |
| `model_param` | JSON (nullable) | the flat model-parameter vector the learner scored to produce `action_prob`, stored opaquely; echoed to the `/action` response as `model_param` and returned verbatim on the idempotent replay (§2.2). `NULL` on warm-up rows and for learners that expose no such vector. |
| `random_state` | JSON | sample-buffer cursor positions for this draw (for replay) |
| `model_parameters_id` | int (FK) | which `model_parameters` row was used |
| `request_timestamp` | datetime | timestamp the host stamped on the `/action` request |
| `timestamp` | datetime | server-side row timestamp |

Unique constraint: `(group_id, decision_type, decision_idx)`.

### 5.3 `data_uploads`

Append-only log of every `/upload_data` call. Each row is a
**full snapshot** of every variable in §4.1 (`data.X` is always present,
possibly `"miss"`). The "current value of field X for dyad Y" is simply
`data.X` from the most recent row for Y. The `/action` endpoint reads the
latest row at decision time; `/update` walks the timeline to derive
outcomes.

| Column | Type | Notes |
|---|---|---|
| `rid` | string (unique) | per-call `rid` returned by the `/upload_data` response that wrote this row |
| `group_id` | string | FK-by-value to `groups.group_id` |
| `data` | JSON | the flat `data` dict as posted; every key in §4.1 is present |
| `request_timestamp` | datetime | timestamp on the `/upload_data` request |
| `created_at` | datetime | row creation timestamp |

### 5.4 `study_data`

Now an **update-derived** table: one row per `(action, derived
outcome)` pair, written during `/update`. The matching `actions` row is
located by `(group_id, decision_type, decision_idx)`; the outcome fields are
read from later `data_uploads` rows on the timeline; the scalar reward is
computed and stored.

| Column | Type | Notes |
|---|---|---|
| `group_id` | string |  |
| `decision_idx` | int |  |
| `decision_type` | string |  |
| `action` | int | echoed from the matching `actions` row |
| `action_prob` | float | echoed from the matching `actions` row |
| `state` | JSON | feature vector at decision time (echoed from `actions.state`) |
| `raw_context` | JSON | context at decision time (echoed from `actions.raw_context`) |
| `outcome` | JSON | realized outcome fields read from later `data_uploads` rows |
| `reward` | float | scalar reward computed from `outcome` |
| `request_timestamp` | datetime | echoed from the matching `actions` row |
| `derived_at` | datetime | timestamp of the `/update` that produced this row |
| `created_at` | datetime | row creation timestamp |

Unique constraint: `(group_id, decision_type, decision_idx)`. The table is
idempotent across re-runs of `/update` — an existing row for a given
`(group_id, decision_type, decision_idx)` is updated in place if its
`outcome` window has subsequently filled in (e.g. a late upload arrived).

### 5.5 `model_parameters`

Unified store for all algorithm parameters — the fixed-probability policy
row and the Empirical-Bayes snapshots.

Each row is either:
- a **policy** row (`snapshot_type IS NULL`) — set at app init and after each
  `/update`. Holds only `probability_of_action`. `actions.model_parameters_id`
  FKs to the latest such row (the one in effect at action time).
- an **EB snapshot** row (`snapshot_type IN {"local_fit", "hyper", "posterior"}`).
  Holds the EB fields. Written by `/update` for each (snapshot_type, group_id,
  decision_type, agent_decision_index). Read by the learner at action time and
  by analysis tooling.

| Column | Type | Notes |
|---|---|---|
| `probability_of_action` | float (nullable) | set on legacy/policy rows; NULL on EB snapshot rows |
| `snapshot_type` | string (nullable) | `local_fit` / `hyper` / `posterior` for EB rows; NULL on policy rows |
| `group_id` | string (nullable) | dyad id for per-dyad EB rows; NULL for pooled `hyper` rows and for policy rows |
| `decision_type` | string (nullable) | which agent's learner the EB row belongs to |
| `agent_decision_index` | int (nullable) | bookkeeping — how many decisions had been observed when this snapshot was written |
| `sample_size` | int (nullable) | n of decisions in the EB fit |
| `feature_dim` | int (nullable) | dimensionality of `phi(s, a)` |
| `theta` | JSON (nullable) | posterior mean / point estimate (length = `feature_dim`) |
| `covariance` | JSON (nullable) | posterior covariance (`feature_dim` × `feature_dim`) |
| `perturbation` | JSON (nullable) | RLSVI perturbation draw, if applicable |
| `metadata_json` | JSON (nullable) | free-form (algorithm version, hyperparam values, sampler cursor positions) |
| `timestamp` | datetime | when this row was inserted |

### 5.6 `model_update_requests`

One row per `/update`. Tracks async fit progress. Completion is observed by
reading `status` and `completed_at` here; the API does not POST anywhere.

| Column | Type | Notes |
|---|---|---|
| `rid` | string (unique) | per-call `rid` returned in the `202` response; the key the monitoring algorithm polls |
| `status` | string | `processing` / `completed` / `failed` |
| `request_timestamp` | datetime | from the `/update` payload |
| `created_at` | datetime | row creation timestamp |
| `completed_at` | datetime (nullable) | set when fit terminates |
| `error_message` | string (nullable) | set on failure |

### 5.7 `thompson_sampling_params`

One row per `(group_id, decision_type)` when `RL_ALGORITHM = "thompson_sampling"`.
Each row holds the bandit posterior for one independent Thompson Sampling bandit.

| Column | Type | Notes |
|---|---|---|
| `group_id` | string |  |
| `decision_type` | string |  |
| `params` | JSON | `{action_0: {...}, action_1: {...}}` posterior moments per action |
| `updated_at` | datetime | last write |

Unique constraint: `(group_id, decision_type)`.

### 5.8 `standardization_baselines`

Per-dyad week-1 means and stds used to standardize continuous state variables before
they enter the learner. Written once per
(dyad, decision_type, variable) at the first `/update` after enough week-1 data is in;
never modified thereafter.

| Column | Type | Notes |
|---|---|---|
| `group_id` | string |  |
| `decision_type` | string |  |
| `variable_name` | string | name of the standardized field (e.g. `aya_app_burden`) |
| `mu` | float | week-1 mean |
| `sigma` | float | week-1 standard deviation |
| `sample_size` | int | number of week-1 observations the baseline was computed from |
| `created_at` | datetime |  |

Unique constraint: `(group_id, decision_type, variable_name)`.

### 5.9 `update_reproducibility_snapshots`

Pointers to on-disk full copies of `data_uploads`, `actions`, and `groups`
taken immediately before each `/update` completes. The actual data lives on
disk under `repro_snapshots/<rid>/`; this table is the index. Consumed
by `tools/reproduce_run.py` to replay a study. (`study_data` is itself
derived during `/update`, so the snapshot copies the upstream
`data_uploads`.)

| Column | Type | Notes |
|---|---|---|
| `rid` | string | the `/update` call's `rid`; matches `model_update_requests.rid` |
| `model_parameters_id` | int (nullable) | the `model_parameters` row produced by this update |
| `snapshot_dir` | string | absolute or repo-relative path to the on-disk snapshot |
| `data_uploads_count` | int | row count at snapshot time |
| `actions_count` | int |  |
| `groups_count` | int |  |
| `total_bytes` | int | total snapshot size on disk |
| `created_at` | datetime |  |

---

## 6. Reproducibility

The learner draws no fresh randomness at runtime. A pre-study step (`flask init-buffer`)
pre-samples a long sequence of standard-normal and uniform primitives into a `.npz` buffer
seeded by `SAMPLE_BUFFER_SEED`. At runtime each draw pulls the next primitive(s) from this
buffer; the cursor position is stamped into each `actions` row and restored on restart so a
crash does not re-consume primitives. Given the same buffer and the same ordered event log,
the service produces byte-identical actions and updates. `tools/reproduce_run.py` replays a
study from a buffer + snapshot/exports and asserts a bit-for-bit match.

---

## 7. Failure handling & fallback

- **Idempotency:** a repeated `(group_id, decision_type, decision_idx)` on
  `/action` is rejected, so host retries are safe. `/upload_data` is
  append-only and has no idempotency key — re-posting the same values writes
  a new row but is harmless (the latest-values store is unchanged).
- **At least one `/upload_data` before `/action`.** If a dyad
  has zero `data_uploads` rows when `/action` is called, the API returns
  `409`. The host is responsible for ensuring the first upload happens
  before the first action. Once the dyad has any upload history, subsequent
  `/action` calls succeed even if some fields are missing — they are masked.
- **No corrections.** The host never re-sends corrected values for a past
  upload; every upload is simply the latest snapshot. A measurement that
  arrives late on the host side appears in the next scheduled upload and is
  picked up at the next `/update`.
- **Missed `/update`.** The monitoring algorithm should re-ping if a
  scheduled update is missed; the API itself does not schedule.
- **API unreachable:** the host draws `Bernoulli(0.5)` locally for that
  decision and records `action_prob = 0.5`. The API has no record of such
  decisions, so they never enter model fits (the fit uses only API-issued
  actions paired with host uploads).
- See `ADAPTS-HCT-RL-API/Possible_System_Failure.md` for the full failure-mode catalog.

### 7.1 Asynchronous `/update` fit outcomes

Request-level (synchronous) errors for all four endpoints are in their §2 error
tables. Because `/update` returns `202` *before* fitting, its fit outcomes are
reported only through `model_update_requests` (§5.6) — never in the HTTP
response. Guiding principle: `/update` never publishes a bad policy; it keeps the
last good parameters.

| Fit outcome | Effect |
|---|---|
| sanity-gate failure | fit not published; previous parameters stay active; request marked `failed` |
| empty batch | fit skipped; previous parameters stay active; request marked `completed` |
| per-action reward-derivation error | that action skipped (`reward = NULL`); the rest derived |
| background-thread crash / timeout | request stays `processing`; monitoring re-triggers; previous parameters stay active |

---

## 8. Open items (to resolve with the dev team)

1. **Scheduling robustness:** timezone/DST handling for per-dyad decision
   windows, and the guarantee that the weekly `/update` completes before the
   first `/action` of the week.
