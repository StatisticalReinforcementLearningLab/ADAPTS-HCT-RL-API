# ADAPTS-HCT RL API — Specification

**Status:** lives in-repo at `API-Spec.md`. Reflects the implemented service as
of 2026-05-28, *with one planned change documented but not yet merged*:
`/action` is being made context-free, with the context carried on the
preceding `/upload_data` (see §3.2, §3.3, and §9 item 1). Until that lands,
the code in `app/routes/action.py` and `app/routes/data.py` still uses the
older "context on `/action`" shape.

This supersedes an earlier MiWaves-derived draft (`References/ADAPTS-HCT RL
API Spec.md` in the broader ADAPTS workspace) that used token auth,
`cur_var`/`past3_var` context, a `seed` field in the action response, Fitbit
sleep, 48 h notification-dose counts, and a daily + weekly update split. Items
that differ from that draft are flagged **[changed]**; planned-but-not-yet-
implemented items are flagged **[planned]**.

---

## 1. Overview

The RL API is a Flask REST service. The study **host** (app backend + scheduler, owned by
the Michigan team) calls it at each decision time; the API returns a randomized action,
logs the realized context and outcome, and re-fits the learner on a periodic schedule.

- **The API is the system of record for all model state.** The host relays raw context and
  outcomes; it does not compute or store features, parameters, or the policy.
- **Three decision types (agents)**, all served by one learner with cross-dyad pooling:
  `aya_message` (twice daily), `cp_message` (daily), `dyad_game` (weekly).
- **Reproducibility:** every action and update is deterministic given (i) a pre-sampled
  random-primitive buffer (`.npz`) and (ii) the ordered event log. See §7.

---

## 2. Conventions

- **Base path:** all endpoints are under `/api/v1` (e.g. `POST /api/v1/action`). **[changed]**
- **Transport:** JSON request and response bodies; `Content-Type: application/json`.
- **Timestamps:** ISO-8601 strings (`YYYY-MM-DDTHH:MM:SS`), interpreted in the study timezone.
- **`decision_type`:** one of `"aya_message"`, `"cp_message"`, `"dyad_game"`.
- **Missing values:** any context/outcome field may carry the literal token `"miss"`
  (or JSON `null`) when the host cannot supply it. The API masks missing values internally
  via a shared missing indicator; it does **not** reject the decision. **[changed]**
- **Auth:** none currently implemented. The earlier `/auth/register|login|logout` token flow
  is **not present** in the service. Access control will be handled at the deployment layer
  (network / reverse proxy) and is **[planned]** to be revisited with the dev team.
- **PHI:** the service logs request method, path, and content-length only — never request or
  response bodies.

---

## 3. Endpoints

### 3.1 `POST /api/v1/add_group` — register a dyad

Registers a dyad (a group of two participants) at recruitment. **[changed: was `/register`]**

Request:

| Field | Type | Notes |
|---|---|---|
| `group_id` | string | unique dyad identifier |
| `member_list` | list | participant identifiers, e.g. `[cp_id, aya_id]` |
| `consent_start_date` | `YYYY-MM-DD` | onboarding/consent complete |
| `consent_end_date` | `YYYY-MM-DD` | active window end (≈ start + 100 days) |
| `warmup` | bool (optional, default `false`) | if `true`, every decision for this dyad is a pure `Bernoulli(0.5)` draw (bypasses the learner). The host sets this for the first 5 enrolled dyads to seed the EB hyper-prior. **[changed: replaces the `status_list` lifecycle]** |

Request body (example):

```json
{
  "group_id": "dyad_007",
  "member_list": ["cp_007", "aya_007"],
  "consent_start_date": "2026-05-27",
  "consent_end_date": "2026-09-04",
  "warmup": false
}
```

Response `201`:

```json
{
  "status": "success",
  "message": "Group added successfully.",
  "group_id": "dyad_007",
  "warmup": false
}
```

`400` — group already exists, or a required field is missing.

There is **no** `REGISTERED → STARTED → COMPLETED` status machine in the current service;
the only persisted lifecycle flag is `warmup`. A status lifecycle is **[planned]** if the
host needs it.

### 3.2 `POST /api/v1/action` — request an action

Called by the host shortly before each decision window for a dyad (see §5).

**Context is no longer sent on `/action`.** The host must call `/upload_data` immediately before every `/action`; the API uses the context carried on that prior `/upload_data` for the same `(group_id, decision_type)`. **[changed: `/action` is now context-free; see §3.3.]**

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
  "timestamp": "2026-05-27T09:00:00",
  "decision_idx": 12,
  "decision_type": "aya_message"
}
```

Response `201`:

```json
{
  "status": "success",
  "message": "Action requested successfully.",
  "group_id": "dyad_007",
  "action": 1,
  "action_prob": 0.65,
  "rid": "a1b2c3d4",
  "timestamp": "2026-05-27T09:00:01",
  "state": [1.0, 1.0, 0.0, 0.6, 0.4, 0.7, 0.8, 3.0, 1.5, 0.14, 1.0]
}
```

- `action` ∈ {0 (do not send / game off), 1 (send / game on)}.
- `action_prob` is **Pr(chosen action)**, not Pr(action = 1). Analysis code must convert:
  `pi1 = action_prob if action == 1 else 1 - action_prob`.
- `rid` — unique id for this action; the host should retain it for reference.
- `state` — the feature vector the learner used. **Currently returned, but redundant: the API
  persists it in its own `actions` table. [planned: remove from the response; the host does
  not need it.]** The earlier draft's `seed` field is **not** returned (the RNG state is
  internal). **[changed]**

Error responses: `404` group not found / no model parameters; `400` `decision_idx` already
exists for this dyad (idempotent — a repeated `(group_id, decision_idx)` is rejected, so the
host can safely retry); `409` no prior `/upload_data` for this `(group_id, decision_type)`
(the host violated the "upload before action" contract); `500` internal error.

### 3.3 `POST /api/v1/upload_data` — provide current context + (optionally) log a prior outcome

**Dual role.** **[changed]** The host calls `/upload_data` immediately before every
`/action`. Each call:

1. **Always:** stashes `context` as the latest-known context for that
   `(group_id, decision_type)`. The next `/action` for the same pair reads it.
2. **Optional:** if `decision_idx` matches a prior `/action`, also logs the outcome of that
   prior decision — computes the scalar reward (§4.3) and writes a `study_data` row.

The first `/upload_data` for a dyad+decision_type carries only context (no outcome to log,
since no prior `/action` has happened). All subsequent calls carry both — the outcome of
decision `N-1` and the context for decision `N`.

Request body (example — `aya_message`; see §4.1 / §4.2 for `cp_message` and `dyad_game` shapes):

```json
{
  "group_id": "dyad_007",
  "decision_idx": 12,
  "decision_type": "aya_message",
  "timestamp": "2026-05-27T21:00:00",
  "data": {
    "context": {
      "slot": "am",
      "agent_decision_index": 12,
      "day_in_study": 7,
      "week_in_study": 1,
      "prior_med_adherence": 1,
      "aya_diary": { "mood": 0.6, "physical": 0.4 },
      "relationship_quality_cp": 0.7,
      "relationship_quality_aya": 0.8,
      "aya_app_engagement": 3,
      "aya_app_burden": 1.5,
      "aya_missing_rate_7d": 0.14,
      "current_game_on": 1
    },
    "outcome": {
      "med_adherence": 1,
      "prompted_by_message": true
    }
  }
}
```

**Cold-start request** (first call for a dyad+decision_type — no prior outcome to log):

```json
{
  "group_id": "dyad_007",
  "decision_idx": 1,
  "decision_type": "aya_message",
  "timestamp": "2026-05-21T08:00:00",
  "data": {
    "context": {
      "slot": "am",
      "agent_decision_index": 1,
      "day_in_study": 1,
      "week_in_study": 1,
      "prior_med_adherence": "miss",
      "aya_diary": { "mood": "miss", "physical": "miss" },
      "relationship_quality_cp": "miss",
      "relationship_quality_aya": "miss",
      "aya_app_engagement": 1,
      "aya_app_burden": 0.0,
      "aya_missing_rate_7d": 0.0,
      "current_game_on": 0
    }
  }
}
```

Outcome shapes by decision type (replace the `outcome` block):

```json
// cp_message
"outcome": { "daily_diary_completed": true, "daily_diary_score": 4.2 }

// dyad_game
"outcome": { "weekly_survey_completed": true, "weekly_relationship_score": 5.7 }
```

**Semantics:**
- `context` is **required**: it is the context the *next* `/action` will use.
- `outcome` is **optional**: present iff a prior `/action` with this `decision_idx` exists.
  If both exist, the API matches by `(group_id, decision_idx)`, computes reward, and writes a
  `study_data` row.
- The earlier-draft `data.action`, `data.action_prob`, and `data.state` fields are **no
  longer needed**: the API already persists them in the `actions` row keyed by
  `(group_id, decision_idx)` (and the host doesn't choose the action). **[changed]**
- The `outcome` is the **realized** outcome and may reflect a delivered intervention that
  differs from the logged `action` if there was a host-side delivery failure.

Responses: `201` success; `404` group not found; `404` `outcome` present but no matching
`/action` row; `400` invalid context/outcome; `500` internal error.

### 3.4 `POST /api/v1/update` — re-fit the model

Asynchronous. The host calls this **once per week, Monday 3 AM** (§5); the API fits in a
background thread and notifies the host via callback.

Request:

| Field | Type | Notes |
|---|---|---|
| `timestamp` | ISO-8601 | time of the update request |
| `callback_url` | string | URL the API POSTs to on completion |

Request body (example):

```json
{
  "timestamp": "2026-06-01T03:00:00",
  "callback_url": "https://host.example.com/adapts/rl_update_callback"
}
```

Immediate response `202`:

```json
{
  "status": "processing",
  "update_id": "e999a61c-fb5c-4f01-9942-cb7dbe501013"
}
```

Callback POST body (sent by the API to `callback_url`):

```json
// success
{
  "status": "completed",
  "update_id": "e999a61c-fb5c-4f01-9942-cb7dbe501013",
  "timestamp": "2026-06-01T03:04:21"
}

// failure
{
  "status": "failed",
  "update_id": "e999a61c-fb5c-4f01-9942-cb7dbe501013",
  "message": "ModelFit: singular hyper-covariance"
}
```

On completion the API POSTs to `callback_url`:
- success: `{ "status": "completed", "update_id", "timestamp" }`
- failure: `{ "status": "failed", "update_id", "message" }`

Behavior:
- Optionally backs up all tables to a timestamped zip before fitting (`BACKUP_DATABASE`).
- Writes a pre-update reproducibility snapshot (copy of `study_data`, decision states, groups).
- Re-fits the learner over all `study_data` and writes new `ModelParameters`.

The earlier draft's split into daily `/update_parameters` + weekly `/update_hyperparameters`
is **not** implemented; there is a single `/update`. The host should re-ping if a scheduled
update is missed. **[changed]**

### 3.5 Monitoring (auxiliary)

A monitoring blueprint is mounted at `/api/v1/monitor` (health/diagnostics per the
`Monitoring_Algorithm` package). Not required for the core decision loop.

---

## 4. Context & outcome schemas

Per `app/protocol.py`. Field-type encodings: `binary` = {0,1}; `binary_or_miss` = {0,1,"miss"};
`float_or_miss` = real or "miss"; `engagement` ∈ {1,2,3,4}; `unit_interval` ∈ [0,1];
`nonneg_float` ≥ 0; `diary_block` = `{ "mood": <num|"miss">, "physical": <num|"miss"> }`;
`slot` ∈ {"am","pm"}; `positive_int` ≥ 1.

All three context schemas include bookkeeping fields `agent_decision_index` (positive int),
`week_in_study` (positive int), and — for AYA/CP — `day_in_study` (positive int).

### 4.1 Context — by decision type

**`aya_message`:**

| Field | Type |
|---|---|
| `slot` | slot (am/pm) |
| `prior_med_adherence` | binary_or_miss |
| `aya_diary` | diary_block (mood, physical) |
| `relationship_quality_aya` | float_or_miss |
| `relationship_quality_cp` | float_or_miss |
| `aya_app_engagement` | engagement (1–4) |
| `aya_app_burden` | nonneg_float |
| `aya_missing_rate_7d` | unit_interval |
| `current_game_on` | binary |

**`cp_message`:**

| Field | Type |
|---|---|
| `cp_diary` | diary_block (mood, physical) |
| `relationship_quality_aya` | float_or_miss |
| `relationship_quality_cp` | float_or_miss |
| `cp_app_engagement` | engagement (1–4) |
| `cp_app_burden` | nonneg_float |
| `cp_missing_rate_7d` | unit_interval |
| `current_game_on` | binary_or_miss |

**`dyad_game`:**

| Field | Type |
|---|---|
| `relationship_quality_aya` | float_or_miss |
| `relationship_quality_cp` | float_or_miss |
| `aya_app_engagement` | engagement (1–4) |
| `cp_app_engagement` | engagement (1–4) |
| `aya_app_burden` | nonneg_float |
| `cp_app_burden` | nonneg_float |
| `prior_game_action` | binary_or_miss |
| `aya_diary_summary` | unit_interval |
| `cp_diary_summary` | unit_interval |

Differences from the earlier draft **[changed]**: app engagement is an ordinal 1–4 (not levels
0/1/2); there is **no** Fitbit sleep-quality variable and **no** 48 h notification-dose count
(subsumed by app-burden); affect/relationship enter as the per-role diary and
`relationship_quality_*` fields rather than the older "indicator × strength" composites.

### 4.2 Outcome — by decision type

| Decision type | Outcome fields |
|---|---|
| `aya_message` | `med_adherence` (binary_or_miss), `prompted_by_message` (bool) |
| `cp_message` | `daily_diary_completed` (bool), `daily_diary_score` (nonneg_float) |
| `dyad_game` | `weekly_survey_completed` (bool), `weekly_relationship_score` (nonneg_float) |

### 4.3 Reward (computed server-side from the outcome)

- **`aya_message`** — 4-tier ordinal:
  `0` if `med_adherence` is missing (no usable report);
  `1` if `med_adherence == 0` (reported non-adherent);
  `2` if reported adherent **and** (`prompted_by_message` or `action == 1`) — adherent after a prompt;
  `3` if reported adherent and unprompted.
- **`cp_message`** — `daily_diary_score` if `daily_diary_completed`, else `0`.
- **`dyad_game`** — `weekly_relationship_score` if `weekly_survey_completed`, else `0`.

---

## 5. Decision schedule (host-driven)

Each dyad is registered once (`/add_group`) and then active for its ≈100-day window.
**There is no cohort-wide decision clock:** each AYA and CP independently selects a 2-hour
**AM** and **PM** time window, and `/action` fires relative to *their* windows. **[changed]**

| Decision | When (dyad's own clock) | Frequency | Recipient |
|---|---|---|---|
| AYA supportive message | before the AM **and** PM window; not Sundays | twice daily | AYA |
| CP supportive message | before the AM window only; not Sundays | daily | CP |
| `dyad_game` (game on/off) | Monday morning | weekly | dyad |

Around the per-dyad calls:
- **`/upload_data` precedes every `/action`.** **[changed]** The host gathers the latest
  context (diary, engagement, burden, missing-rate, relationship_quality, bookkeeping
  indices, slot) and POSTs it via `/upload_data` *immediately before* the corresponding
  `/action`. The same call also carries the outcome of the previous decision when one
  exists. The API uses the context from this call to serve the next `/action`.
- `/update`: weekly batch re-fit, **Monday 3 AM**; must complete before that morning's
  `/action` calls or a stale model is used.
- `/add_group`: as dyads enroll (≈1/week, non-sequential; overlapping active windows).

(The bundled simulator approximates this with a single Sunday clock; production issues
per-window calls.)

---

## 6. Persisted data model (API-internal)

Authoritative source: `app/models.py`. The host does not write these directly; all
mutations happen through the four endpoints in §3. `flask export-csv` dumps every
table to `exports/` for post-study analysis.

Every table has a synthetic `id` integer primary key (autoincrement) which is omitted
from the column listings below.

### 6.1 `groups`

One row per dyad. Written by `/add_group`.

| Column | Type | Notes |
|---|---|---|
| `group_id` | string (unique) | host-supplied dyad identifier |
| `group_info` | JSON | `{member_list, consent_start_date, consent_end_date}` |
| `warmup` | bool | if `true`, every `/action` for this dyad is `Bernoulli(0.5)` regardless of the learner |
| `created_at` | datetime | row creation timestamp |

### 6.2 `actions`

One row per `/action`. Records the realized decision and the random-state cursor used,
so the action can be replayed deterministically.

| Column | Type | Notes |
|---|---|---|
| `rid` | string (unique) | unique action id returned in the response |
| `group_id` | string | FK-by-value to `groups.group_id` |
| `decision_idx` | int | host's per-dyad decision index (idempotency key with `group_id`) |
| `decision_type` | string | `aya_message` / `cp_message` / `dyad_game` |
| `raw_context` | JSON | the per-agent context used at decision time (post-refactor: copied from the most-recent `/upload_data`) |
| `state` | JSON | the feature vector `phi(s, a)` fed to the learner |
| `action` | int | chosen action ∈ {0, 1} |
| `action_prob` | float | Pr(chosen action), not Pr(action = 1) |
| `random_state` | JSON | sample-buffer cursor positions for this draw (for replay) |
| `model_parameters_id` | int (FK) | which `model_parameters` row was used |
| `request_timestamp` | datetime | timestamp the host stamped on the `/action` request |
| `timestamp` | datetime | server-side row timestamp |

### 6.3 `study_data`

One row per `/upload_data` that carries an `outcome`. The matching `actions` row is
located by `(group_id, decision_idx)` and the scalar reward is computed server-side
per §4.3. Context-only `/upload_data` calls (cold-start) do **not** write here.

| Column | Type | Notes |
|---|---|---|
| `group_id` | string |  |
| `decision_idx` | int |  |
| `decision_type` | string |  |
| `action` | int | echoed from the matching `actions` row |
| `action_prob` | float | echoed from the matching `actions` row |
| `state` | JSON | feature vector at decision time |
| `raw_context` | JSON | context at decision time (same value as `actions.raw_context`) |
| `outcome` | JSON | realized outcome per `OUTCOME_SCHEMAS[decision_type]` (§4.2) |
| `reward` | float | scalar reward computed from `outcome` per §4.3 |
| `request_timestamp` | datetime | timestamp on the `/upload_data` request |
| `created_at` | datetime | row creation timestamp |

### 6.4 `model_parameters`

Versioned policy parameters. The latest row is used at action time (via the FK from
`actions.model_parameters_id`). `/update` inserts a new row.

| Column | Type | Notes |
|---|---|---|
| `probability_of_action` | float | currently the only learner-agnostic field stored here; richer per-learner state lives in `empirical_bayes_snapshots` / `thompson_sampling_params`. **[planned: generalize this table to a polymorphic `params` JSON.]** |
| `timestamp` | datetime | when this row was inserted (i.e. when the corresponding `/update` finished) |

### 6.5 `model_update_requests`

One row per `/update`. Tracks async fit progress.

| Column | Type | Notes |
|---|---|---|
| `update_id` | string | UUID returned in the 202 response |
| `status` | string | `processing` / `completed` / `failed` |
| `callback_url` | string | URL the API POSTs to on completion |
| `request_timestamp` | datetime | from the `/update` payload |
| `created_at` | datetime | row creation timestamp |
| `completed_at` | datetime (nullable) | set when fit terminates |
| `error_message` | string (nullable) | set on failure |

### 6.6 `thompson_sampling_params`

One row per `(group_id, decision_type)` when `RL_ALGORITHM = "thompson_sampling"`.
Each row holds the bandit posterior for one independent Thompson Sampling bandit.

| Column | Type | Notes |
|---|---|---|
| `group_id` | string |  |
| `decision_type` | string |  |
| `params` | JSON | `{action_0: {...}, action_1: {...}}` posterior moments per action |
| `updated_at` | datetime | last write |

Unique constraint: `(group_id, decision_type)`.

### 6.7 `empirical_bayes_snapshots`

Persisted local fits, pooled hyperparameters, and posterior summaries for the EB
learner. Written by `/update`. Read by analysis tooling and `tools/reproduce_run.py`.

| Column | Type | Notes |
|---|---|---|
| `snapshot_type` | string | e.g. `local_fit`, `pooled_hyperparams`, `posterior` |
| `group_id` | string (nullable) | NULL for cohort-level (pooled) snapshots |
| `decision_type` | string | which agent's learner |
| `agent_decision_index` | int | bookkeeping — how many decisions had been observed |
| `sample_size` | int | n of decisions in the fit |
| `feature_dim` | int | dimensionality of `phi(s, a)` |
| `theta` | JSON | posterior mean / point estimate (length = feature_dim) |
| `covariance` | JSON | posterior covariance (feature_dim × feature_dim) |
| `perturbation` | JSON (nullable) | RLSVI perturbation draw, if applicable |
| `metadata_json` | JSON (nullable) | free-form (algorithm version, hyperparam values) |
| `created_at` | datetime |  |

### 6.8 `standardization_baselines`

Per-dyad week-1 means and stds used to standardize continuous state variables before
they enter the learner (`main.tex` §3 "Variable Standardization"). Written once per
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

### 6.9 `update_reproducibility_snapshots`

Pointers to on-disk full copies of `study_data`, `actions`, and `groups` taken
immediately before each `/update` completes. The actual data lives on disk under
`repro_snapshots/<update_id>/`; this table is the index. Consumed by
`tools/reproduce_run.py` to replay a study.

| Column | Type | Notes |
|---|---|---|
| `update_id` | string | matches `model_update_requests.update_id` |
| `model_parameters_id` | int (nullable) | the `model_parameters` row produced by this update |
| `snapshot_dir` | string | absolute or repo-relative path to the on-disk snapshot |
| `study_data_count` | int | row count at snapshot time |
| `actions_count` | int |  |
| `groups_count` | int |  |
| `total_bytes` | int | total snapshot size on disk |
| `created_at` | datetime |  |

---

## 7. Reproducibility

The learner draws no fresh randomness at runtime. A pre-study step (`flask init-buffer`)
pre-samples a long sequence of standard-normal and uniform primitives into a `.npz` buffer
seeded by `SAMPLE_BUFFER_SEED`. At runtime each draw pulls the next primitive(s) from this
buffer; the cursor position is stamped into each `actions` row and restored on restart so a
crash does not re-consume primitives. Given the same buffer and the same ordered event log,
the service produces byte-identical actions and updates. `tools/reproduce_run.py` replays a
study from a buffer + snapshot/exports and asserts a bit-for-bit match.

---

## 8. Failure handling & fallback

- **Idempotency:** a repeated `(group_id, decision_idx)` on `/action` is rejected, so host
  retries are safe.
- **`/upload_data` before `/action` is mandatory.** **[changed]** If the host calls
  `/action` for a `(group_id, decision_type)` that has no prior `/upload_data`, the API
  returns `409`. The host is responsible for ensuring the upload happens first; the API
  does not fall back to a population-default context.
- **Missed/late host calls:** the host should re-ping a missed `/update`; late-arriving or
  corrected context is an open item — see below.
- **API unreachable:** the host draws `Bernoulli(0.5)` locally for that decision and flags it
  as excluded from the next update (`excluded_from_update`), per `Study_Design/main.tex`
  fallback rows F-A1/F-A2. **[planned: confirm the host marks these and that the API can
  ingest the flag on `/upload_data`.]**
- See `ADAPTS-HCT-RL-API/Possible_System_Failure.md` for the full failure-mode catalog.

---

## 9. Open items (to resolve with the dev team)

1. **Implementation diff for the new `/action`/`/upload_data` contract.** **[changed]**
   The code in `app/routes/action.py`, `app/routes/data.py`, and the simulator still uses
   the old "context on `/action`" shape. Required changes:
   - `/action` request validator no longer requires `context`.
   - A new "latest context per (group_id, decision_type)" lookup feeds `/action`.
   - `/upload_data` accepts context-only payloads (no `action`/`action_prob`/`state`/`outcome`)
     for the cold-start call; treats `outcome` as optional thereafter.
   - The `actions` table either keeps storing `raw_context` (from the prior upload) for
     reproducibility, or references the originating `study_data`/`latest_context` row.
2. **Drop `state`** from the `/action` response (the API already persists it).
3. **Auth / access control** at the deployment layer (no app-level auth today).
4. **Delayed & corrected data:** which context streams are not final at decision time, and the
   reconciliation policy when finalized values arrive (overwrite the imputed value, keep both,
   or run a correction process).
5. **`excluded_from_update` ingestion** on `/upload_data` for host-side fallback decisions.
6. **Status lifecycle** (`REGISTERED/STARTED/COMPLETED`) only if the host needs it.
7. **Scheduling robustness:** timezone/DST handling for per-dyad windows; `/update`-before-
   `/action` ordering guarantee.
