# ADAPTS-HCT RL API — Specification

**Status:** lives in-repo at `API-Spec.md`. As of **2026-05-31** this is the
**implemented** host ↔ API contract. The redesign below is live in code
(`flask db upgrade` applies migration `20260529_01`).

The contract in one paragraph, from the host's perspective:

1. **`/upload_data` is a flat "latest values" snapshot.** No context/outcome
   distinction; the host sends the latest value of every variable in §5.1. No
   `decision_type` or `decision_idx`. See §3.3 and §5.
2. **`/action` is context-free.** The API reads the dyad's most recent
   uploaded snapshot and projects out the subset the requested `decision_type`
   needs. See §3.2.
3. **`/update` issues no callback.** The monitoring algorithm schedules updates
   and watches for completion via the `model_update_requests` table; the API
   does not POST back. See §3.4.
4. **Reward derivation is server-side, at `/update` time.** Each action is
   paired with subsequent `data_uploads` rows on the timeline; the scalar
   reward is computed there. The host never sends rewards or outcomes
   explicitly. See §5.3 and §6.3.

This supersedes an earlier MiWaves-derived draft (`References/ADAPTS-HCT RL
API Spec.md` in the broader ADAPTS workspace) that used token auth,
`cur_var`/`past3_var` context, a `seed` field in the action response, Fitbit
sleep, 48 h notification-dose counts, and a daily + weekly update split.

---

## 1. Overview

The RL API is a Flask REST service. The study **host** (app backend +
scheduler, owned by the Michigan team) calls it at each decision time; the
API returns a randomized action, logs the latest values of every variable
the host posts, and re-fits the learner on a periodic schedule triggered by
the monitoring algorithm.

- **The API is the system of record for all model state.** The host relays
  raw values (the latest reading of every field listed in §5.1) and does not
  compute or store features, parameters, the context/outcome distinction, or
  the policy. The learner picks the subset of fields it needs per decision
  type at action time, and pairs actions with their outcomes at update time.
- **Three decision types (agents)**, all served by one learner with cross-dyad pooling:
  `aya_message` (twice daily), `cp_message` (daily), `dyad_game` (weekly).
- **Reproducibility:** every action and update is deterministic given (i) a pre-sampled
  random-primitive buffer (`.npz`) and (ii) the ordered event log. See §7.

---

## 2. Conventions

- **Base path:** all endpoints are under `/api/v1` (e.g. `POST /api/v1/action`).
- **Transport:** JSON request and response bodies; `Content-Type: application/json`.
- **Timestamps:** ISO-8601 strings (`YYYY-MM-DDTHH:MM:SS`), interpreted in the study timezone.
- **`decision_type`:** one of `"aya_message"`, `"cp_message"`, `"dyad_game"`.
- **Missing values:** any `/upload_data` field may carry the literal token
  `"miss"` (or JSON `null`) when the host cannot supply it. Every
  `/upload_data` is a **full upload**: every field listed in §5.1 must be
  present (with `"miss"` used to mark explicit missingness). The learner
  masks `"miss"` values internally via a shared missing indicator; it does
  **not** reject the decision.
- **Auth:** none currently implemented. The earlier `/auth/register|login|logout` token flow
  is **not present** in the service. Access control will be handled at the deployment layer
  (network / reverse proxy) and is to be revisited with the dev team.
- **PHI:** the service logs request method, path, and content-length only — never request or
  response bodies.

---

## 3. Endpoints

### 3.1 `POST /api/v1/add_group` — register a dyad

Registers a dyad (a group of two participants) at recruitment.

Request:

| Field | Type | Notes |
|---|---|---|
| `group_id` | string | unique dyad identifier |
| `member_list` | list | participant identifiers, e.g. `[cp_id, aya_id]` |
| `consent_start_date` | `YYYY-MM-DD` | onboarding/consent complete |
| `consent_end_date` | `YYYY-MM-DD` | active window end (≈ start + 100 days) |

There is **no** host-supplied `warmup` field. Warm-up is determined entirely
by the API at decision time from the cohort size and the dyad's CP-decision
count (§3.2); the host cannot force or suppress it.

Request body (example):

```json
{
  "group_id": "dyad_007",
  "member_list": ["cp_007", "aya_007"],
  "consent_start_date": "2026-05-27",
  "consent_end_date": "2026-09-04"
}
```

Response `201`:

```json
{
  "status": "success",
  "message": "Group added successfully.",
  "group_id": "dyad_007"
}
```

`400` — group already exists, or a required field is missing.

There is **no** `REGISTERED → STARTED → COMPLETED` status machine in the current
service, and no persisted per-dyad lifecycle flags. A status lifecycle could be
added if the host needs it.

### 3.2 `POST /api/v1/action` — request an action

Called by the host shortly before each decision window for a dyad (see §4).

**Context is not sent on `/action`.** The host calls
`/upload_data` before each `/action` (and may call it any number of times in
between). At action time the API reads the dyad's most recent uploaded values
from the `data_uploads` log and projects out the subset of fields the
requested `decision_type` needs (§5.2). The host does not tell the API which
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
  "warmup": false,
  "warmup_reason": null,
  "state": [1.0, 1.0, 0.0, 0.6, 0.4, 0.7, 0.8, 3.0, 1.5, 0.14, 1.0]
}
```

- `action` ∈ {0 (do not send / game off), 1 (send / game on)}.
- `action_prob` is **Pr(chosen action)**, not Pr(action = 1). Analysis code must convert:
  `pi1 = action_prob if action == 1 else 1 - action_prob`. During warm-up `action_prob`
  is always `0.5`.
- `rid` — unique id for this action; the host should retain it for reference.
- `warmup` — `true` if this decision was a pure `Bernoulli(0.5)` draw (the learner was
  bypassed); `false` if the learner produced it. `warmup_reason` ∈ {`"cohort"`, `"week1"`,
  `null`} records *why* (§3.2 warm-up). The host does not need to act on these,
  but they are surfaced for logging/diagnostics.
- `state` — the feature vector the learner used (`null` on warm-up decisions, which bypass the
  feature builder). **Currently returned, but redundant: the API persists it in its own
  `actions` table. [planned: remove from the response; the host does not need it.]** The
  earlier draft's `seed` field is **not** returned (the RNG state is internal).

**Warm-up.** Early in the study some decisions are returned as pure
`Bernoulli(0.5)` draws ("warm-up") to seed the learner. **This is decided
entirely by the API; the host does nothing differently.** When a decision is a
warm-up draw the response carries `warmup = true` and `action_prob = 0.5`
(and `state` is `null`); otherwise `warmup = false`. The host still calls
`/upload_data` before the `/action` as usual (the `409` guard applies to
warm-up decisions too). The host cannot force or suppress warm-up, and the
selection rule is an internal API concern — not part of the host contract.

**Idempotency key:** `(group_id, decision_type, decision_idx)`. Each of the three agents
(`aya_message`, `cp_message`, `dyad_game`) has its own per-dyad counter, so the same
`decision_idx` value can legitimately appear once per `decision_type` for the same dyad.

Error responses: `404` group not found / no model parameters; `400` `decision_idx` already
exists for this `(group_id, decision_type)` (idempotent — a repeated triple is rejected, so
the host can safely retry); `409` no `/upload_data` has ever been received for this
`group_id` (the API has no values to construct a state); `500` internal error.

The API may relax the `409` to a `200` with a fully-masked
state (all variables treated as missing) once the missing-indicator behavior
has been validated end-to-end. See §9.

### 3.3 `POST /api/v1/upload_data` — provide a full snapshot of dyad data

The host posts a **full snapshot** of every variable listed
in §5.1. There is **no** context/outcome distinction — the host does not need
to know which variables the learner uses as context and which as outcome.
There is also **no** `decision_type` or `decision_idx` — each upload is a
flat "current state of the dyad" snapshot, not tied to a particular decision.

**Upload schedule.** The host calls `/upload_data` **immediately before
every `/action`** — exactly one upload per `/action` call, in the same
sequence the decisions are delivered.

- **Monday morning** (3 uploads, run sequentially):
1. pre-`dyad_game` upload → `POST /action dyad_game` returns the week's
   game action $a^{(g)}$;
2. pre-AYA-AM upload → `POST /action aya_message`;
3. pre-CP upload → `POST /action cp_message`.
   - **Tuesday–Saturday morning** (2 uploads, run sequentially):
     pre-AYA-AM upload → `POST /action aya_message`;
     pre-CP upload → `POST /action cp_message`.
   - **Every evening except Sunday** (1 upload):
     pre-AYA-PM upload → `POST /action aya_message`.

§5.1 specifies, for each variable, what value to send at each upload. For
most variables the value is identical across the uploads of a single
morning (the host's measurements don't change in minutes); the exceptions
are `current_game_on` and `prior_game_action`, whose values depend on
whether the Monday-morning `dyad_game` /action has happened yet (see §5.1).
For brevity §5.1 refers to "AM upload value" (= the value at any morning
upload, with Monday-morning exceptions noted) and "PM upload value".

Request:

| Field | Type | Notes |
|---|---|---|
| `group_id` | string | must be a registered dyad |
| `timestamp` | ISO-8601 | wall-clock time of this snapshot |
| `data` | object | flat dict; every key listed in §5.1 must be present. Use the literal `"miss"` (or JSON `null`) to mark an unobservable value. |

Request body (example — a full snapshot before an AYA AM decision):

```json
{
  "group_id": "dyad_007",
  "timestamp": "2026-05-27T08:00:00",
  "data": {
    "day_in_study": 7,
    "week_in_study": 1,
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

**Semantics:**
- Every upload is a **full snapshot** — every field in §5.1 must be present.
  A field whose underlying measurement is unavailable for this upload must
  be sent as `"miss"` (or JSON `null`); it cannot simply be omitted.
- The learner masks `"miss"` values via the shared missing-indicator
  mechanism.
- The host does not tag uploads as "this is for AYA" / "this is the outcome
  of decision 12". The learner handles all such matching server-side at
  `/action` time (latest-snapshot lookup) and `/update` time (timeline-based
  reward derivation, §5.3).
- The earlier `data.context` / `data.outcome` / `data.action` /
  `data.action_prob` / `data.state` envelope is **gone**. `data` is a flat
  dict.

Responses: `201` success; `404` group not found; `400` missing key, unknown
key, or type-invalid value (see §5.1 for the accepted set); `500` internal
error.

### 3.4 `POST /api/v1/update` — re-fit the model

Asynchronous. The **monitoring algorithm** (a separate component, see
`Monitoring_Algorithm/`) is responsible for triggering this on its own
schedule (production target: weekly, Monday 3 AM, §4) and for watching for
completion. The API fits in a background thread.

**No callback.** The earlier callback-on-completion mechanism
is gone. Completion is observed by reading the `model_update_requests` table
(or via a `GET /api/v1/update/<update_id>` endpoint — see §9). The API does
not POST anywhere on completion.

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

Immediate response `202`:

```json
{
  "status": "processing",
  "update_id": "e999a61c-fb5c-4f01-9942-cb7dbe501013"
}
```

Behavior:
- Optionally backs up all tables to a timestamped zip before fitting (`BACKUP_DATABASE`).
- Writes a pre-update reproducibility snapshot (copy of `data_uploads`, `actions`, `groups`).
- For every `actions` row not yet paired, walks forward on the `data_uploads`
  timeline to derive that decision's outcome (per §5.3) and writes a
  `study_data` row.
- Re-fits the learner over all `study_data` and writes new `ModelParameters`.
- On completion, sets `model_update_requests.status` to `completed` (and
  stamps `completed_at`) or `failed` (and stamps `error_message`).

The earlier draft's split into daily `/update_parameters` + weekly `/update_hyperparameters`
is **not** implemented; there is a single `/update`. The monitoring algorithm should re-ping
if a scheduled update is missed.

### 3.5 Monitoring (auxiliary)

A monitoring blueprint is mounted at `/api/v1/monitor` (health/diagnostics per the
`Monitoring_Algorithm` package). Not required for the core decision loop.

---

## 4. Endpoint calling flow (worked example)

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

**One-time, at recruitment.** Host POSTs `/add_group` once per dyad (any
time before the first decision; dyads enroll ≈1/week, non-sequential, with
overlapping active windows).

**Weekly cron — Monday 03:00.** Monitoring algorithm POSTs `/update`. The
fit runs in a background thread; the monitoring algorithm watches
`model_update_requests` for completion (§3.4). Must finish before 06:00,
when the first `/action` of the week is requested.

**Weekly cron — Monday 06:00** (`dyad_game` decision; once per week).

1. **`POST /upload_data`** (pre-`dyad_game`):
   - `current_game_on = "miss"` (this week's game action not yet decided)
   - `prior_game_action` = action returned by week $w - 1$'s `dyad_game`
     /action (or `"miss"` in week 1)
   - bookkeeping (`day_in_study`, `week_in_study`), all dyad-level
     weekly aggregates (`*_diary_summary`, `relationship_quality_*`,
     `weekly_survey_completed`, `weekly_relationship_score`), all
     AYA/CP-side fields per §5.1
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
   - all remaining fields per §5.1
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
   - all remaining fields per §5.1
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

Conflicts with `ADAPTS-HCT-Interaction-Flow.md`. The older
doc places the weekly `/update` and `dyad_game` calls on **Sunday**
morning rather than Monday; it also does not describe the per-day AYA-AM
/ CP / AYA-PM cadence. The schedule above (Mon 03:00 update, Mon 06:00
`dyad_game`, daily AYA-AM + CP + AYA-PM Mon–Sat) supersedes it. See §9
for an open item on reconciling the two.

---

## 5. Data field dictionary

`/upload_data` carries a **full snapshot** of every variable listed in §5.1.
The host does not need to know which variables the learner uses as context
vs. outcome — it simply sends the latest measured value of each at every
upload. The learner consults a fixed subset of these fields at `/action`
time (§5.2) and at `/update` time (§5.3).

**Upload events.** The host calls `/upload_data` **once before every
`/action` call** (see §3.3 for the per-day schedule). The number of
uploads per day is:

- **Monday:** 4 uploads — pre-`dyad_game`, pre-AYA-AM, pre-CP, pre-AYA-PM
  (in that order).
- **Tuesday–Saturday:** 3 uploads — pre-AYA-AM, pre-CP, pre-AYA-PM.
- **Sunday:** 0 uploads (no `/action` calls on Sunday).

For brevity, "AM upload value" below means the value at any morning upload
on a given day (the pre-`dyad_game`, pre-AYA-AM, and pre-CP uploads
typically carry the same value for a given field; exceptions for
`current_game_on` and `prior_game_action` are called out explicitly).
"PM upload value" means the value at the pre-AYA-PM upload.

For variables measured on a weekly cadence (relationship-quality survey)
or computed once per week (app-engagement bucket), the host caches the
most recent value and re-sends it on every upload until the next
recomputation. For the dose-trace burdens (`aya_app_burden`,
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
- `slot` ∈ {"am", "pm"}
- `positive_int` ≥ 1

Any field whose value is unavailable for this upload must be sent as the
literal `"miss"` (or JSON `null`); the learner masks `"miss"` values via the
shared missing-indicator mechanism.

### 5.1 Field dictionary

Every variable below must be present in every `/upload_data` (full
snapshot). For each variable, "AM value" and "PM value" specify what the
host should send at the AM upload and the PM upload respectively. For
summary statistics, the exact computation and the missingness policy are
stated.

#### Bookkeeping

**`day_in_study`** — `positive_int`. 1-indexed day since the dyad's
`consent_start_date`.
- *AM/PM value:* `day = (current_date - consent_start_date).days + 1`.
  Identical at AM and PM on the same calendar day.
- *Missingness:* never missing (always computable).

**`week_in_study`** — `positive_int`. 1-indexed week since
`consent_start_date`.
- *AM/PM value:* `week = floor((day_in_study - 1) / 7) + 1`.
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

**`aya_app_engagement`** — `engagement` ∈ {1, 2, 3, 4}. Most recent weekly
engagement bucket for the AYA, computed by the host.
- *Measurement schedule:* recomputed once per week (e.g. each Monday) over
  the previous 7 days of app sessions. Between recomputations the host
  caches the bucket and re-sends it on every upload.
- *AM/PM value:* the most recent computed bucket. Identical across all
  uploads within a week.
- *Missingness:* `"miss"` only in week 1 if no sessions have occurred yet;
  the host's engagement spec is authoritative for the bucket boundaries.

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

**`cp_app_engagement`** — `engagement`. Same definition, measurement
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
  learner ignores it via §5.3).

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
  `current_game_on` (see §5.2), so this `"miss"` does not affect the
  decision.
- *Pre-AYA-AM upload (Monday morning, after the `dyad_game` /action):*
  equals the action $a^{(g)} \in \{0, 1\}$ just returned by the
  `dyad_game` /action. The `aya_message` and `cp_message` learners read
  this value as a state feature for that morning's decisions (§5.2).
- *Pre-CP upload (Monday morning):* same as the pre-AYA-AM upload value.
- *Pre-AYA-PM upload (Monday evening) and every upload Tue–Sat:* equals
  Monday's $a^{(g)}$, carried forward unchanged through the week.
- *Missingness:* `"miss"` at the pre-`dyad_game` Monday upload by
  construction; never `"miss"` at any subsequent upload in the week (the
  action is always either `0` or `1`).

**`prior_game_action`** — `binary_or_miss`. The action chosen at the
**most-recently-completed** `dyad_game` decision — the feature read by
the `dyad_game` learner at this Monday's decision (§5.2).
- *Pre-`dyad_game` upload (Monday morning of week $w \geq 2$):* equals
  the action returned by week $w - 1$'s `dyad_game` /action.
- *All subsequent uploads in week $w$ (pre-AYA-AM, pre-CP, pre-AYA-PM,
  through Saturday evening):* unchanged from the pre-`dyad_game` value —
  the host does **not** advance `prior_game_action` to the action just
  decided. Advancement happens only at week $w + 1$'s pre-`dyad_game`
  upload.
- *Missingness:* `"miss"` in study week 1 (no previous week's `dyad_game`
  decision exists).

**`aya_diary_summary`** — `unit_interval`. Weekly aggregate of the AYA's
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
  `"miss"`. (See §9 for an open decision on whether to use a neutral
  default like `0.5` instead.)

**`cp_diary_summary`** — `unit_interval`. Weekly aggregate of the CP's
daily mood-diary entries.
- *Formula:* let $S$ be the set of CP daily-diary submissions in the past
  7 days with a non-missing `cp_diary_mood`, and let $\widetilde{m}_d \in
  [0, 1]$ be the normalized mood response. Then
  $$
  \text{cp\_diary\_summary} = \frac{1}{|S|}\sum_{d \in S} \widetilde{m}_d.
  $$
- *AM/PM value:* recomputed at upload time.
- *Missingness:* if $|S| = 0$, send `"miss"`. (Same open decision as
  `aya_diary_summary`.)

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
  learner ignores it via §5.3).

Differences from the earlier draft: app engagement is an
ordinal 1–4 (not levels 0/1/2); there is **no** Fitbit sleep-quality
variable and **no** 48 h notification-dose count (subsumed by app-burden);
affect / relationship enter as the per-role diary fields (`*_diary_mood`,
`aya_diary_physical`) and `relationship_quality_*` rather than the older
"indicator × strength" composites; `aya_diary` and `cp_diary` are now flat
top-level variables (`aya_diary_mood`, `aya_diary_physical`,
`cp_diary_mood`) rather than nested objects; `med_adherence` has been
renamed to `previous_med_adherence` to reflect that the value at upload time
is the adherence at the just-elapsed MTW.

### 5.2 Which fields the learner reads, by decision type

When `/action` fires with a given `decision_type`, the learner reads the
following subset of the latest snapshot. The host always sends every field
(full snapshot); if a relevant field is `"miss"` the learner masks it.

**`aya_message`** — `slot`, `previous_med_adherence`, `aya_diary_mood`,
`aya_diary_physical`, `relationship_quality_aya`,
`relationship_quality_cp`, `aya_app_engagement`, `aya_app_burden`,
`aya_missing_rate_7d`, `current_game_on`, plus bookkeeping (`day_in_study`,
`week_in_study`).

**`cp_message`** — `cp_diary_mood`, `relationship_quality_aya`,
`relationship_quality_cp`, `cp_app_engagement`, `cp_app_burden`,
`cp_missing_rate_7d`, `current_game_on`, plus bookkeeping (`day_in_study`,
`week_in_study`).

**`dyad_game`** — `relationship_quality_aya`, `relationship_quality_cp`,
`aya_app_engagement`, `cp_app_engagement`, `aya_app_burden`,
`cp_app_burden`, `prior_game_action`, `aya_diary_summary`,
`cp_diary_summary`, plus bookkeeping (`week_in_study`).

### 5.3 Reward — derived server-side at `/update` time

Each action's reward is computed at `/update` time by walking
forward from the action's timestamp on the `data_uploads` timeline and
reading the relevant outcome value from the **next** scheduled upload of
the matching kind. The outcome window boundaries are spec-only at this
point and will be finalized during the implementation diff (§9):

- **`aya_message`** — outcome window = the next AYA upload following the
  decision (AM decision → next PM upload; PM decision → next AM upload),
  so ≈12 h. The 4-tier ordinal reward is computed from that upload's
  `previous_med_adherence` (which by §5.1 is the adherence at the
  just-elapsed MTW) and `prompted_by_message`:
  - `0` if `previous_med_adherence == "miss"` (no usable report);
  - `1` if `previous_med_adherence == 0` (reported non-adherent);
  - `2` if `previous_med_adherence == 1` **and** (`prompted_by_message` is
    `true` **or** the logged `action == 1`);
  - `3` if `previous_med_adherence == 1` **and** `prompted_by_message` is
    `false` **and** the logged `action == 0`.
- **`cp_message`** — outcome window = the next CP-decision upload (next
  AM, ≈24 h). Reward = `daily_diary_score` from that upload if
  `daily_diary_completed == true`, else `0`.
- **`dyad_game`** — outcome window = the next `dyad_game` upload (next
  Monday AM, ≈7 days). Reward = `weekly_relationship_score` from that
  upload if `weekly_survey_completed == true`, else `0`.

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
| `created_at` | datetime | row creation timestamp |

### 6.2 `actions`

One row per `/action`. Records the realized decision and the random-state cursor used,
so the action can be replayed deterministically.

| Column | Type | Notes |
|---|---|---|
| `rid` | string (unique) | unique action id returned in the response |
| `group_id` | string | FK-by-value to `groups.group_id` |
| `decision_idx` | int | per-`(dyad, decision_type)` decision counter; component of the idempotency key |
| `decision_type` | string | `aya_message` / `cp_message` / `dyad_game` |
| `raw_context` | JSON | the per-agent context used at decision time — projected at action time from the dyad's latest values in `data_uploads` (§5.2). Recorded explicitly so the decision is reproducible even if later uploads overwrite individual fields. |
| `state` | JSON | the feature vector `phi(s, a)` fed to the learner |
| `action` | int | chosen action ∈ {0, 1} |
| `action_prob` | float | Pr(chosen action), not Pr(action = 1); always `0.5` on warm-up rows |
| `is_warmup` | bool | `true` if this decision was a `Bernoulli(0.5)` warm-up draw (learner bypassed), per §3.2 |
| `warmup_reason` | string (nullable) | `cohort` / `week1` on warm-up rows; `NULL` otherwise |
| `random_state` | JSON | sample-buffer cursor positions for this draw (for replay) |
| `model_parameters_id` | int (FK) | which `model_parameters` row was used |
| `request_timestamp` | datetime | timestamp the host stamped on the `/action` request |
| `timestamp` | datetime | server-side row timestamp |

Unique constraint: `(group_id, decision_type, decision_idx)`.

### 6.3 `data_uploads`

Append-only log of every `/upload_data` call. Each row is a
**full snapshot** of every variable in §5.1 (`data.X` is always present,
possibly `"miss"`). The "current value of field X for dyad Y" is simply
`data.X` from the most recent row for Y. The `/action` endpoint reads the
latest row at decision time; `/update` walks the timeline to derive
outcomes (§5.3).

| Column | Type | Notes |
|---|---|---|
| `group_id` | string | FK-by-value to `groups.group_id` |
| `data` | JSON | the flat `data` dict as posted; every key in §5.1 is present |
| `request_timestamp` | datetime | timestamp on the `/upload_data` request |
| `created_at` | datetime | row creation timestamp |

### 6.4 `study_data`

Now an **update-derived** table: one row per `(action, derived
outcome)` pair, written during `/update`. The matching `actions` row is
located by `(group_id, decision_type, decision_idx)`; the outcome fields are
read from `data_uploads` per §5.3; the scalar reward is computed and stored.

| Column | Type | Notes |
|---|---|---|
| `group_id` | string |  |
| `decision_idx` | int |  |
| `decision_type` | string |  |
| `action` | int | echoed from the matching `actions` row |
| `action_prob` | float | echoed from the matching `actions` row |
| `state` | JSON | feature vector at decision time (echoed from `actions.state`) |
| `raw_context` | JSON | context at decision time (echoed from `actions.raw_context`) |
| `outcome` | JSON | realized outcome fields read from later `data_uploads` rows per §5.3 |
| `reward` | float | scalar reward computed from `outcome` per §5.3 |
| `derived_at` | datetime | timestamp of the `/update` that produced this row |
| `created_at` | datetime | row creation timestamp |

Unique constraint: `(group_id, decision_type, decision_idx)`. The table is
idempotent across re-runs of `/update` — an existing row for a given
`(group_id, decision_type, decision_idx)` is updated in place if its
`outcome` window has subsequently filled in (e.g. a late upload arrived).

### 6.5 `model_parameters`

Unified store for all algorithm parameters — both the legacy fixed-probability
config row and the Empirical-Bayes snapshots that previously lived in a
separate `empirical_bayes_snapshots` table.

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

### 6.6 `model_update_requests`

One row per `/update`. Tracks async fit progress. The
`callback_url` column is removed in the new design — completion is observed
by reading `status` and `completed_at` here, rather than by being POSTed to.

| Column | Type | Notes |
|---|---|---|
| `update_id` | string | UUID returned in the 202 response |
| `status` | string | `processing` / `completed` / `failed` |
| `request_timestamp` | datetime | from the `/update` payload |
| `created_at` | datetime | row creation timestamp |
| `completed_at` | datetime (nullable) | set when fit terminates |
| `error_message` | string (nullable) | set on failure |

### 6.7 `thompson_sampling_params`

One row per `(group_id, decision_type)` when `RL_ALGORITHM = "thompson_sampling"`.
Each row holds the bandit posterior for one independent Thompson Sampling bandit.

| Column | Type | Notes |
|---|---|---|
| `group_id` | string |  |
| `decision_type` | string |  |
| `params` | JSON | `{action_0: {...}, action_1: {...}}` posterior moments per action |
| `updated_at` | datetime | last write |

Unique constraint: `(group_id, decision_type)`.

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

Pointers to on-disk full copies of `data_uploads`, `actions`, and `groups`
taken immediately before each `/update` completes. The actual data lives on
disk under `repro_snapshots/<update_id>/`; this table is the index. Consumed
by `tools/reproduce_run.py` to replay a study. Previously the
snapshot copied `study_data`; under the new design `study_data` is itself
derived during `/update`, so the snapshot copies the upstream
`data_uploads` instead.

| Column | Type | Notes |
|---|---|---|
| `update_id` | string | matches `model_update_requests.update_id` |
| `model_parameters_id` | int (nullable) | the `model_parameters` row produced by this update |
| `snapshot_dir` | string | absolute or repo-relative path to the on-disk snapshot |
| `data_uploads_count` | int | row count at snapshot time |
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

- **Idempotency:** a repeated `(group_id, decision_type, decision_idx)` on
  `/action` is rejected, so host retries are safe. `/upload_data` is
  append-only and has no idempotency key — re-posting the same values writes
  a new row but is harmless (the latest-values store is unchanged).
- **At least one `/upload_data` before `/action`.** If a dyad
  has zero `data_uploads` rows when `/action` is called, the API returns
  `409`. The host is responsible for ensuring the first upload happens
  before the first action. Once the dyad has any upload history, subsequent
  `/action` calls succeed even if some fields are missing — they are masked.
- **Late or corrected uploads.** Uploads that arrive *after* the action they
  would have informed are still useful: `/update` re-derives `study_data` on
  each run, so a later-arriving outcome can fill a previously-empty reward
  window. The reconciliation policy for *corrected* values (overwrite vs.
  keep both) is an open item — see below.
- **Missed `/update`.** The monitoring algorithm should re-ping if a
  scheduled update is missed; the API itself does not schedule.
- **API unreachable:** the host draws `Bernoulli(0.5)` locally for that decision and flags it
  as excluded from the next update (`excluded_from_update`), per `Study_Design/main.tex`
  fallback rows F-A1/F-A2. **[planned: confirm the host marks these and that the API can
  ingest the flag on `/upload_data`.]**
- See `ADAPTS-HCT-RL-API/Possible_System_Failure.md` for the full failure-mode catalog.

---

## 9. Open items (to resolve with the dev team)

1. **Implementation diff for the new contract.** All four
   planned changes listed at the top of this document are spec-only. Code
   changes required:
   - `app/routes/data.py` — `/upload_data` now accepts a flat `data` dict
     with no `decision_type` / `decision_idx`; drop the
     `context`/`outcome` envelope; append to a new `data_uploads` table.
   - `app/routes/action.py` — drop the `context` requirement from the
     request body; replace the per-call context with a lookup against the
     dyad's latest values in `data_uploads`; evaluate the server-side
     warm-up gates (§3.2) before invoking the learner — `COUNT(*)` on
     `groups` for `cohort`, the dyad's CP `decision_idx` for `week1` — and on
     warm-up draw `Bernoulli(0.5)` from the sample buffer, set
     `action_prob = 0.5`, and stamp `is_warmup` / `warmup_reason` on the
     `actions` row.
   - `app/models.py` — add `actions.is_warmup` and `actions.warmup_reason`
     (Alembic migration); add the `WARMUP_WEEK1_CP_DECISIONS` config constant.
   - `app/routes/update.py` — drop `callback_url` from the request body
     and the callback-on-completion machinery; add timeline-based outcome
     derivation that produces `study_data` rows.
   - `app/models.py` + Alembic migration — add `data_uploads`; reshape
     `study_data` for update-time derivation (or migrate existing rows);
     drop `model_update_requests.callback_url`.
   - `app/protocol.py` — replace the per-decision-type context/outcome
     schemas with the single flat field dictionary (§5.1).
   - `tests/simulate_adapts_hct.py` and the Bruno collection — update to
     the new request shapes.
2. **Materialization of latest-values store.** Compute on demand at `/action`
   time (one query against `data_uploads`) vs. maintain an in-memory cache
   updated by `/upload_data`. The first is simpler; the second is faster.
3. **Outcome window boundaries.** §5.3 says "until the next decision of the
   same type", but the precise rule needs nailing down (especially for the
   AYA AM↔PM boundary and for actions taken late in the study window).
4. **Late & corrected uploads.** Re-deriving `study_data` on every `/update`
   handles late uploads naturally. Corrected values (same field, same dyad,
   different number) need an explicit policy: overwrite the prior value in
   the latest-values store, or keep both with a "supersedes" link.
5. **GET `/api/v1/update/<update_id>`.** Should the monitoring algorithm
   poll the DB directly or have an HTTP endpoint? Either is fine; pick one.
6. **Drop `state` from the `/action` response** (the API already persists it).
7. **Auth / access control** at the deployment layer (no app-level auth today).
8. **`excluded_from_update` ingestion** on `/upload_data` for host-side fallback decisions.
9. **Status lifecycle** (`REGISTERED/STARTED/COMPLETED`) only if the host needs it.
10. **Scheduling robustness:** timezone/DST handling for per-dyad windows;
    `/update`-before-`/action` ordering guarantee. Note the **warm-up
    boundary no longer depends on date math** — it is gated on the cohort
    registration count and the dyad's CP-decision count (§3.2), both
    timezone-immune integer counters — so this item now covers only the
    decision-window scheduling and the update ordering, not warm-up.
11. **Reconciliation with `ADAPTS-HCT-Interaction-Flow.md`.** The older
    interaction-flow doc places `/update` and the `dyad_game` /action on
    **Sunday** morning, while §4 here places them on **Monday**.
    The older doc also describes only the once-per-week events and does
    not specify the daily AYA-AM / CP / AYA-PM cadence. Decision needed:
    update the older doc to match §4, or shift §4 to Sunday.
