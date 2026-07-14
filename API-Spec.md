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

Response `201`:

```json
{
  "status": "success",
  "message": "Group registered successfully.",
  "group_id": "dyad_007"
}
```

The canonical path is `/api/v1/register_group`. The former name
`/api/v1/add_group` is retained as a **deprecated alias** mapping to the same
handler, so existing host integrations keep working; new callers should use
`/register_group`.

**Re-registration with an existing `group_id` is an update, not an error.** When
`/register_group` is called for a `group_id` that is already registered, the API
overwrites that dyad's `consent_start_date` and `consent_end_date` from the
request body and returns `201` with `message: "Group consent window updated."`
(idempotent upsert). The `member_list` in the repeat request is **ignored** —
the members recorded at first registration stand. Re-registration is therefore
the supported mechanism for correcting a dyad's active window.

**Error responses.** Registration is one-time and off the real-time decision
path; the guiding principle is "never lose a recruitable dyad."

- `400` — a required field is missing or malformed. No partial row is
  written; correct and re-submit. (An existing `group_id` is **not** a `400`;
  it is the consent-window upsert described above.)
- `500` / `503` — database write failure. Safe to retry with backoff: the
  request writes exactly one row and has no other side effects. Registration
  may run any time before the dyad's first decision (§3), so a transient
  outage never affects an in-flight intervention.

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

Response `201`:

```json
{
  "status": "success",
  "message": "Action requested successfully.",
  "group_id": "dyad_007",
  "action": 1,
  "action_prob": 0.65,
  "rid": "a1b2c3d4",
  "timestamp": "2026-06-10T07:30:01",
  "warmup": false
}
```

- `action` ∈ {0 (do not send / game off), 1 (send / game on)}.
- `action_prob` is **Pr(chosen action)**, not Pr(action = 1). Analysis code must convert:
  `pi1 = action_prob if action == 1 else 1 - action_prob`. During warm-up `action_prob`
  is always `0.5`.
- `rid` — unique id for this action; the host should retain it for reference.
- `warmup` — `true` if this decision was a pure `Bernoulli(0.5)` draw (the learner was
  bypassed); `false` if the learner produced it. Warm-up is managed entirely by the
  API — the host does not need to act on this flag; it is surfaced for logging only.
**Idempotency key:** `(group_id, decision_type, decision_idx)`. Each of the three agents
(`aya_message`, `cp_message`, `dyad_game`) has its own per-dyad counter, so the same
`decision_idx` value can legitimately appear once per `decision_type` for the same dyad.

**Error responses.** `/action` is the only hard-real-time endpoint — the host
is about to act — so a decision must never be lost to an error. Whenever the
host cannot get a usable response at decision time, the recommended local
fallback is always the same: **draw `Bernoulli(0.5)` yourself and record
`action_prob = 0.5`**.

- `400` — duplicate `(group_id, decision_type, decision_idx)`. The triple is
  the idempotency key: a repeat is rejected outright, so a duplicate
  submission never mints a second action.
- `400` — malformed request envelope (missing or type-invalid field).
  Correct and re-send.
- `404` — `group_id` not registered, or model parameters not initialized.
  Apply the local fallback for this decision and fix the registration /
  deployment before the next one.
- `409` — no `/upload_data` has ever been received for this dyad, so the API
  has no values to construct a state. Send the first snapshot before the
  first `/action`; if a decision is due right now, apply the local fallback.
- `500` — internal error, including a failure inside the learner. Apply the
  local fallback. **[planned — tracked in `IMPLEMENTATION_TODO.md`: the API
  will instead return `200` with a randomized `action` and
  `action_prob = 0.5`, so the host proceeds as with any normal response.]**

### 2.3 `POST /api/v1/upload_data` — provide a full snapshot of dyad data

The host posts a **full snapshot** of every variable listed
in §4.1. There is **no** context/outcome distinction — the host does not need
to know which variables the learner uses as context and which as outcome.
There is also **no** `decision_type` or `decision_idx` — each upload is a
flat "current state of the dyad" snapshot, not tied to a particular decision.

**Upload schedule.** The host calls `/upload_data` **immediately before
every `/action`** — exactly one upload per `/action` call, in the same
sequence the decisions are delivered.

- **Monday morning** (3 uploads, run sequentially): pre-`dyad_game` upload →
  `POST /action dyad_game` (returns the week's game action $a^{(g)}$); then
  pre-AYA-AM upload → `POST /action aya_message`; then pre-CP upload →
  `POST /action cp_message`.
- **Tuesday–Saturday morning** (2 uploads, run sequentially): pre-AYA-AM
  upload → `POST /action aya_message`; then pre-CP upload →
  `POST /action cp_message`.
- **Every evening except Sunday** (1 upload): pre-AYA-PM upload →
  `POST /action aya_message`.

§4.1 specifies, for each variable, what value to send at each upload. For
most variables the value is identical across the uploads of a single
morning (the host's measurements don't change in minutes); the exceptions
are `current_game_on` and `prior_game_action`, whose values depend on
whether the Monday-morning `dyad_game` /action has happened yet (see §4.1).
For brevity §4.1 refers to "AM upload value" (= the value at any morning
upload, with Monday-morning exceptions noted) and "PM upload value".

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

**Semantics:**
- Every upload is a **full snapshot** — every field in §4.1 must be present.
  A field whose underlying measurement is unavailable for this upload must
  be sent as `"miss"` (or JSON `null`); it cannot simply be omitted.
- The learner masks `"miss"` values via the shared missing-indicator
  mechanism.
- The host does not tag uploads as "this is for AYA" / "this is the outcome
  of decision 12". The learner handles all such matching server-side at
  `/action` time (latest-snapshot lookup) and `/update` time (timeline-based
  reward derivation).

**Responses.** `201` on success. Uploads are the sole inputs to reward
derivation, so no posted data should ever be silently lost — on any error,
correct and re-send.

- `400` — missing key, unknown key, or type-invalid value (see §4.1 for the
  accepted set). The snapshot is all-or-nothing: a field with no measurement
  must be sent explicitly as `"miss"`. Correct and re-send.
  **[planned — tracked in `IMPLEMENTATION_TODO.md`: rejected payloads will
  additionally be preserved verbatim for post-trial analysis.]**
- `404` — unknown `group_id`. Register the dyad, then re-send the upload.
- `500` / `503` — database write failure. Safe to retry: uploads are
  append-only, and a duplicate row is harmless (the latest-value lookup is
  unchanged).

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

**Upload events.** The host calls `/upload_data` **once before every
`/action` call** (see §2.3 for the per-day schedule). The number of
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
| `state` | JSON | the feature vector `phi(s, a)` fed to the learner |
| `action` | int | chosen action ∈ {0, 1} |
| `action_prob` | float | Pr(chosen action), not Pr(action = 1); always `0.5` on warm-up rows |
| `is_warmup` | bool | `true` if this decision was a `Bernoulli(0.5)` warm-up draw (learner bypassed), per §2.2 |
| `warmup_reason` | string (nullable) | `cohort` / `week1` on warm-up rows; `NULL` otherwise |
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
| `update_id` | string | UUID returned in the 202 response |
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
disk under `repro_snapshots/<update_id>/`; this table is the index. Consumed
by `tools/reproduce_run.py` to replay a study. (`study_data` is itself
derived during `/update`, so the snapshot copies the upstream
`data_uploads`.)

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

### 7.1 Per-endpoint server-side fallback (summary)

Each endpoint's error handling is detailed inline in §2; this table
consolidates the "what happens on failure" view. Guiding principle by
endpoint: `/register_group` — never lose a recruitable dyad; `/action` — always return
a safe action, never "no decision"; `/upload_data` — preserve every byte, never
silently drop; `/update` — never publish a bad policy, keep the last good one.

| Endpoint | Failure | Behavior & recommended host action |
|---|---|---|
| `/register_group` | duplicate `group_id` | `201` — consent-window upsert, not an error |
| `/register_group` | missing/invalid field | `400` — no partial row; correct and re-submit |
| `/register_group` | DB write failure | `500` / `503` — safe to retry with backoff |
| `/action` | internal learner failure | today `404`/`500` — host decides locally (`Bernoulli(0.5)`, `action_prob = 0.5`); **[planned]** `200` with randomized `action`, `action_prob = 0.5` |
| `/action` | no upload history | `409` — host decides locally (`Bernoulli(0.5)`, `action_prob = 0.5`) |
| `/action` | duplicate decision triple | `400` — no second action minted |
| `/action` | unregistered `group_id` | `404` — host decides locally and fixes the registration |
| `/action` | sample buffer unavailable | `200` with `action_prob = 0.5`; infrastructure alert raised |
| `/upload_data` | malformed/schema-invalid payload | `400` — correct and re-send; **[planned]** payload preserved verbatim for post-trial analysis |
| `/upload_data` | missing key | `400` — full snapshot required; send `"miss"` explicitly |
| `/upload_data` | unknown `group_id` | `404` — register the dyad, then re-send |
| `/upload_data` | DB write failure | `500` / `503` — safe to retry (append-only) |
| `/update` | sanity-gate failure | fit not published; previous parameters stay active; request marked `failed` |
| `/update` | empty batch | fit skipped; previous parameters stay active; request marked `completed` |
| `/update` | per-action reward-derivation error | that action skipped (`reward = NULL`); the rest derived |
| `/update` | background-thread crash / timeout | request stays `processing`; monitoring re-triggers; previous parameters stay active |

---

## 8. Open items (to resolve with the dev team)

1. **Scheduling robustness:** timezone/DST handling for per-dyad decision
   windows, and the guarantee that the weekly `/update` completes before the
   first `/action` of the week.
