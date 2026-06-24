# Algorithm Monitoring

This document describes the algorithm-monitoring layer for the ADAPTS-HCT RL service. It is a clean-Markdown rendering of the "Algorithm Monitoring" and "Database Schema" sections of the authoritative design doc (`Overleaf/Algorithm_Design/main.tex`, §Algorithm Monitoring and §Database Schema).

The RL service produces an auditable per-decision and per-update event stream. *Algorithm monitoring* is the layer that turns that stream into actionable alerts when the running algorithm deviates from its specified behavior.

We adopt the three-pillar framework of Trella et al. (2026), developed across the Oralytics and MiWaves trials:

1. **Enumerate** potential issues and classify each by a *severity* level (red, yellow, green).
2. **Fall back** — specify pre-deployed methods that preserve participant experience and data integrity in real time when an issue fires.
3. **Document** every incident in enough detail to support post-deployment statistical analyses.

The implementation lives in `Monitoring_Algorithm/`, registered into the Flask app as a blueprint at `/api/v1/monitor`, together with in-process hooks at the relevant routes and a `flask monitor` CLI subgroup.

---

## 1. Severity taxonomy

Severity is assigned per check. The operational principle (from Trella et al. 2026, Table 1) is that **an effective fallback can downgrade a red issue to a yellow one** by safeguarding the participant in real time even while the algorithm itself remains broken. Red issues with no viable fallback — most often retroactive ones, like a burdensome cumulative number of prompts only visible at week's end — must still be enumerated and surfaced.

| Severity | Scope | ADAPTS-HCT exemplar | Action |
|---|---|---|---|
| **Red** | Compromises participant experience or the scientific utility of the trial data. | Cumulative weekly prompt count for an AYA exceeds the protocol cap; `decision_idx` gap (the algorithm acted in the world without a logged decision). | On-call engineer paged and study PI emailed; remediate, document, and verify resolution same day. |
| **Yellow** | Compromises the algorithm's ability to learn or to select actions, through a technological (not statistical/modeling) failure. | EB fit returns a non-PSD covariance; the host's `/upload_data` POST is rejected; a per-dyad week-1 baseline is not populated by day 9. | Dev-team digest; the fallback (§2) keeps the system running while the dev team triages. |
| **Green** | Inconsistencies or events that need no remediation but **must** be documented to support valid post-deployment analyses. | Outcome-feature missingness drift; weekly reward-distribution shift; the resolution log of any red or yellow incident. | Routed to the data analyst; appended to the documentation log used by the primary post-trial analysis. |

---

## 2. Fallback methods

Fallbacks are pre-specified procedures that execute automatically when a monitoring check fires, so the intervention keeps behaving safely while the issue is investigated. Following Trella et al. (2026, Table 2), we specify a fallback at each of the two RL pipeline stages — *action selection* and *learning/update* — and treat the fallbacks as part of the deployed algorithm, not out-of-band emergency behavior. Every fallback execution is recorded on its corresponding `MonitorEvent` so fallback frequency can be reported alongside the issue rate.

When the RL API fails to return a usable `(A, π)` pair (network failure, malformed response, fallback flag set), the host runs a random policy `A ~ Bernoulli(0.5)` locally and records the decision with `π = 0.5`. This keeps the intervention functioning during transient outages and keeps a known propensity for downstream off-policy analyses.

### Action selection (`/api/v1/action`)

| ID | Trigger | Fallback behavior |
|---|---|---|
| **F-A1** | API unreachable (network failure, RL service down). | Host product rule: draw `A ~ Bernoulli(0.5)` locally and act on it; the host logs the decision with `π = 0.5` and `fallback_flag="API_UNREACHABLE"`. When the API recovers, the host backfills the missed `Action` rows so the decision-index stream stays contiguous; these rows are flagged `excluded_from_update=TRUE` so the random-policy decisions do not enter the next EB fit. |
| **F-A2** | API reachable but cannot compute a valid `π` for `(i, g)`: corrupted posterior, missing state row, or missing week-1 baseline. | API returns 200 with `action_prob=0.5` and `fallback_flag="POLICY_UNAVAILABLE"`; the host draws `A ~ Bernoulli(0.5)` and persists the row with the fallback flag. The decision is excluded from the next `/update`. |

### Learning / update (`/api/v1/upload_data`, `/api/v1/update`)

| ID | Trigger | Fallback behavior |
|---|---|---|
| **F-U1** | `/upload_data` payload is malformed or fails schema validation. | API responds 4xx and persists a `StudyData` row with the raw payload captured verbatim and `excluded_from_update=TRUE`. The row is preserved for post-trial analysis but excluded from the next EB fit. |
| **F-U2** | `/update` fit fails the C3 posterior-sanity check (NaN/∞, non-PSD, trace blow-up). | A new `ModelParameters` row is **not** published; the previous parameters stay in force. The `ModelUpdateRequests` row is marked `status="failed"` with the failure reason captured. |
| **F-U3** | `/update` batch is empty (no new uploads since the last successful update). | Update is skipped; previous parameters remain. Logged at green severity for the data analyst (helps interpret apparent flat-lines in learning curves). |

**What we do not adopt.** Oralytics deployed a fourth fallback that pushed a 70-day backup intervention schedule to the app each morning, so the app could keep delivering prompts even if the RL service was unreachable for days. ADAPTS-HCT does not adopt this because (i) the API is per-decision rather than per-week — the host fetches an action when it is about to act — and (ii) the trial's product default of *no message sent* (F-A1) is itself safe; unlike Oralytics, where a blank schedule meant a participant got nothing all week.

---

## 3. Checks

The checks are grouped by the pipeline stage that produces the relevant data and ordered within each group by severity. Thresholds are stated per *active* dyad, where active means a dyad inside its consent window with at least one `Action` row newer than 2× the agent's decision cadence (12 h AYA, 24 h CP, 7 d REL).

Each check has one of three triggering surfaces:

- **In-process hook** — fired inside `app/routes/` immediately after the relevant row is committed (the hook is defined never to raise).
- **Cron blueprint** — run on a daily timer via `flask monitor run` and the read-only `POST /api/v1/monitor/run` endpoint.
- **Blocking gate** — for C3 alone, run *before* the new `ModelParameters` row is published.

Severity codes: **R** = red, **Y** = yellow, **G** = green. "Fallback" lists the entry from §2 executed when the check fires; "—" means the check is informational and triggers no fallback.

### A. Action selection (`/api/v1/action`, `Action` table)

| ID | Sev. | Check | Threshold | Cadence | Fallback |
|---|---|---|---|---|---|
| A1 | R | Clipped-probability streak: `π[g][i,k] ∉ (p, 1−p)` for `k` consecutive decisions of dyad `i`, with `p = 0.1`. | `k ≥ 8` (AYA) / 7 (CP) / 4 (REL); `2k` raises red urgency. | per `/action` | — |
| A4 | R | Per-agent active-dyad fraction with `A=1` at a single decision point: for each agent `g`, the mean of `A[g][i,k]` over active dyads at decision `k`. | out of `[0.2, 0.8]` for 3 consecutive decision points of that agent. | per decision pt (per agent) | — |
| A5 | R | Per-agent mean `π` across active dyads at a decision point. | out of `[0.15, 0.85]` for 3 consecutive decision points of that agent. | per decision pt (per agent) | — |
| A6 | R | Weekly cumulative burden (upper): the sum of `A[g][i,k]` over the week exceeds the protocol cap `K[g]`. | `K_AYA, K_CP, K_REL` to be finalized with the study team. | weekly (end-of-week) | — (retrospective) |
| A7 | R | Weekly starvation (lower): the weekly sum of `A[g][i,k]` falls below the protocol floor `L[g]`. Mirror of A6 on the under-prompting side. | `L_AYA, L_CP, L_REL` to be finalized with the study team. | weekly (end-of-week) | — (retrospective) |
| A3 | R | `decision_idx` non-contiguity *or* duplication per `(i, g)`: fires if any pair has `cur − prev > 1` (gap) or if the tuple `(i, g, decision_idx)` has ≥ 2 rows (duplicate). | any gap or any duplicate. | per `/action` | — |
| A2 | Y | Daily action-count check: actual `/action` rows today < protocol expected count for that weekday. Sundays expect 0 for all agents. | expected: AYA 2, CP 1, REL 1 on Mon (REL 0 otherwise); Sun 0 everywhere. | daily evening cron | F-A1, F-A2 |
| A8 | Y | Consent-window guard: `/action` call received for dyad `i` outside `[consent_start_date, consent_end_date]`. Route returns the F-A2 sentinel and increments a per-day counter. | any individual request; daily roll-up. | per request | F-A2 |

### B. Data update (`/api/v1/upload_data`, `StudyData` table)

| ID | Sev. | Check | Threshold | Cadence | Fallback |
|---|---|---|---|---|---|
| B5 | R | Week-1 standardization baseline for `(i, g)` not populated by day 9 of dyad enrollment. | missing ⇒ red, **and** the policy serves F-A2 for that `(i, g)` until repaired. | daily | F-A2 |
| B6 | R | Active-dyad set consistency: `Group` rows with a valid consent window do not match the host's official trial roster (e.g. delivered as a nightly manifest). | any divergence. | daily | — |
| B1 | Y | Failed `/upload_data` request: the host's POST was rejected (bad payload, server error, or auth failure). | *every* failed request fires yellow; sustained failure share > 20% over rolling 24 h escalates to red. | per `/upload_data`; daily rollup | F-U1 |
| B2 | Y | Quietly-NULL reward: fraction of `StudyData` rows with `reward IS NULL` **and** `excluded_from_update=FALSE` per agent over rolling 7 d. (Restricting to `excluded_from_update=FALSE` avoids double-counting the F-U1 path, already covered by B7.) | > 5% yellow, > 15% escalates to red. | daily | — |
| B4 | Y | Bidirectional `Action` ↔ `StudyData` reconciliation: every `Action` older than `Δ[g]` (48 h / 7 d / 14 d for AYA / CP / REL) must have exactly one matching `StudyData` row, **and** every `StudyData` row must reference an existing `Action`. | any unmatched in either direction (zero-tolerance). | daily | — |
| B7 | Y | Semantic validation of `/upload_data`: payload passes schema validation but a field violates its protocol-defined support (e.g. AYA reward outside `{0,1,2,3}`; CP/REL reward outside `[1,5] ∪ {0}`; `outcome_completed=True` with the required outcome fields absent). | any row. | per `/upload_data` | F-U1 |
| B3 | G | Outcome-feature missingness drift: per-agent rate of missing `v_j` (the `I_j` indicator in `φ`). | `|Δ| > 0.15` vs prior 7 d. | daily | — |

### C. Policy update (`ModelUpdateRequests`, `EmpiricalBayesSnapshot`)

| ID | Sev. | Check | Threshold | Cadence | Fallback |
|---|---|---|---|---|---|
| C1 | Y | Update-job status: `failed` or `completed_at IS NULL` more than 30 min after `request_timestamp`. | any. | per `/update` | F-U2 |
| C3 | Y | Posterior sanity (blocking gate): `theta`/`covariance` contain NaN/∞, non-PSD covariance, or `tr Σ > 10 × tr Σ0[g,cold]` (cold-start diagonal). Runs *before* `ModelParameters` is published. | any failure. | per `/update` | F-U2 |
| C5 | Y | Weekly reproducibility audit (`tools/reproduce_run.py`): replay the past week's event log against the last `EmpiricalBayesSnapshot` and the deterministic buffer cursor; assert every logged `action_prob` (and sampled `A`) matches bit-for-bit. | any mismatch fires yellow; mismatch on ≥ 2 consecutive weeks escalates to red. | weekly | — |
| C2 | G | Update latency = `completed_at − request_timestamp`. | > 5 min green, > 15 min escalates to yellow. | per `/update` | — |
| C4 | G | EB-Gradient convergence: hyper-update EM did not converge in `K_max` iterations or hit the diagonal floor `τ_k² ≤ 1e-6` on ≥ 1 component. | any (post-warmup). | per `/update` | — |

### E. Infrastructure and access

| ID | Sev. | Check | Threshold | Cadence | Fallback |
|---|---|---|---|---|---|
| E1 | R | Process resource exhaustion on the RL host: memory, disk, or DB connection pool above threshold. | memory > 85%, disk > 90%, pool > 90%. | every 5 min | downstream F-A1 / F-U2 |
| E2 | Y | Unauthorized request: shared-secret header missing or invalid on `/upload_data`, `/update`, or `/action`. | any individual request; rate rolled up daily. | per request | — |
| G1 | G | Fallback execution counter: count of each fallback (F-A1, F-A2, F-U1, F-U2, F-U3) fired today, per agent, **and** broken down by sub-cause (e.g. F-A2 by {corrupted posterior, missing state, missing baseline, feature-builder exception}). | report-only. | daily | — |

### D. Data export

| ID | Sev. | Check | Threshold | Cadence | Fallback |
|---|---|---|---|---|---|
| D1 | G | Daily DB export health: `flask export-csv` completed, all tables present, per-table row counts within ±20% of the prior day's, and the export-manifest checksum was written. | any missing table, row-count delta out of band, or absent checksum. | daily | — |

> The infrastructure check **E1** was added in anticipation of the failure mode that prompted MiWaves to retrofit a low-memory check mid-trial.

### Check notes

**A3 (decision-index gaps and duplicates).** The API assigns `decision_idx = 0, 1, 2, …` to consecutive `/action` calls within each `(i, g)`. A3 scans those rows ordered by `decision_idx` and fires on either failure of the contiguous-sequence invariant: a *gap* (`cur − prev > 1`) or a *duplicate* (the tuple `(i, g, decision_idx)` has multiple rows). A gap catches silent insert loss, a non-`+1` next-index computation in `app/routes/action.py`, or an out-of-band deletion; a duplicate catches the symmetric race-condition double-insert and the host-retry-against-a-successful-write pattern reported in MiWaves. All are red because Inf-LSVI's Bellman target indexes successor states by `k+1`: a hole corrupts the target and a duplicate yields two different `(A, π)` assignments at the same time index. A3 is complementary to A2: A2 catches `/action` calls stopping altogether (downgraded to yellow by F-A1 / F-A2 because the missing decision is recorded as "no message"); A3 catches calls that keep arriving but violate the local invariant, which no fallback can repair after the fact.

**A6 (weekly burden).** A1 detects boundary clipping decision-by-decision and A4/A5 detect imbalance at a single decision point; none catch the slow case where `π` oscillates near 0.5 yet the total number of `A=1` draws over a week exceeds the cumulative burden the protocol committed to. A6 closes that gap by summing `A[g][i,k]` over rolling 7 d. This is one of the red severity classes with **no** viable real-time fallback — once a participant has been over-prompted across the week the harm is done — so the value of the check is to detect, document, and let the study team adjust enrollment or thresholds for later weeks. The per-agent caps `K[g]` are open, to be finalized with Billie and the study team before deployment.

**B5 / F-A2 interaction.** Without a per-dyad week-1 standardization baseline, the feature builder in `app/feature_builder.py` cannot construct a well-defined `φ(s, a)` for dyad `i`; the prior covariance is also calibrated against standardized inputs. B5 fires red if the baseline is missing past day 9. F-A2 then keeps the system running for that dyad by serving `action_prob=0.5` with the fallback flag set; the resulting decision is excluded from the next `/update`. Together they realize the recommended pattern: detect at red severity, downgrade the participant-facing impact to yellow via a fallback, and let the data analyst handle the missing learning signal post-hoc.

**B6 (active-dyad consistency).** MiWaves found a yellow incident where fraudulent participants were removed by staff but not communicated to the RL algorithm, causing one participant to be served another's decision rule. B6 forestalls this by reconciling the set of `Group` rows with valid consent windows against the host's official trial roster. The mechanism for delivering that roster (nightly signed manifest, host endpoint queried at check time, etc.) is a host-API contract to be finalized before deployment.

**C5 (weekly reproducibility audit).** Every Sunday after `/update` completes, the runner invokes `tools/reproduce_run.py` against the ordered API event log for the past week, the previous week's `EmpiricalBayesSnapshot`, and the deterministic sample-buffer cursor recorded on each `Action` row. The audit succeeds iff the recomputed `(A, π)` for every logged decision matches the persisted value bit-for-bit — the FDA / IRB-style replayability guarantee of the deterministic-sampler design. A mismatch is yellow because the algorithm is still serving valid actions, but the snapshot-to-live coupling that supports post-trial causal analysis has broken; two consecutive weeks of mismatch escalates to red because the scientific utility of the affected decisions becomes suspect. Unlike C3, this check has no fallback — there is nothing to "serve safely"; the cure is for the dev team to diagnose the divergence (nondeterministic dependency, buffer cursor drift, snapshot write race, schema change) before the next analysis cut.

**C3 as a blocking gate.** C3 is the only check with the authority to reject an algorithm output. Because Inf-LSVI feeds the smooth-logistic allocation, a non-PSD covariance yields an ill-defined posterior contrast `(m, v)`, and an exploded trace (relative to the cold-start diagonal) is a near-certain symptom of a degenerate fit. F-U2 keeps the previous `ModelParameters` row in force until the next `/update`; this preserves continuity of decision-making while the cause is investigated, and is exactly the mechanism that downgrades a red would-be issue to a yellow one.

---

## 4. Operational interface

Every fired check writes a row to a `MonitorEvent` table with fields `(check_id, severity, value, threshold, computed_at, affected_groups, fallback_executed, resolution)` and is surfaced through four channels:

- **Automated email alerts.** Each **R** event fires immediately to the on-call engineer and the study PI; **Y** accumulates into a daily digest to the RL dev team; **G** accumulates into a weekly digest to the data analyst. The schema follows Trella et al. (2026, "Operational Workflow"): timestamp, affected `(i, g)` list, fallback executed, and a free-text resolution field the dev team fills in to close the incident.
- **Daily positive-monitoring email.** Mirroring MiWaves (Fig. 2(b)), a daily email summarizes per-agent counts of `Action` rows and the share with `A=1` over rolling 7 d, alongside the protocol-expected counts. This is not an alert — it is a positive signal that the system is running as expected — and is sent regardless of whether any event fired.
- **Dashboard.** The JSON blueprint at `/api/v1/monitor` is the data layer; a Grafana or Retool dashboard sits on top as the manual-review surface, reviewed daily by the study coordinator. Notification transport (PagerDuty, SES) is deployment-specific and not pinned here.
- **Quantitative report.** `flask monitor report --since DATE` emits the publication-ready summary: one table per severity with (issue ID, # occurrences, # affected dyads, fallback executed) plus a per-fallback execution roll-up over the requested window. This is the artifact the post-trial paper draws on.

---

## 5. Adaptation during the trial

The check list above is the pre-deployment design and is expected to evolve. Precedent: MiWaves added a cloud-memory red check mid-trial after two crashes, and Oralytics added several yellow checks after observing recurrent silent failures in the backend controller. ADAPTS-HCT's three-agent, multi-rate structure — in particular the weekly REL agent, whose update lag is comparable to a typical incident-response window — is a likely source of monitoring patterns not covered by the two prior case studies. The dev team should plan for at least one revision cycle to the check list within the first eight weeks of deployment.

---

## 6. Database schema

These are the ORM tables maintained by the live API (`app/models.py`). The schema covers eight tables; the columns referenced by the §3 checks and by the C5 reproducibility audit are noted inline. A ninth table, `MonitorEvent`, is part of the §4 design but is not yet present in `models.py`; its schema is given below as the agreed columns for the monitoring blueprint.

1. **Group** (`groups`) — one row per enrolled dyad with protocol metadata and the warm-up flag.
2. **Action** (`actions`) — one row per `/action` call carrying the `(s, a, π)` tuple, the sample-buffer cursor, and the FK to the `ModelParameters` row used.
3. **StudyData** (`study_data`) — one row per `/upload_data` call carrying the outcome and the constructed reward.
4. **ModelParameters** (`model_parameters`) — row-id anchor used by every `Action` so the policy under which the decision was made is recoverable.
5. **ModelUpdateRequests** (`model_update_requests`) — one row per `/update` call, with status, timestamps, and failure message.
6. **EmpiricalBayesSnapshot** (`empirical_bayes_snapshots`) — persists local Inf-LSVI fits, the pooled EB hyper-fit, and per-dyad posterior summaries.
7. **StandardizationBaseline** (`standardization_baselines`) — one row per `(i, g, j)` for the per-dyad week-1 baselines used by the feature builder; B5 reads this.
8. **UpdateReproducibilitySnapshot** (`update_reproducibility_snapshots`) — pointer to the on-disk full DB copy taken immediately before an `/update` completes; C5 reads this.
9. **MonitorEvent** (`monitor_events`) — one row per fired check (design only; not yet in `models.py`).

### Group (`groups`)

One row per enrolled dyad.

| Column | Type | Description |
|---|---|---|
| `id` | Integer (PK) | Auto-increment primary key. |
| `group_id` | String (unique) | Dyad identifier supplied by the host at `/add_group`. |
| `group_info` | JSON | Protocol metadata: `consent_start_date`, `consent_end_date`, agent-enable flags, dyad-specific timing. B6 reconciles the active subset against the host roster. |
| `warmup` | Boolean | Recomputed on every `/action` call as `1{request date < consent_start_date + 7 days}`; when True, the learner is bypassed and `action_prob = 0.5` is served, so each dyad's first calendar week is purely randomized. |
| `created_at` | DateTime | Row-insertion timestamp. |

### Action (`actions`)

One row per `/action` call.

| Column | Type | Description |
|---|---|---|
| `id` | Integer (PK) | Auto-increment primary key. |
| `group_id` | String | Dyad identifier; FK to `groups.group_id`. |
| `rid` | String (unique) | Host-supplied request identifier; the primary cross-reference for the corresponding `StudyData` row at `/upload_data`. |
| `state` | JSON (nullable) | Standardized feature vector `φ(s, ·)` the algorithm consumed at this decision. NULL only when F-A2 was served with no state available. |
| `decision_idx` | Integer | Per-`(group_id, decision_type)` monotonic counter starting at 0. A3 enforces contiguity and uniqueness on this column. |
| `decision_type` | String | One of `aya_message`, `cp_message`, `dyad_game`. |
| `raw_context` | JSON | The host's raw payload before standardization (used to rebuild `s` for the C5 audit). |
| `action` | Integer ∈ {0,1} | Sampled action `A`. |
| `action_prob` | Float ∈ [0,1] | Sampling probability `π` at the time of decision. A1/A4/A5 read this. |
| `random_state` | JSON | Cursor into the deterministic sample buffer recording the Bernoulli draw that produced `action`. C5 replays from this. |
| `model_parameters_id` | Integer (FK) | FK to `model_parameters.id` of the policy used; allows post-hoc identification of which posterior produced this decision. |
| `request_timestamp` | DateTime | The host's request time. |
| `timestamp` | DateTime | Row-insertion time. |

### StudyData (`study_data`)

One row per `/upload_data` call. (An `excluded_from_update` boolean column, used by F-U1 and consumed by B2/B7, is part of the design and to be added.)

| Column | Type | Description |
|---|---|---|
| `id` | Integer (PK) | Auto-increment primary key. |
| `group_id` | String | Dyad identifier; FK to `groups.group_id`. |
| `decision_idx` | Integer | Matches the `Action` row's `decision_idx` for B4 reconciliation. |
| `decision_type` | String | Matches the `Action` row. |
| `action` | Integer | Copy of `Action.action` for QA. |
| `action_prob` | Float | Copy of `Action.action_prob` for QA. |
| `state` | JSON | Copy of `Action.state`. |
| `raw_context` | JSON | Copy of `Action.raw_context`. |
| `outcome` | JSON | Host's outcome payload (the per-agent reward fields). |
| `reward` | Float (nullable) | Scalar reward built from the outcome schema. B2 monitors the rate of NULLs with `excluded_from_update=False`; B7 fires when `outcome` passes the schema but violates a field's support. |
| `request_timestamp` | DateTime | The host's `/upload_data` request time. |
| `created_at` | DateTime | Row-insertion time. |

### ModelParameters (`model_parameters`)

Row-id anchor for `Action.model_parameters_id`.

| Column | Type | Description |
|---|---|---|
| `id` | Integer (PK) | Auto-increment primary key; every `Action` row's `model_parameters_id` points here. |
| `probability_of_action` | Float | For the `flat_prob` fallback algorithm. Under the EB path this column is unused at decision time; the operative posterior is recoverable via the `EmpiricalBayesSnapshot` row keyed by `(decision_type, agent_decision_index)` at `timestamp`. |
| `timestamp` | DateTime | When this parameter row became active. |

### ModelUpdateRequests (`model_update_requests`)

One row per `/update` call.

| Column | Type | Description |
|---|---|---|
| `id` | Integer (PK) | Auto-increment primary key. |
| `update_id` | String | Logical update identifier (one per scheduled `/update`). |
| `status` | String | One of `processing`, `completed`, `failed`. C1 fires when this is not `completed` 30 min after `request_timestamp`. |
| `callback_url` | String | Host endpoint to notify when the update completes. |
| `request_timestamp` | DateTime | The host's request time. |
| `created_at` | DateTime | Row-insertion time. |
| `completed_at` | DateTime (nullable) | Set on successful completion. C2 monitors `completed_at − request_timestamp`. |
| `error_message` | String (nullable) | Captured failure reason when `status='failed'`; populated by F-U2. |

### EmpiricalBayesSnapshot (`empirical_bayes_snapshots`)

Persists local fits, the hyper-pool, and posterior summaries.

| Column | Type | Description |
|---|---|---|
| `id` | Integer (PK) | Auto-increment primary key. |
| `snapshot_type` | String | One of `local` (per-dyad Inf-LSVI fit), `hyper` (EB-Gradient pooled hyperparameters), or `posterior` (per-dyad EB posterior from the final E-step). |
| `group_id` | String (nullable) | Dyad identifier for `local` and `posterior` rows; NULL for `hyper` rows (population-level). |
| `decision_type` | String | Per-agent (`aya_message`, `cp_message`, `dyad_game`). |
| `agent_decision_index` | Integer | Per-agent update counter, monotonic. |
| `sample_size` | Integer | Number of `(s, a, r)` tuples used to produce the row. |
| `feature_dim` | Integer | Dimension of `φ` for that agent. |
| `theta` | JSON | Flattened mean vector. |
| `covariance` | JSON | Flattened covariance matrix. C3 reads this and asserts PSD, finite, and `tr Σ ≤ 10 × tr Σ0[g,cold]` (cold-start diagonal). |
| `perturbation` | JSON (nullable) | RLSVI perturbation draw for the Inf-LSVI step, when present. |
| `metadata_json` | JSON (nullable) | Free-form notes: EM iterations to convergence, final marginal log-likelihood, `τ²` diagonal at termination (C4), warm-up flag at fit time. |
| `created_at` | DateTime | Row-insertion time. |

### StandardizationBaseline (`standardization_baselines`)

Per-`(i, g, j)` week-1 baselines; B5 fires if any required row is missing past day 9. Constraint: `UNIQUE(group_id, decision_type, variable_name)`.

| Column | Type | Description |
|---|---|---|
| `id` | Integer (PK) | Auto-increment primary key. |
| `group_id` | String | Dyad identifier. |
| `decision_type` | String | Per-agent. |
| `variable_name` | String | State-feature name (e.g. `adherence_rate`, `app_engagement`). |
| `mu` | Float | Week-1 baseline mean for this `(i, g, j)`. |
| `sigma` | Float | Week-1 baseline SD; the feature builder maps `v → (v − μ)/σ`. |
| `sample_size` | Integer | Number of week-1 observations used for the baseline. |
| `created_at` | DateTime | Row-insertion time (day 8 of dyad enrollment). |

### UpdateReproducibilitySnapshot (`update_reproducibility_snapshots`)

Anchors the C5 weekly reproducibility audit; `tools/reproduce_run.py` replays from `snapshot_dir`.

| Column | Type | Description |
|---|---|---|
| `id` | Integer (PK) | Auto-increment primary key. |
| `update_id` | String | Matches `ModelUpdateRequests.update_id`. |
| `model_parameters_id` | Integer (nullable) | The `ModelParameters` row published at the end of this update, when one is published. |
| `snapshot_dir` | String | On-disk directory containing the snapshot of `study_data`, `actions`, and `groups`. |
| `study_data_count` | Integer | Row count of the snapshot's `study_data` dump. |
| `actions_count` | Integer | Row count of the snapshot's `actions` dump. |
| `groups_count` | Integer | Row count of the snapshot's `groups` dump. |
| `total_bytes` | BigInteger | Total size of the on-disk snapshot. |
| `created_at` | DateTime | Row-insertion time. |

### MonitorEvent (`monitor_events`)

One row per fired check. Not yet present in `models.py`; specified here as the agreed schema for the monitoring blueprint.

| Column | Type | Description |
|---|---|---|
| `id` | Integer (PK) | Auto-increment primary key. |
| `check_id` | String | Identifier from the §3 check tables (e.g. `A1`, `B7`, `G1`). |
| `severity` | Enum | One of `red`, `yellow`, `green`. |
| `value` | Float (nullable) | Measured value that triggered the event, when scalar. |
| `threshold` | Float (nullable) | The configured threshold the value crossed. |
| `group_id` | String (nullable) | Affected dyad for dyad-scoped events. |
| `decision_type` | String (nullable) | Affected agent for agent-scoped events. |
| `affected_groups` | JSON (nullable) | List of dyads for population-scoped events (A4, A5). |
| `fallback_executed` | String (nullable) | One of F-A1, F-A2, F-U1, F-U2, F-U3, plus a sub-cause label (G1 breakdown). |
| `message` | String (nullable) | Human-readable summary used in the alert email. |
| `computed_at` | DateTime | When the check produced the event. |
| `resolution` | Text (nullable) | Free-text resolution note filled in by the dev team to close the incident. |

---

## Reference

Trella et al. (2026), *Effective monitoring of online AI decision-making algorithms in just-in-time adaptive interventions*, npj Digital Medicine (in press), doi:10.1038/s41746-026-02669-4. The severity taxonomy (§1), fallback structure (§2), operational workflow (§4), and trial-adaptation precedents (§5) follow this paper, drawing on the Oralytics and MiWaves deployments.
