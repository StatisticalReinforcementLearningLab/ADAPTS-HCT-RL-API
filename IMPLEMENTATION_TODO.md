# Implementation TODO

Tracked work items for the ADAPTS-HCT RL API. Each item lists the concrete code
and doc changes, plus an acceptance check. Cross-references point at
`API-Spec.md`.

> **Status (items 1–2 DONE; items 3–4 open).** Items 1–2 implemented in `app/routes/group.py`, with
> callers and docs updated; `pytest tests/` passes (72 tests). Notes:
> - `/add_group` is kept as a **deprecated alias** (the optional transition
>   alias), so the host contract is not broken. Both paths map to
>   `group.register_group`.
> - The simulator's internal event-type label `"add_group"` was **left
>   unchanged** (it is an in-process token, not the HTTP path) to minimize churn.
> - Doc `[planned]` flags removed and the corresponding `API-Spec.md` open
>   item closed (since deleted from §8).

---

## 1. Rename endpoint `/add_group` → `/register_group`  — DONE

**Why.** "Register" better describes the operation, and (with item 2) it stops
being a pure create. Resolves the `Change endpoint name to register_group` note
at the top of `API-Spec.md` §2.1.

**Required changes (the HTTP path is the contract surface):**

- [ ] `app/routes/group.py` — change the route decorator (line ~30) from
      `@group_blueprint.route("/add_group", ...)` to
      `@group_blueprint.route("/register_group", ...)`; rename the handler
      `def add_group()` → `def register_group()`. (Blueprint name and
      `url_prefix="/api/v1"` in `app/__init__.py` are unchanged.)
- [ ] `tests/test_groups.py` — update the imported symbol (`from
      app.routes.group import ... add_group`) and every `"/api/v1/add_group"`
      path string.
- [ ] `tests/conftest.py` — update the `"/api/v1/add_group"` path.
- [ ] `tests/simulate_adapts_hct.py` — update the
      `client.post("/api/v1/add_group", ...)` call (line ~525). Optionally
      rename the internal event-type label `"add_group"` → `"register_group"`
      for consistency (this is an in-process label, not the HTTP path — rename
      it everywhere or nowhere: lines ~82–161, ~521–530).
- [ ] Other test/tool callers that post to the path or key on the event label:
      `tests/run_simulation.py`, `tests/test_simulation.py`,
      `tests/resource_estimate.py`, `tests/test_warmup.py`.
- [ ] Docs — update the endpoint name in `API-Spec.md` (§2.1 header, §3 worked
      example, §7.1 fallback table), `README.md`, `bruno/README.md` + the Bruno
      request, `Algorithm-Monitoring.md`, and `Possible_System_Failure.md`.

**Optional — transition alias.** To avoid a breaking change for the host, keep
`/add_group` as a second route on the same handler for one release, logging a
deprecation warning, then remove it. Decide with the dev team.

**Acceptance.** `pytest tests/test_groups.py tests/test_warmup.py` passes
against the new path; `flask routes` lists `/api/v1/register_group`; a full
`tests/run_simulation.py` drive completes with no `404` on registration.

---

## 2. Allow re-registration to change consent start/end dates (upsert)  — DONE

**Why.** A repeat call for an existing dyad should correct its active window,
not error. Implements the `API-Spec.md` §2.1 target contract;
currently `app/routes/group.py` returns `400 "Group already exists."`.

**Required changes:**

- [ ] `app/routes/group.py` — replace the existing-group `400` branch
      (lines ~57–59) with an in-place update: when the `group_id` already
      exists, overwrite `group_info["consent_start_date"]` and
      `group_info["consent_end_date"]` from the request, **leave
      `member_list` unchanged** (ignore the repeat's `member_list`), commit,
      and return `201`. Reassign `group_info` (or use a JSON-mutation-safe
      update) so SQLAlchemy persists the change.
- [ ] `tests/test_groups.py` — replace `test_add_group_duplicate` (which
      currently asserts `400`) with a test that re-registers an existing dyad
      with new consent dates and asserts `201` **and** that the persisted
      `consent_start_date`/`consent_end_date` changed while `member_list` did
      not.
- [ ] Confirm the `400` path now fires **only** for missing/malformed fields
      (already covered by `test_add_group_missing_field`).
- [ ] Docs — once shipped, drop the `[planned: ...]` flags in `API-Spec.md`
      §2.1 and close the corresponding §8 open item.

**Acceptance.** Re-registering a dyad updates only the consent window; the
`groups` row keeps its original `member_list`; the endpoint returns `201`; the
old "group already exists" `400` no longer occurs.

---

**Suggested order.** Do item 2 first (behavioral change, isolated to
`group.py` + one test), then item 1 (mechanical rename touching many files), so
the rename lands on already-correct behavior in a single sweep.

---

## 3. `/action` learner-failure fallback (F-A2) — TODO

**Why.** `API-Spec.md` §2.2 (server-side fallback) specifies that when the
learner cannot produce a valid decision — corrupted or non-PSD posterior,
feature-builder exception, missing week-1 standardization baseline, sampler
error, or a cold-start race on `model_parameters` — `/action` must degrade to
a buffer-drawn `Bernoulli(0.5)` with `action_prob = 0.5` rather than surface
`404`/`500`. Today those paths return `404`/`500` and the host must fall back
locally (F-A1).

**Required changes:**

- [ ] `app/models.py` + migration — add `actions.excluded_from_update`
      (bool, default `FALSE`).
- [ ] `app/routes/action.py` — wrap the `make_state` / `get_action` path in a
      try/except; on failure draw `Bernoulli(0.5)` from the deterministic
      sample buffer (`_draw_warmup_action`), persist the row with
      `is_warmup = true` and `excluded_from_update = TRUE`, and return `200`
      with `action_prob = 0.5`. Same handling for the
      `model_parameters`-missing cold-start race.
- [ ] `/update` — skip `excluded_from_update` rows when building the fit pool.
- [ ] Tests — force a posterior/feature failure and assert `200`,
      `action_prob == 0.5`, and that the row is excluded from the next fit.

**Acceptance.** With a corrupted posterior injected, `/action` still returns a
usable `(action, action_prob)`, the decision replays deterministically from
the buffer cursor, and the next `/update` fit ignores the flagged row.

---

## 4. `/upload_data` preserve-rejected-payloads (F-U1) — TODO

**Why.** `API-Spec.md` §2.3 (server-side fallback) requires rejected uploads
(malformed / unknown-key / type-invalid) to be persisted verbatim with
`excluded_from_update = TRUE` for post-trial analysis ("preserve every byte,
never silently drop"). Today an invalid payload is rejected with `400` and
nothing is persisted.

**Required changes:**

- [ ] `app/models.py` + migration — add `data_uploads.excluded_from_update`
      (bool, default `FALSE`).
- [ ] `app/routes/data.py` — on validation failure, persist the raw payload
      in a `data_uploads` row flagged `excluded_from_update = TRUE`, then
      return the `4xx` unchanged.
- [ ] `app/routes/action.py` — the latest-snapshot lookup must skip flagged
      rows; `app/reward_derivation.py` must skip them on the timeline walk.
- [ ] Tests — an invalid upload returns `400`, a flagged row exists, and
      `/action` + `/update` behave as if it were never sent.

**Acceptance.** No byte posted to `/upload_data` is ever lost, and flagged
rows are invisible to the learner (state construction and reward derivation).
