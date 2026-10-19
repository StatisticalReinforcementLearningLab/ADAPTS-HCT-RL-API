# Implementation TODO

Tracked work items for the ADAPTS-HCT RL API. Each item lists the concrete code
and doc changes, plus an acceptance check. Cross-references point at
`API-Spec.md`.

> **Status (2026-10-19).** API items 1–4 and 8 are implemented. Items 5–6 were
> superseded by the finalized Monitoring Spec: legacy check IDs remain, and
> alarm responses include `owner` and `raised_at`. Item 7 is a simulation and
> manuscript rerun, not an API implementation item. The current full suite is
> the acceptance source of truth. Notes:
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

## 3. `/action` learner-failure fallback (F-A2) — DONE

Implemented by `app/routes/action.py`: no-state, no-policy, learner-error and
invalid-probability cases return a persisted `201` decision using exactly one
saved uniform, with `action_prob = 0.5`, `warmup = true`, and a machine-readable
`warmup_reason`. Such a decision enters learning when both its design state and
outcome are available. Route and monitoring tests cover every fallback cause,
replay, buffer use, and the A13 invalid-probability gate.

---

## 4. `/upload_data` preserve-rejected-payloads (F-U1) — DONE

Implemented with the separate access-controlled `upload_quarantine` table.
Every rejected `400` body is stored byte-for-byte with its validation result,
SHA-256 fingerprint and size; it never enters `data_uploads`, decisions,
rewards, ordinary backups, logs, or alarms. The binary representation also
preserves malformed non-UTF-8 requests exactly.

---

## 5. Consistent ID format for monitoring checks and error codes — NOT ADOPTED

**Why.** `Monitoring-Spec.md` uses two ID schemes side by side: monitoring
checks carry short legacy IDs (`A13`, `C7`, `L2`, ...) while endpoint errors
carry structured codes (`AC-409-NOSTATE`, `UP-500-...`; §6.1). The two look
unrelated in alarm feeds and in `GET /monitor/alarms` (`code` field, §8.1).
Give checks the same `<endpoint>-<kind>-<short name>` shape, e.g.
`A13` → `AC-13-RANGE` (π out of range on `/action`).

**Proposed format.**

```
CHECK = <endpoint prefix>-<check number>-<SHORT_NAME>   e.g. AC-13-RANGE
ERROR = <endpoint prefix>-<HTTP status>-<SHORT_NAME>    e.g. AC-409-NOSTATE
```

The middle slot stays unambiguous: checks use 1–2 digits, errors use a 3-digit
HTTP status. Prefixes are the §6.1 ones (`RG`, `UD`, `AC`, `UP`, `MA`, `XX`).

**Required changes:**

- [ ] Decide the section-letter → prefix map (A → `AC`; B → `RG`/`UD` per
      check; C, L → `UP`; E → `MA`/`XX`; D, G → `XX`) and a short name for
      every check (A2–A15, B3–B11, C1–C11, L1–L4, D1, D3, E1–E7, G0).
- [ ] `Monitoring-Spec.md` — rename IDs in §5 tables, §7 fallbacks, §8.1
      examples, §11 work items/tests, and §12 open questions; extend §6.1 to cover
      checks; add a new-ID column to Appendix A so old notes stay readable.
- [ ] `API-Spec.md`, `Algorithm-Monitoring.md` — update any check-ID references.
- [ ] Code — use the new IDs wherever `monitor_alarms.code` is written.

**Acceptance.** Every alarm and error ID matches
`^(RG|UD|AC|UP|MA|XX)-[0-9]{1,3}-[A-Z][A-Z0-9_]*$`, and Appendix A maps each
old ID to its new one.

---

## 6. `/monitor/alarms`: return only the message, no extra alarm fields — NOT ADOPTED

**Why.** `Monitoring-Spec.md` §8.1 proposes adding `owner` and `host_action` to
each alarm returned to the host **[open #2]**. Keep the host contract of
`API-Spec.md` §2.5 instead: each alarm already carries a human-readable
`message`, and that is what the host displays. Whatever the host needs to know
(including what it should do) goes into the `message` text. This closes open
#2 against adding the two fields.

**Required changes:**

- [ ] `Monitoring-Spec.md` §8.1 — drop the `owner` and `host_action` rows from
      the additions table and from the JSON example; state that the host
      action, when there is one, is written into `message`.
- [ ] `Monitoring-Spec.md` §8.2 — remove the reliance on `owner` for routing
      alerts (the host routes on `severity` only, as in `API-Spec.md` §2.5).
- [ ] `Monitoring-Spec.md` §9.1 — keep `owner` and `host_action` as internal
      `monitor_alarms` columns if useful, but do not send them; build the
      outgoing `message` from them.
- [ ] `Monitoring-Spec.md` §12 — update open #2 with this decision.
- [ ] Code — the `/monitor/alarms` serializer returns only the §2.5 fields
      (`event_id`, `item_id`, `severity`, `message`, `detail`).

**Acceptance.** The `/monitor/alarms` response matches `API-Spec.md` §2.5 with
no added per-alarm fields, and each host-owned alarm's `message` says what the
host should do.

---

## 7. Update the appendix and the m sweep for the 2/1/0 AYA reward — TODO

**Why.** On 2026-10-13 `Overleaf_bridge/Algorithm_Design/main_new.tex` switched
the AYA reward \eqref{eq:reward-aya} from (3, 0) to (2, 1, 0): 2 = dose reported
taken, 1 = reported missed, 0 = no report (S. Murphy; E23 in
`Simulation_Testbed/`). Every experiment in the testbed appendix and the m sweep
(Figure `fig:runin-m`) still ran under the (3, 0) coding, and the doc says so
in the appendix's *Observed rewards* paragraph.

**Required changes:**

- [ ] Rerun the STE calibration under the new coding (STE is defined on the
      AYA reward level, so `te_scale` moves).
- [ ] Rerun the m sweep (E18) and the Appendix B.11 design-choice arms with
      the AYA learner on the new reward and its own noise scale (coin-flip
      reward variance 0.64, `Simulation_Testbed/results/e23/sigma_emp_aya_coding.json`).
- [ ] `main_new.tex` — update the m-sweep paragraph, the appendix results and
      figures; then drop the "earlier binary coding" sentence and point the
      STE definition, the *Evaluation* paragraph and the m-sweep paragraph
      back to \eqref{eq:reward-aya}.

**Acceptance.** No result in `main_new.tex` rests on the (3, 0) coding, and
the doc no longer mentions it.

---

## 8. Normalize every reward to [0, 1] — DONE

**Why.** On 2026-10-15 the team adopted S. Murphy's suggestion (comment on the
reward ranges, `main_new.tex` §5.3) to put all three rewards on [0, 1]: each
reward is divided by its range maximum (every range starts at 0), AYA by 2, CP
by 5, REL by 5 + 2·w_adh. The testbed already does this
(`Simulation_Testbed/sim/env.py`, `TrialConfig.normalize_rewards`, on by
default). The rescaling is a change of units for the local fit, σ̂², τ²₀ and
the EB step, which all move with the reward. The allocation slope b_ρ is the
exception: it acts on the contrast in reward units, so the same b_ρ = 20 gives
a flatter allocation for any agent whose range exceeds 1. On one testbed trial,
normalizing with each agent's b_ρ multiplied by its r_max reproduced every
action of the native-scale run, while normalizing with b_ρ = 20 moved π by
0.02 (AYA), 0.03 (CP) and 0.05 (REL) on average.

Implemented in `app/learner/rewards.py` and `app/hyperparams.py`. The design
learner stores normalized rewards in `study_data.reward`, retains native values
in `study_data.outcome["reward_native"]`, uses the fixed normalized noise
scales, and applies the per-agent slope `b_rho * r_max`. Tests cover each
agent's minimum and maximum reward and the derived hyperparameters.
