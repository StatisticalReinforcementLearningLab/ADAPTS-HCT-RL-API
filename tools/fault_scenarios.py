"""
The Monitoring-Spec §11.3 fault table as data, for ``tools/live_trial.py``.

Each :class:`FaultScenario` says how one fault is injected and which alarm
``item_id`` values must then appear in ``monitor_alarms`` (Monitoring-Spec
§9.1). Injection kinds (implementation contract D21, §4):

- host-side simulator options (:class:`HostFaultSpec`, resolved against the
  trial plan into ``tests.simulate_adapts_hct.HostFaults``);
- server environment ``FAULT_*`` variables, for the whole run
  (``server_env``) or for a logical-time window (``server_env_window``: the
  harness restarts the server with and then without them);
- a ``hyperparameters.yaml`` override for the whole run (``hyperparams``,
  e.g. ``numerics.k_max: 1``);
- a database tamper callback after a decision (``db_tamper``);
- a server restart with an edited settings file after freezing, which must
  be refused (``ServerAction(kind="restart_with_yaml")``);
- killing the server while an ``/update`` runs (``ServerAction(kind="kill_during_update")``);
- stopping the monitoring ticks for one logical hour (``monitor_pause``).

Times are :class:`When` values relative to the plan's first calendar Monday
(calendar week ``c``, weekday, clock). With the default plan (10 dyads
enrolled one a week from 2026-11-02, 16 calendar weeks) learning starts in
calendar week 6 (the fifth dyad completes week 1 in calendar week 5,
contract D2), so faults are placed in calendar weeks 8-10.

Clean-baseline expectation (Monitoring-Spec §11.3 item 3): no red or yellow
alarm except C1, C10 and the L-checks. Some yellow checks can also fire by
chance under correct behaviour and are listed separately in
:data:`CHANCE_ITEMS` (reported as warnings, not failures); see the note there.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from tests.simulate_adapts_hct import (
    HostFaults,
    TrialPlan,
    at,
    study_week,
    week_monday,
)

# Red/yellow items the clean baseline may raise (Monitoring-Spec §11.3 item 3).
BASELINE_ALLOWED_ITEMS = ("C1", "C10", "L1", "L2", "L3", "L4")

# Yellow items that fire with non-negligible probability under a correct API
# and host, so a clean run may show them; reported as warnings.
#   A6/A7: delivered share of a completed week > 80% / < 20%. Under
#     Bernoulli(1/2) warm-up a CP week (7 opportunities) has P(>= 6 sends) =
#     P(<= 1 send) = 8/128 = 0.0625; learned pi in [0.2, 0.8] makes it larger.
#   A17: three consecutive weekly dyad_game actions all equal has probability
#     1/4 per dyad-week under warm-up.
#   A5: the whole cohort at an allocation limit for 7 days (possible once the
#     posteriors concentrate).
CHANCE_ITEMS = ("A5", "A6", "A7", "A17")

SEVERITIES_THAT_FAIL = ("red", "yellow")


@dataclass(frozen=True)
class When:
    """A logical time: weekday (Monday = 0) and clock of calendar week ``week`` of the plan."""

    week: int
    weekday: int = 0
    clock: dt.time = dt.time(0, 0)

    def resolve(self, plan: TrialPlan) -> dt.datetime:
        day = plan.start_monday + dt.timedelta(days=7 * self.week + self.weekday)
        return at(day, self.clock)

    def to_json(self) -> dict:
        return {"week": self.week, "weekday": self.weekday, "clock": self.clock.isoformat()}


@dataclass(frozen=True)
class HostFaultSpec:
    """
    A host-side fault, resolved against the plan. ``dyad`` is 1-based
    (``dyad_001``). Kinds: ``skip_action`` / ``bad_diary`` (``decision_type`` +
    ``when``: the decision at that time), ``withhold_closing`` /
    ``skip_update`` (``when.week``: that Monday), ``delay_update`` /
    ``resend_update`` (``when``: Monday and new clock), ``stuck_game``
    (``study_week``: first study week to consider).
    """

    kind: str
    dyad: int | None = None
    decision_type: str | None = None
    when: When | None = None
    study_week: int | None = None


@dataclass(frozen=True)
class ServerEnvWindow:
    """Run the server with ``env`` from ``start`` until ``end`` (two restarts)."""

    env: dict
    start: When
    end: When


@dataclass(frozen=True)
class ServerAction:
    """
    ``restart_with_yaml``: at ``when`` stop the server, start it with the
    settings file changed by ``yaml_overrides`` (must be refused when
    ``expect_refusal``), then restore the file and start it again.
    ``kill_during_update``: before the Monday ``when.week`` update, restart
    with ``FAULT_UPDATE_HANG_SECONDS = hang_seconds``; right after that
    ``/update`` is accepted, kill the server and restart it normally.
    """

    kind: str
    when: When
    yaml_overrides: dict | None = None
    expect_refusal: bool = True
    hang_seconds: float = 60.0


@dataclass(frozen=True)
class DbTamper:
    """After the first learned decision at or after ``after``, apply tamper ``name`` (see ``tools/live_trial.py``)."""

    name: str
    after: When
    delta: float = 0.25


@dataclass(frozen=True)
class MonitorPause:
    """No monitoring ticks in ``[start, start + duration)`` (the monitoring program is stopped)."""

    start: When
    duration: dt.timedelta = dt.timedelta(hours=1)


@dataclass(frozen=True)
class ExtraDyad:
    """An additional dyad consenting on ``when`` (its date)."""

    when: When


@dataclass(frozen=True)
class FaultScenario:
    name: str
    spec_row: str
    description: str
    expected_items: tuple[str, ...] = ()
    allowed_items: tuple[str, ...] = ()
    host: tuple[HostFaultSpec, ...] = ()
    server_env: dict = field(default_factory=dict)
    server_env_window: ServerEnvWindow | None = None
    hyperparams: dict | None = None
    db_tamper: DbTamper | None = None
    server_actions: tuple[ServerAction, ...] = ()
    monitor_pause: MonitorPause | None = None
    extra_dyads: tuple[ExtraDyad, ...] = ()
    allowed_statuses: tuple[tuple[str, int], ...] = ()
    checks: tuple[str, ...] = ()
    min_weeks: int = 12  # faults sit in calendar weeks 8-10 and their checks run up to two weeks later

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "spec_row": self.spec_row,
            "description": self.description,
            "expected_items": list(self.expected_items),
            "allowed_items": list(self.allowed_items),
            "host": [vars(h) | {"when": h.when.to_json() if h.when else None} for h in self.host],
            "server_env": dict(self.server_env),
            "server_env_window": (
                {
                    "env": dict(self.server_env_window.env),
                    "start": self.server_env_window.start.to_json(),
                    "end": self.server_env_window.end.to_json(),
                }
                if self.server_env_window
                else None
            ),
            "hyperparams": self.hyperparams,
            "db_tamper": vars(self.db_tamper) | {"after": self.db_tamper.after.to_json()} if self.db_tamper else None,
            "server_actions": [vars(a) | {"when": a.when.to_json()} for a in self.server_actions],
            "monitor_pause": (
                {"start": self.monitor_pause.start.to_json(), "minutes": self.monitor_pause.duration.total_seconds() / 60}
                if self.monitor_pause
                else None
            ),
            "extra_dyads": [d.when.to_json() for d in self.extra_dyads],
            "allowed_statuses": [list(s) for s in self.allowed_statuses],
            "checks": list(self.checks),
        }


# Fault placement (default plan): learning from calendar week 6.
_WED_W9_AM = When(9, 2, dt.time(7, 30))
_W9_WED_0000 = When(9, 2, dt.time(0, 0))
_W9_THU_0000 = When(9, 3, dt.time(0, 0))

SCENARIOS: dict[str, FaultScenario] = {
    s.name: s
    for s in (
        FaultScenario(
            name="baseline",
            spec_row="(no fault) — no red or yellow alarms except C1, C10 and the L-checks",
            description="Clean run: staggered any-weekday enrollment, the full API-Spec §3 calendar.",
            checks=("no_unexplained_alarms",),
            min_weeks=1,  # no fault to place: any trial length works (e.g. a short 8-week smoke run)
        ),
        FaultScenario(
            name="skip_action",
            spec_row="host skips one /action -> A2",
            description="dyad_001 misses its Wednesday AYA-AM upload + /action pair in calendar week 9 (host flips its own coin, F-A1).",
            expected_items=("A2",),
            allowed_items=("A16",),  # the host's own coin may deliver a message with no API action
            host=(HostFaultSpec("skip_action", dyad=1, decision_type="aya_message", when=_WED_W9_AM),),
        ),
        FaultScenario(
            name="action_crash",
            spec_row="the algorithm crashes inside /action -> AC-201-FALLBACK (F-A2 used)",
            description="FAULT_ACTION_CRASH=aya_message for one logical day (Wednesday of calendar week 9).",
            expected_items=("AC-201-FALLBACK",),
            server_env_window=ServerEnvWindow({"FAULT_ACTION_CRASH": "aya_message"}, _W9_WED_0000, _W9_THU_0000),
            checks=("fallback_reason:learner_error",),
        ),
        FaultScenario(
            name="force_pi",
            spec_row="pi forced to 0.95 by a modified algorithm -> A13 (the 50/50 fallback is used instead)",
            description="FAULT_FORCE_PI=0.95 for one logical day (Wednesday of calendar week 9).",
            expected_items=("A13",),
            allowed_items=("AC-201-FALLBACK",),
            server_env_window=ServerEnvWindow({"FAULT_FORCE_PI": "0.95"}, _W9_WED_0000, _W9_THU_0000),
            checks=("fallback_reason:invalid_probability",),
        ),
        FaultScenario(
            name="sampler_shift",
            spec_row="the random-number source shifted by one -> A4",
            description="FAULT_SAMPLER_SHIFT=1 for one logical day (Wednesday of calendar week 9).",
            expected_items=("A4",),
            server_env_window=ServerEnvWindow({"FAULT_SAMPLER_SHIFT": "1"}, _W9_WED_0000, _W9_THU_0000),
        ),
        FaultScenario(
            name="posterior_tamper",
            spec_row="a saved posterior changed after the decision -> A4",
            description="After the first learned decision on or after Monday of calendar week 9, add 0.25 to theta[0] of the posterior row it used.",
            expected_items=("A4",),
            allowed_items=("C5",),  # the week's rerun no longer matches the (tampered) stored posterior
            db_tamper=DbTamper("shift_posterior", When(9, 0, dt.time(0, 0)), 0.25),
        ),
        FaultScenario(
            name="withhold_closing",
            spec_row="one dyad's week-closing upload withheld -> B11",
            description="dyad_001 sends no week-closing upload on Monday of calendar week 10.",
            expected_items=("B11",),
            host=(HostFaultSpec("withhold_closing", dyad=1, when=When(10)),),
        ),
        FaultScenario(
            name="bad_diary_upload",
            spec_row="upload with a diary item of 7 -> UD-400-RANGE, F-U1",
            description="dyad_002's Wednesday AYA-AM upload in calendar week 9 is first sent with aya_diary_mood = 7 (400), then corrected.",
            expected_items=("UD-400-RANGE",),
            host=(HostFaultSpec("bad_diary", dyad=2, decision_type="aya_message", when=_WED_W9_AM),),
            allowed_statuses=(("upload_data", 400),),
            checks=("quarantine_rows",),
        ),
        FaultScenario(
            name="update_killed",
            spec_row="the update killed while running -> update marked failed after 30 min; C7 at 05:45",
            description="Monday of calendar week 10: the server is killed right after accepting /update; the host does not resend.",
            expected_items=("C7",),
            allowed_items=("C1", "B5", "AC-201-FALLBACK"),  # a dyad first eligible that week has no earlier policy: F-A3 (B5, AC-201-FALLBACK)
            server_actions=(ServerAction("kill_during_update", When(10, 0, dt.time(2, 50)), hang_seconds=60.0),),
            checks=("update_timed_out",),
        ),
        FaultScenario(
            name="update_killed_resend",
            spec_row="the update killed while running -> marked failed after 30 min; no C7 because the host sends it again",
            description="As update_killed, and the host resends /update at 04:00.",
            # B5: when the killed run is marked failed, a dyad first eligible that week has no earlier
            # policy (F-A3) until the 04:00 resend publishes one.
            allowed_items=("C1", "B5"),
            host=(HostFaultSpec("resend_update", when=When(10, 0, dt.time(4, 0))),),
            server_actions=(ServerAction("kill_during_update", When(10, 0, dt.time(2, 50)), hang_seconds=60.0),),
            checks=("update_timed_out", "no_item:C7"),
        ),
        FaultScenario(
            name="skip_update",
            spec_row="host skips the Monday /update -> C11, then C7",
            description="No /update on Monday of calendar week 10.",
            expected_items=("C11", "C7"),
            allowed_items=("B5", "AC-201-FALLBACK"),  # a dyad first eligible that week has no earlier policy: F-A3 (B5, AC-201-FALLBACK)
            host=(HostFaultSpec("skip_update", when=When(10)),),
        ),
        FaultScenario(
            name="delay_update",
            spec_row="update delayed past 05:45 -> C7",
            description="Monday of calendar week 10: /update sent at 06:00 instead of 03:00 (C11 also fires at 03:30).",
            expected_items=("C7",),
            allowed_items=("C11", "B5", "AC-201-FALLBACK"),  # a dyad first eligible that week has no earlier policy: F-A3 (B5, AC-201-FALLBACK)
            host=(HostFaultSpec("delay_update", when=When(10, 0, dt.time(6, 0))),),
        ),
        FaultScenario(
            name="eb_kmax_1",
            spec_row="empirical-Bayes optimizer limited to 1 iteration -> C1, F-U5",
            description="hyperparameters.yaml sets numerics.k_max: 1 for the whole run.",
            expected_items=("C1",),
            hyperparams={"numerics": {"k_max": 1}},
        ),
        FaultScenario(
            name="settings_edited",
            spec_row="settings file edited after freezing -> C9 (the API refuses to start)",
            description="Monday of calendar week 8, 01:00 (after the first learning update froze the set): restart with penalty.m: 3; the server must refuse; then the file is restored.",
            expected_items=("C9",),
            server_actions=(
                ServerAction("restart_with_yaml", When(8, 0, dt.time(1, 0)), yaml_overrides={"penalty": {"m": 3.0}}),
            ),
            checks=("restart_refused",),
        ),
        FaultScenario(
            name="game_stuck",
            spec_row="current_game_on stuck at last week's value -> A15",
            description="dyad_001 reports last week's current_game_on during the first study week >= 8 whose game action changed.",
            expected_items=("A15",),
            host=(HostFaultSpec("stuck_game", dyad=1, study_week=8),),
        ),
        FaultScenario(
            name="monitoring_stopped",
            spec_row="monitoring program stopped for 1 hour -> E5 on restart; no G0 if stopped over 07:00",
            description="No monitoring ticks from Wednesday 06:30 to 07:30 of calendar week 9.",
            expected_items=("E5",),
            monitor_pause=MonitorPause(When(9, 2, dt.time(6, 30)), dt.timedelta(hours=1)),
        ),
        FaultScenario(
            name="tuesday_start",
            spec_row="a dyad registered with a Tuesday start -> no alarm; the dyad begins in a partial week",
            description="An extra dyad consents on Tuesday of calendar week 8 (default enrollment already has Tuesday starts).",
            extra_dyads=(ExtraDyad(When(8, 1)),),
            checks=("no_unexplained_alarms", "extra_dyad_partial_week"),
        ),
    )
}


# ------------------------------------------------------------- resolution


def plan_with_extra_dyads(plan: TrialPlan, scenario: FaultScenario) -> TrialPlan:
    """The plan with the scenario's extra dyads appended to the enrollment."""
    if not scenario.extra_dyads:
        return plan
    starts = list(plan.consent_starts()) + [d.when.resolve(plan).date() for d in scenario.extra_dyads]
    return TrialPlan(**{**vars(plan), "enrollment": tuple(starts), "num_dyads": len(starts)})


def decision_idx_at(consent_start: dt.date, decision_type: str, when: dt.datetime) -> int:
    """The host's ``decision_idx`` of the decision at ``when`` (simulator counters, API-Spec §3)."""
    days = (when.date() - consent_start).days
    if days < 0:
        raise ValueError(f"{when} is before consent start {consent_start}")
    if decision_type == "aya_message":
        return 2 * days + (1 if when.hour < 12 else 2)
    if decision_type == "cp_message":
        return days + 1
    if decision_type == "dyad_game":
        return study_week(consent_start, when.date())
    raise ValueError(f"unknown decision_type {decision_type!r}")


def resolve_host_faults(scenario: FaultScenario, plan: TrialPlan) -> HostFaults:
    """Concrete :class:`HostFaults` of ``scenario`` for ``plan``."""
    specs = {s.index + 1: s for s in plan.dyad_specs()}
    faults = HostFaults(allowed_statuses=set(scenario.allowed_statuses))
    for spec in scenario.host:
        dyad = specs.get(spec.dyad) if spec.dyad is not None else None
        if spec.dyad is not None and dyad is None:
            raise ValueError(f"scenario {scenario.name}: the plan has no dyad {spec.dyad}")
        if spec.kind in ("skip_action", "bad_diary"):
            when = spec.when.resolve(plan)
            key = (dyad.group_id, spec.decision_type, decision_idx_at(dyad.consent_start, spec.decision_type, when))
            (faults.skip_actions if spec.kind == "skip_action" else faults.bad_diary_uploads).add(key)
        elif spec.kind == "withhold_closing":
            faults.withhold_closing.add((dyad.group_id, week_monday(spec.when.resolve(plan).date())))
        elif spec.kind == "skip_update":
            faults.skip_updates.add(week_monday(spec.when.resolve(plan).date()))
        elif spec.kind in ("delay_update", "resend_update"):
            when = spec.when.resolve(plan)
            target = faults.delay_updates if spec.kind == "delay_update" else faults.resend_updates
            target[week_monday(when.date())] = when.time()
        elif spec.kind == "stuck_game":
            faults.stuck_game[dyad.group_id] = int(spec.study_week)
        else:
            raise ValueError(f"unknown host fault kind {spec.kind!r}")
    return faults


def merge_yaml(base: dict | None, overrides: dict | None) -> dict:
    """Nested dict merge (``overrides`` win)."""
    out: dict[str, Any] = dict(base or {})
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_yaml(out[key], value)
        else:
            out[key] = value
    return out


def judge_alarms(
    scenario: FaultScenario, alarm_items: dict[str, dict[str, int]]
) -> dict[str, list[str]]:
    """
    Compare raised alarms (``{item_id: {severity: n}}``) with the scenario:
    ``missing`` expected items, ``unexplained`` red/yellow items outside the
    baseline allowance, the scenario's own items and the chance items, and
    ``chance`` items that appeared.
    """
    present = set(alarm_items)
    missing = [item for item in scenario.expected_items if item not in present]
    explained = set(BASELINE_ALLOWED_ITEMS) | set(scenario.expected_items) | set(scenario.allowed_items)
    unexplained, chance = [], []
    for item, severities in sorted(alarm_items.items()):
        if not any(severities.get(s) for s in SEVERITIES_THAT_FAIL):
            continue
        if item in explained:
            continue
        (chance if item in CHANCE_ITEMS else unexplained).append(item)
    return {"missing": missing, "unexplained": unexplained, "chance": chance}
