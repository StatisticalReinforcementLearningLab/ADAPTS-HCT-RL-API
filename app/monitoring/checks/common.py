"""
Shared pieces of the monitoring checks (Monitoring-Spec §2.2, §3; implementation
contract D2, D5, D20).

Every check is ``check(ctx) -> list[Finding]``: a pure function of the database
and of ``ctx`` (the logical time it runs at and the window it examines). It
uses host-logical timestamps (``request_timestamp``) for study-calendar logic
and reads a dyad's consent dates as they were at the time being checked from
``group_history`` (§9.6), never from ``groups``.
"""

from __future__ import annotations

import dataclasses
import datetime
from collections.abc import Iterable, Mapping
from typing import Any

from app.monitoring.findings import Finding  # noqa: F401 - re-exported for the check modules
from app.study_calendar import study_week, week1_start, week_monday

DECISION_TYPES = ("aya_message", "cp_message", "dyad_game")
AGENT_ORDER = ("dyad_game", "aya_message", "cp_message")
OPERATIONAL_FALLBACKS = ("no_state", "no_policy", "learner_error", "invalid_probability")
SCHEDULED_WARMUP = ("cohort", "week1")
MISS = "miss"


@dataclasses.dataclass
class CheckContext:
    """The logical time a check runs at and the window it examines (``(window_start, window_end]``)."""

    now: datetime.datetime
    window_start: datetime.datetime
    window_end: datetime.datetime
    run_id: int | None = None
    kind: str = "regular"

    @property
    def config(self) -> Mapping[str, Any]:
        from flask import current_app

        return current_app.config

    @property
    def hp(self):
        from app.hyperparams import get_hyperparams

        return get_hyperparams()


# ------------------------------------------------------------------ consents


@dataclasses.dataclass(frozen=True)
class Consent:
    group_id: str
    start: datetime.date
    end: datetime.date

    @property
    def week1(self) -> datetime.date:
        return week1_start(self.start)


def _as_date(value) -> datetime.date | None:
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    try:
        return datetime.date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


class ConsentTimeline:
    """
    Every registration in recording order (``group_history``; Monitoring-Spec
    §9.6), loaded once and viewed as of any server time. Dyads present in
    ``groups`` with no history row at all (registered before
    ``group_history`` existed) enter at their ``groups.created_at``.
    """

    def __init__(self, events: list[tuple[datetime.datetime | None, int, Consent]]):
        self._events = sorted(events, key=lambda e: (e[0] or datetime.datetime.min, e[1]))

    @classmethod
    def load(cls, group_id: str | None = None) -> "ConsentTimeline":
        from app.models import Group, GroupHistory

        events: list[tuple[datetime.datetime | None, int, Consent]] = []
        q = GroupHistory.query
        if group_id is not None:
            q = q.filter(GroupHistory.group_id == group_id)
        with_history: set[str] = set()
        for row in q.all():
            with_history.add(row.group_id)
            start, end = _as_date(row.consent_start_date), _as_date(row.consent_end_date)
            if start is not None and end is not None:
                events.append((row.recorded_at, row.id, Consent(row.group_id, start, end)))
        gq = Group.query
        if group_id is not None:
            gq = gq.filter(Group.group_id == group_id)
        for group in gq.all():
            if group.group_id in with_history:
                continue
            info = group.group_info or {}
            start, end = _as_date(info.get("consent_start_date")), _as_date(info.get("consent_end_date"))
            if start is not None and end is not None:
                # Negative ids keep these after history rows recorded at the same time.
                events.append((group.created_at, -group.id, Consent(group.group_id, start, end)))
        return cls(events)

    def as_of(self, server_time: datetime.datetime | None) -> dict[str, Consent]:
        """Latest consent dates per dyad recorded at or before ``server_time`` (``None``: no cutoff)."""
        out: dict[str, Consent] = {}
        for recorded_at, _, consent in self._events:
            if server_time is not None and recorded_at is not None and recorded_at > server_time:
                break
            out[consent.group_id] = consent
        return out


def consents_as_of(server_time: datetime.datetime | None) -> dict[str, Consent]:
    """
    Consent dates per dyad as recorded by ``server_time`` (server clock):
    the latest ``group_history`` row per dyad with ``recorded_at <=
    server_time`` (``None``: no cutoff); see :class:`ConsentTimeline`.
    """
    return ConsentTimeline.load().as_of(server_time)


def consent_of(group_id: str, server_time: datetime.datetime | None) -> Consent | None:
    """One dyad's consent dates as recorded by ``server_time``."""
    return ConsentTimeline.load(group_id).as_of(server_time).get(group_id)


def consents_at_logical(logical: datetime.datetime | None) -> dict[str, Consent]:
    """:func:`consents_as_of` at the server time matching study time ``logical``."""
    from app.monitoring.timeutil import server_time_for

    return consents_as_of(server_time_for(logical))


def start_dates(consents: Mapping[str, Consent]) -> dict[str, datetime.date]:
    """``{group_id: consent_start}`` for the study-calendar helpers (contract D5)."""
    return {gid: c.start for gid, c in consents.items()}


# ------------------------------------------------------------------ calendar


def trial_weeks() -> int:
    from app.hyperparams import get_hyperparams

    return int(get_hyperparams().trial_weeks)


def active_on(consent: Consent, day: datetime.date, T: int | None = None) -> bool:
    """
    Monitoring-Spec §3 "active dyad on day d": d within the consent dates and
    the dyad in its partial week or in study weeks ``1..T`` that week.
    """
    if T is None:
        T = trial_weeks()
    if not (consent.start <= day <= consent.end):
        return False
    return study_week(consent.start, day) <= int(T)


def active_in_week(consent: Consent, monday: datetime.date, T: int | None = None) -> bool:
    """Active on at least one day of the calendar week starting ``monday``."""
    return any(active_on(consent, monday + datetime.timedelta(days=k), T) for k in range(7))


def calendar() -> Mapping[str, Mapping]:
    from flask import current_app

    return current_app.config.get("MONITOR_DECISION_CALENDAR") or {}


def expected_calls(decision_type: str, consent: Consent, day: datetime.date, T: int | None = None) -> int:
    """
    Expected ``/action`` calls of one agent for one dyad on ``day``
    (``MONITOR_DECISION_CALENDAR``; Monitoring-Spec §3 "Decision calendar"):
    AYA 2 and CP 1 every day incl. the partial week; REL 1 on Mondays from week 1.
    """
    spec = calendar().get(decision_type) or {}
    if not active_on(consent, day, T):
        return 0
    if day.weekday() not in set(spec.get("weekdays") or []):
        return 0
    if study_week(consent.start, day) == 0 and not spec.get("partial_week", False):
        return 0
    return int(spec.get("per_day", 0))


def weekly_opportunities(decision_type: str) -> int:
    """``n_week^g`` of Monitoring-Spec §3: per-day calls times scheduled weekdays (AYA 14, CP 7, REL 1)."""
    spec = calendar().get(decision_type) or {}
    return int(spec.get("per_day", 0)) * len(set(spec.get("weekdays") or []))


def day_bounds(day: datetime.date) -> tuple[datetime.datetime, datetime.datetime]:
    start = datetime.datetime.combine(day, datetime.time.min)
    return start, start + datetime.timedelta(days=1)


def deadline_spec(name: str, default: str) -> str:
    from flask import current_app

    return str(current_app.config.get(name) or default)


# ------------------------------------------------------------------- updates


def updates_of_week(monday: datetime.date, *, until: datetime.datetime | None = None) -> list:
    """``model_update_requests`` of calendar week ``monday`` in arrival order (``request_timestamp``, ``id``)."""
    from app.models import ModelUpdateRequests

    start = datetime.datetime.combine(monday, datetime.time.min)
    end = start + datetime.timedelta(days=7)
    q = ModelUpdateRequests.query.filter(
        ModelUpdateRequests.request_timestamp >= start, ModelUpdateRequests.request_timestamp < end
    )
    if until is not None:
        q = q.filter(ModelUpdateRequests.request_timestamp <= until)
    return q.order_by(ModelUpdateRequests.request_timestamp.asc(), ModelUpdateRequests.id.asc()).all()


def week_closing_close(monday: datetime.date) -> datetime.datetime:
    """
    End of the week-closing window of week ``monday`` (API-Spec §2.3: after
    Sunday midnight, before the Monday update): the first ``/update`` of the
    week if it arrived by the publication deadline, else Monday 03:00.
    """
    from app.study_calendar import weekly_deadline

    deadline = weekly_deadline(monday, deadline_spec("MONITOR_WEEK_READY_BY", "MON 05:45"))
    first = updates_of_week(monday, until=deadline)
    if first:
        return first[0].request_timestamp
    return datetime.datetime.combine(monday, datetime.time(3, 0))


def update_event_time(req) -> datetime.datetime:
    """Logical time of an update's outcome: its publication time, else its request time."""
    return req.published_at_study or req.request_timestamp


def finished_updates(start: datetime.datetime, end: datetime.datetime) -> list:
    """Finished (not ``processing``) update requests stamped in ``[start, end]``."""
    from app.models import ModelUpdateRequests

    return (
        ModelUpdateRequests.query.filter(
            ModelUpdateRequests.request_timestamp >= start,
            ModelUpdateRequests.request_timestamp <= end,
            ModelUpdateRequests.status != "processing",
        )
        .order_by(ModelUpdateRequests.request_timestamp.asc(), ModelUpdateRequests.id.asc())
        .all()
    )


def update_week(req) -> datetime.date:
    return req.study_week_monday or week_monday(req.request_timestamp)


def lookback_start(ctx: CheckContext) -> datetime.datetime:
    from app.monitoring.catalog import UPDATE_LOOKBACK_DAYS

    return min(ctx.window_start, ctx.now - datetime.timedelta(days=UPDATE_LOOKBACK_DAYS))


# ------------------------------------------------------------------- actions


def actions_between(start: datetime.datetime, end: datetime.datetime, *, include_start: bool = False,
                    decision_type: str | None = None) -> list:
    """Actions stamped in ``(start, end]`` (or ``[start, end]``) in decision order."""
    from app.models import Action

    lower = Action.request_timestamp >= start if include_start else Action.request_timestamp > start
    q = Action.query.filter(lower, Action.request_timestamp <= end)
    if decision_type is not None:
        q = q.filter(Action.decision_type == decision_type)
    return q.order_by(Action.request_timestamp.asc(), Action.id.asc()).all()


def actions_on_day(day: datetime.date, decision_type: str | None = None) -> list:
    from app.models import Action

    start, end = day_bounds(day)
    q = Action.query.filter(Action.request_timestamp >= start, Action.request_timestamp < end)
    if decision_type is not None:
        q = q.filter(Action.decision_type == decision_type)
    return q.order_by(Action.request_timestamp.asc(), Action.id.asc()).all()


def design_state_of(action) -> dict | None:
    ctx = action.raw_context if isinstance(action.raw_context, dict) else None
    ds = ctx.get("design_state") if ctx else None
    return ds if isinstance(ds, dict) else None


def is_learned(action) -> bool:
    return not bool(action.is_warmup)


def is_missing(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() == MISS)


def number(value) -> float | None:
    if is_missing(value) or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def iter_dates(start: datetime.date, end: datetime.date) -> Iterable[datetime.date]:
    d = start
    while d <= end:
        yield d
        d += datetime.timedelta(days=1)


__all__ = [
    "CheckContext",
    "Consent",
    "Finding",
    "DECISION_TYPES",
    "AGENT_ORDER",
    "OPERATIONAL_FALLBACKS",
    "SCHEDULED_WARMUP",
    "ConsentTimeline",
    "consents_as_of",
    "consent_of",
    "consents_at_logical",
    "start_dates",
    "active_on",
    "active_in_week",
    "expected_calls",
    "weekly_opportunities",
    "day_bounds",
    "updates_of_week",
    "week_closing_close",
    "update_event_time",
    "finished_updates",
    "update_week",
    "lookback_start",
    "actions_between",
    "actions_on_day",
    "design_state_of",
    "is_learned",
    "is_missing",
    "number",
    "iter_dates",
    "trial_weeks",
    "deadline_spec",
]
