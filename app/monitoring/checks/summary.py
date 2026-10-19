"""
G0, the daily status (Monitoring-Spec §5.G), and the fallback / error-code
counts it shares with ``flask monitor report`` (§8.3).

G0 is raised every day at 07:00, whatever else happened: how many times
monitoring ran in the last 24 h; the previous day's decisions against the
calendar per agent with the share of sends and of coin-flip fallbacks; the
state of this week's update and how long it took; how often each fallback
(§7) was used and each error code (§6) returned, per agent; the EB fallback
rate so far (C1); and the red and yellow alarms raised in the last 24 h. If
it is missing, monitoring is down.
"""

from __future__ import annotations

import collections
import datetime

from app.extensions import db
from app.monitoring.checks.common import (
    AGENT_ORDER,
    CheckContext,
    OPERATIONAL_FALLBACKS,
    actions_between,
    actions_on_day,
    consents_at_logical,
    expected_calls,
)
from app.monitoring.findings import Finding, dedup_key
from app.study_calendar import week_monday


def decisions_against_calendar(day: datetime.date) -> dict[str, dict]:
    """Per agent for ``day``: expected calls, made calls, share of sends, of coin-flip fallbacks and of warm-up."""
    consents = consents_at_logical(datetime.datetime.combine(day, datetime.time.max))
    by_type: dict[str, list] = collections.defaultdict(list)
    for action in actions_on_day(day):
        by_type[action.decision_type].append(action)
    out = {}
    for dt in AGENT_ORDER:
        rows = by_type.get(dt) or []
        expected = sum(expected_calls(dt, c, day) for c in consents.values())
        n = len(rows)
        out[dt] = {
            "expected": expected,
            "made": n,
            "missing": max(expected - n, 0),
            "share_sends": (sum(int(a.action) for a in rows) / n) if n else None,
            "share_fallback": (sum(1 for a in rows if a.warmup_reason in OPERATIONAL_FALLBACKS) / n) if n else None,
            "share_warmup": (sum(1 for a in rows if a.is_warmup) / n) if n else None,
        }
    return out


def fallback_counts(start: datetime.datetime, end: datetime.datetime) -> dict:
    """
    Uses of each fallback of Monitoring-Spec §7 in ``(start, end]`` (study
    clock), per agent where it applies.
    """
    from app.models import ApiErrorEvent, ModelParameters, ModelUpdateRequests, PolicyVersion

    out: dict = {k: collections.Counter() for k in ("F-A1", "F-A2", "F-A3", "F-U1", "F-U2", "F-U3", "F-U4", "F-U5")}
    for action in actions_between(start, end):
        if action.warmup_reason in ("no_state", "learner_error", "invalid_probability"):
            out["F-A2"][action.decision_type] += 1
        elif action.warmup_reason == "no_policy":
            out["F-A3"][action.decision_type] += 1
    d = (start + datetime.timedelta(seconds=1)).date()
    while datetime.datetime.combine(d, datetime.time.min) < end:
        day_end = datetime.datetime.combine(d, datetime.time.max)
        if day_end <= end:
            for dt, row in decisions_against_calendar(d).items():
                if row["missing"]:
                    out["F-A1"][dt] += row["missing"]
        d += datetime.timedelta(days=1)
    for (code,) in db.session.query(ApiErrorEvent.code).filter(
        ApiErrorEvent.occurred_at > start, ApiErrorEvent.occurred_at <= end, ApiErrorEvent.code.like("UD-400-%")
    ):
        out["F-U1"]["upload"] += 1
    reqs = ModelUpdateRequests.query.filter(
        ModelUpdateRequests.request_timestamp > start, ModelUpdateRequests.request_timestamp <= end
    ).all()
    rids = [r.rid for r in reqs]
    for req in reqs:
        if req.status == "failed" and (req.error_message or "") == "timeout":
            out["F-U4"]["update"] += 1
        if req.status == "completed" and PolicyVersion.query.filter_by(update_rid=req.rid).count() == 0:
            out["F-U3"]["update"] += 1
    if rids:
        for v in PolicyVersion.query.filter(PolicyVersion.update_rid.in_(rids)).all():
            if v.status in ("rejected", "failed"):
                out["F-U2"][v.decision_type] += 1
        for row in ModelParameters.query.filter(
            ModelParameters.update_rid.in_(rids), ModelParameters.snapshot_type == "hyper"
        ).all():
            meta = row.metadata_json or {}
            if meta.get("fallback_used") or meta.get("converged") is False:
                out["F-U5"][row.decision_type] += 1
    return {k: dict(v) for k, v in out.items()}


def code_counts(start: datetime.datetime, end: datetime.datetime) -> dict[str, dict]:
    """``{code: {decision_type or "all": n}}`` of the error log in ``(start, end]``."""
    from app.models import ApiErrorEvent

    out: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for code, dt in db.session.query(ApiErrorEvent.code, ApiErrorEvent.decision_type).filter(
        ApiErrorEvent.occurred_at > start, ApiErrorEvent.occurred_at <= end
    ):
        out[code][dt or "all"] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


def eb_fallback_rate(until: datetime.datetime | None = None) -> dict[str, dict]:
    """Per agent: EB hyper-updates so far and how many fell back (F-U5), from the ``hyper`` rows."""
    from app.models import ModelParameters

    out: dict[str, dict] = {}
    q = ModelParameters.query.filter(ModelParameters.snapshot_type == "hyper")
    for row in q.all():
        entry = out.setdefault(row.decision_type, {"updates": 0, "fallbacks": 0, "rate": None})
        entry["updates"] += 1
        meta = row.metadata_json or {}
        if meta.get("fallback_used") or meta.get("converged") is False:
            entry["fallbacks"] += 1
    for entry in out.values():
        entry["rate"] = entry["fallbacks"] / entry["updates"] if entry["updates"] else None
    return out


def update_week_state(monday: datetime.date) -> dict:
    """This week's update: ``update_weeks`` state and the duration of each call."""
    from app.models import UpdateWeek
    from app.monitoring.checks.common import updates_of_week

    row = UpdateWeek.query.filter_by(study_week_monday=monday).first()
    calls = []
    for req in updates_of_week(monday):
        duration = None
        if req.published_at_study is not None:
            duration = (req.published_at_study - req.request_timestamp).total_seconds()
        elif req.completed_at is not None and req.started_at is not None:
            duration = (req.completed_at - req.started_at).total_seconds()
        calls.append({"rid": req.rid, "status": req.status, "request_timestamp": req.request_timestamp,
                      "duration_seconds": duration, "error_message": "timeout" if req.error_message == "timeout"
                      else (None if req.error_message is None else "error")})
    return {
        "study_week_monday": monday,
        "status": row.status if row is not None else None,
        "ready_at": row.ready_at if row is not None else None,
        "c7_raised": bool(row.c7_raised) if row is not None else False,
        "calls": calls,
    }


def alarm_counts(start: datetime.datetime, end: datetime.datetime, exclude=("G0",)) -> dict[str, int]:
    from app.models import MonitorAlarm

    out = collections.Counter()
    for severity, item in db.session.query(MonitorAlarm.severity, MonitorAlarm.item_id).filter(
        MonitorAlarm.raised_at > start, MonitorAlarm.raised_at <= end
    ):
        if item not in exclude:
            out[severity] += 1
    return {s: out.get(s, 0) for s in ("red", "yellow", "green")}


def status_content(now: datetime.datetime) -> dict:
    """The G0 content at ``now`` (also printed by ``flask monitor status``)."""
    from app.models import MonitorRun

    start = now - datetime.timedelta(hours=24)
    runs = MonitorRun.query.filter(MonitorRun.logical_now > start, MonitorRun.logical_now <= now).count()
    yesterday = (now - datetime.timedelta(days=1)).date()
    return {
        "at": now,
        "monitoring_runs_24h": runs,
        "decisions_day": yesterday,
        "decisions": decisions_against_calendar(yesterday),
        "update": update_week_state(week_monday(now)),
        "fallbacks_24h": fallback_counts(start, now),
        "error_codes_24h": code_counts(start, now),
        "eb_fallback_rate": eb_fallback_rate(),
        "alarms_24h": alarm_counts(start, now),
    }


def check_g0(ctx: CheckContext) -> list[Finding]:
    """G0 (daily 07:00): one green status alarm every day."""
    content = status_content(ctx.now)
    alarms = content["alarms_24h"]
    update = content["update"]
    return [Finding(
        "G0",
        f"Daily status {ctx.now.date().isoformat()}: monitoring ran {content['monitoring_runs_24h']} times in the "
        f"last 24 h; {alarms['red']} red and {alarms['yellow']} yellow alarms; this week's update: "
        f"{update['status'] or 'none'}.",
        dedup_key("G0", ctx.now.date()),
        at=ctx.now,
        detail=content,
    )]


__all__ = [
    "decisions_against_calendar",
    "fallback_counts",
    "code_counts",
    "eb_fallback_rate",
    "update_week_state",
    "alarm_counts",
    "status_content",
    "check_g0",
]
