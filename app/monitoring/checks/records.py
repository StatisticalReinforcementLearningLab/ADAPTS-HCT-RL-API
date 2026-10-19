"""
Saved records (Monitoring-Spec §5.D: D1, D3) and the alarms of the error-code
log (Monitoring-Spec §6: "each saved code also becomes an alarm, except
``AC-200-REPLAY``").

Alarm rules per code (``code_finding``):

- ``AC-200-REPLAY``: logged only, no alarm;
- ``AC-201-NOSTATE``: one alarm per dyad, on its first no-state fallback
  (Monitoring-Spec §5.A);
- ``AC-201-FALLBACK``: the reason (``actions.warmup_reason``) goes in
  ``detail``; an ``invalid_probability`` fallback raises A13 instead (§5.A);
- every other code (and ``XX-<status>-UNMAPPED``): one alarm per answer.

The alarm carries coded IDs only: the answer's ``message`` (which may echo a
rejected value) stays in ``api_error_events`` and is not copied (§2.2).
"""

from __future__ import annotations

import collections
import datetime
import json
import os
import re

from app.monitoring import catalog, codes
from app.monitoring.checks.common import CheckContext
from app.monitoring.findings import Finding, capped, dedup_key

_FIELD_RE = re.compile(r"^(?:unknown field '(?P<unknown>[^']*)'|(?P<name>[A-Za-z_][A-Za-z0-9_]*))")


def answer_field(message) -> str | None:
    """The request field a 400 answer names (``unknown field 'x'.`` -> ``x``; ``slot: must be ...`` -> ``slot``)."""
    if not isinstance(message, str):
        return None
    match = _FIELD_RE.match(message)
    if not match:
        return None
    name = match.group("unknown") or match.group("name")
    if name in ("Request", "Invalid", "Internal", "Returning", "The", "Database", "Could", "Monitoring", "Failed",
                "Unknown", "Method", "Action"):
        return None
    return name[:64]


def code_finding(event, *, fallback_reason: str | None = None) -> Finding | None:
    """The alarm of one ``api_error_events`` row (``None`` when the code is not an alarm)."""
    code = event.code
    entry = catalog.lookup(code)
    if not entry.alarm:
        return None
    if code == "AC-201-FALLBACK" and fallback_reason == "invalid_probability":
        return None  # A13 is the alarm for an impossible pi (Monitoring-Spec §5.A)
    key = dedup_key(code, event.event_id)
    fallback = entry.fallback
    detail = {
        "event_id": event.event_id,
        "endpoint": event.endpoint,
        "http_status": event.http_status,
        "answer_title": event.title,
        "rid": event.rid,
    }
    if code == "AC-201-NOSTATE" and event.group_id:
        key = dedup_key(code, event.group_id)
    if code == "AC-201-FALLBACK":
        detail["reason"] = fallback_reason
        fallback = "F-A3" if fallback_reason == "no_policy" else "F-A2"
    if event.http_status == 400:
        field = answer_field(event.message)
        if field:
            detail["field"] = field
    if catalog.is_unmapped(code):
        detail["note"] = "This answer is not in the error-code list (Monitoring-Spec §6); D3 lists it weekly."
    who = ", ".join(
        f"{k}={v}" for k, v in (("group_id", event.group_id), ("decision_type", event.decision_type),
                                 ("decision_idx", event.decision_idx)) if v is not None
    )
    message = f"{entry.title}: {event.http_status} {event.title or ''} on {event.endpoint}".rstrip()
    if who:
        message += f" ({who})"
    message += "."
    return Finding(
        code,
        message,
        key,
        at=event.occurred_at,
        detail=detail,
        group_id=event.group_id,
        decision_type=event.decision_type,
        decision_idx=event.decision_idx,
        rid=event.rid,
        fallback_used=fallback,
    )


def fallback_reason_for(event) -> str | None:
    """``actions.warmup_reason`` of the decision an ``AC-201-FALLBACK`` answer returned (by its rid)."""
    from app.models import Action

    if event.code != "AC-201-FALLBACK" or not event.rid:
        return None
    row = Action.query.with_entities(Action.warmup_reason).filter(Action.rid == event.rid).first()
    return row[0] if row is not None else None


def check_codes(ctx: CheckContext) -> list[Finding]:
    """Every coded answer of the window has its alarm (a safety net behind the after-response step)."""
    from app.models import ApiErrorEvent

    events = (
        ApiErrorEvent.query.filter(
            ApiErrorEvent.occurred_at > ctx.window_start, ApiErrorEvent.occurred_at <= ctx.window_end
        )
        .order_by(ApiErrorEvent.occurred_at.asc(), ApiErrorEvent.id.asc())
        .all()
    )
    out = []
    for event in events:
        finding = code_finding(event, fallback_reason=fallback_reason_for(event))
        if finding is not None:
            out.append(finding)
    return out


# --------------------------------------------------------------------- D1


def export_dir() -> str:
    from flask import current_app

    return str(current_app.config.get("MONITOR_EXPORT_DIR") or "exports")


def read_export_manifest(directory: str | None = None) -> dict:
    """
    One D1 reading: whether ``<export dir>/MANIFEST.json`` exists, when it was
    generated and the row count per exported file. Saved in the run's
    ``monitor_runs.detail`` so the check works from saved readings.
    """
    from app.clock import server_now

    directory = directory or export_dir()
    path = os.path.join(directory, "MANIFEST.json")
    reading = {"path": path, "read_at": server_now().isoformat(), "present": False, "generated_at": None,
               "files": {}}
    try:
        with open(path) as handle:
            manifest = json.load(handle)
    except FileNotFoundError:
        return reading
    except (OSError, ValueError) as exc:
        reading["error"] = type(exc).__name__
        return reading
    files = manifest.get("files") if isinstance(manifest, dict) else None
    reading["present"] = True
    reading["generated_at"] = manifest.get("generated_at") if isinstance(manifest, dict) else None
    if isinstance(files, dict):
        reading["files"] = {
            name: int(entry.get("rows", 0)) for name, entry in files.items() if isinstance(entry, dict)
        }
    return reading


def _parse_time(text):
    from app.clock import to_study_local

    try:
        return to_study_local(text)
    except (TypeError, ValueError):
        return None


def d1_problems(reading: dict, previous: dict | None) -> list[dict]:
    """D1 findings of one reading against the previous day's reading."""
    from app.models import export_models

    if not reading.get("present"):
        return [{"kind": "manifest_missing", "path": reading.get("path")}]
    problems = []
    generated = _parse_time(reading.get("generated_at"))
    read_at = _parse_time(reading.get("read_at"))
    if generated is None or (read_at is not None
                             and read_at - generated > datetime.timedelta(hours=catalog.D1_MAX_AGE_HOURS)):
        problems.append({"kind": "export_stale", "generated_at": reading.get("generated_at"),
                         "max_age_hours": catalog.D1_MAX_AGE_HOURS})
    files = reading.get("files") or {}
    expected = sorted(f"{m.__tablename__}.csv" for m in export_models(include_quarantine=False))
    missing = [f for f in expected if f not in files]
    if missing:
        problems.append({"kind": "tables_missing", "files": missing})
    if previous and previous.get("present"):
        changed = []
        for name, rows in sorted(files.items()):
            before = (previous.get("files") or {}).get(name)
            if before is None or before == 0:
                continue  # no day-before count to compare with
            change = (rows - before) / before
            if abs(change) > catalog.D1_ROW_TOLERANCE:
                changed.append({"file": name, "rows": rows, "rows_day_before": before, "change": round(change, 4)})
        if changed:
            problems.append({"kind": "row_count_change", "tolerance": catalog.D1_ROW_TOLERANCE, "files": changed})
    return problems


def saved_reading(name: str, occurrence: datetime.datetime) -> dict | None:
    """A reading the runner saved for ``occurrence`` under ``monitor_runs.detail["readings"][name]``."""
    from app.models import MonitorRun

    key = occurrence.isoformat()
    runs = (
        MonitorRun.query.filter(
            MonitorRun.logical_now >= occurrence,
            MonitorRun.logical_now <= occurrence + datetime.timedelta(days=2),
        )
        .order_by(MonitorRun.id.asc())
        .all()
    )
    for run in runs:
        readings = ((run.detail or {}).get("readings") or {}).get(name) or {}
        if key in readings:
            return readings[key]
    return None


def check_d1(ctx: CheckContext) -> list[Finding]:
    """
    D1 (daily, green): ``flask export-csv`` ran within the last day, every
    table is present, row counts are within +-20% of the day before, and the
    manifest of exported files with checksums exists. Uses the manifest
    readings the runner saved at each D1 time.
    """
    reading = saved_reading("D1", ctx.now)
    if reading is None:
        return []
    previous = saved_reading("D1", ctx.now - datetime.timedelta(days=1))
    problems = d1_problems(reading, previous)
    if not problems:
        return []
    day = ctx.now.date()
    return [Finding(
        "D1",
        f"Daily export check on {day.isoformat()}: " + ", ".join(p["kind"] for p in problems) + ".",
        dedup_key("D1", day),
        at=ctx.now,
        detail={"day": day, "problems": problems, "manifest_path": reading.get("path"),
                "generated_at": reading.get("generated_at")},
    )]


# --------------------------------------------------------------------- D3


def check_d3(ctx: CheckContext) -> list[Finding]:
    """
    D3 (weekly, green): the API returned an answer last week that is not in
    the error-code list (``XX-<status>-UNMAPPED``), or one whose wording §6
    does not yet document (Monitoring-Spec §6.9 rule 4).
    """
    from app.models import ApiErrorEvent
    from app.study_calendar import week_monday

    this_monday = week_monday(ctx.now)
    start = datetime.datetime.combine(this_monday - datetime.timedelta(days=7), datetime.time.min)
    end = datetime.datetime.combine(this_monday, datetime.time.min)
    events = ApiErrorEvent.query.filter(ApiErrorEvent.occurred_at >= start, ApiErrorEvent.occurred_at < end).all()
    unmapped: dict[tuple, list[str]] = collections.defaultdict(list)
    undocumented: dict[tuple, list[str]] = collections.defaultdict(list)
    for event in events:
        if catalog.is_unmapped(event.code):
            unmapped[(event.code, event.endpoint, event.http_status, event.title)].append(event.event_id)
            continue
        classified = codes.classify(event.endpoint, event.http_status, event.title, event.message)
        if classified.code == event.code and not classified.documented:
            undocumented[(event.code, event.endpoint, event.http_status, event.title)].append(event.event_id)
    if not unmapped and not undocumented:
        return []

    def rows(groups):
        return [{"code": k[0], "endpoint": k[1], "http_status": k[2], "title": k[3], "count": len(v),
                 "event_ids": capped(v, 5)} for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))]

    week = start.date()
    return [Finding(
        "D3",
        f"Answers missing from the error-code list in the week of {week.isoformat()}: "
        f"{sum(len(v) for v in unmapped.values())} unmapped, "
        f"{sum(len(v) for v in undocumented.values())} with undocumented wording.",
        dedup_key("D3", week),
        at=ctx.now,
        detail={"week": week, "unmapped": rows(unmapped), "undocumented_wording": rows(undocumented)},
        value=float(sum(len(v) for v in unmapped.values())),
    )]


__all__ = [
    "answer_field",
    "code_finding",
    "fallback_reason_for",
    "check_codes",
    "export_dir",
    "read_export_manifest",
    "d1_problems",
    "saved_reading",
    "check_d1",
    "check_d3",
]
