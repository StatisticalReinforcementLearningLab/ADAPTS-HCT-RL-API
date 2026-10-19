"""
``flask monitor report`` and ``flask monitor status`` (Monitoring-Spec §8.3).

The report gives the tables for the paper over a period: per severity (check,
number of alarms, dyads, fallback used), per fallback (§7), per error code
(§6), plus the EB fallback rate and the red alarms still without a note.
"""

from __future__ import annotations

import collections
import datetime

from app.extensions import db
from app.monitoring import catalog


def _dyads(row) -> set[str]:
    out = set()
    if row.group_id:
        out.add(row.group_id)
    detail = row.detail if isinstance(row.detail, dict) else {}
    for gid in detail.get("group_ids") or []:
        out.add(str(gid))
    return out


def build_report(since: datetime.datetime, until: datetime.datetime) -> dict:
    from app.models import MonitorAlarm
    from app.monitoring.checks.summary import code_counts, eb_fallback_rate, fallback_counts

    rows = (
        MonitorAlarm.query.filter(MonitorAlarm.raised_at >= since, MonitorAlarm.raised_at <= until)
        .order_by(MonitorAlarm.raised_at.asc(), MonitorAlarm.id.asc())
        .all()
    )
    groups: dict[tuple[str, str], dict] = {}
    by_alarm_fallback: collections.Counter = collections.Counter()
    unnoted_red = []
    for row in rows:
        key = (row.severity, row.item_id)
        entry = groups.setdefault(key, {"item_id": row.item_id, "n_alarms": 0, "dyads": set(), "fallback_used": set()})
        entry["n_alarms"] += 1
        entry["dyads"] |= _dyads(row)
        if row.fallback_used:
            entry["fallback_used"].add(row.fallback_used)
            by_alarm_fallback[row.fallback_used] += 1
        if row.severity == catalog.RED and not row.note:
            unnoted_red.append(row.event_id)
    by_severity = {}
    for severity in catalog.SEVERITIES:
        items = []
        for (sev, item_id), entry in sorted(groups.items()):
            if sev != severity:
                continue
            try:
                title = catalog.lookup(item_id).title
            except KeyError:
                title = ""
            items.append({"item_id": item_id, "title": title, "n_alarms": entry["n_alarms"],
                          "n_dyads": len(entry["dyads"]), "fallback_used": sorted(entry["fallback_used"])})
        by_severity[severity] = items
    start = since - datetime.timedelta(microseconds=1)
    return {
        "since": since,
        "until": until,
        "n_alarms": len(rows),
        "check_list_version": catalog.CHECK_LIST_VERSION,
        "by_severity": by_severity,
        "by_fallback": fallback_counts(start, until),
        "alarms_by_fallback": dict(by_alarm_fallback),
        "by_code": code_counts(start, until),
        "eb_fallback_rate": eb_fallback_rate(),
        "red_alarms_without_note": unnoted_red,
    }


def _table(headers: list[str], rows: list[list]) -> str:
    cells = [[str(h) for h in headers]] + [["" if c is None else str(c) for c in r] for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def format_report(report: dict) -> str:
    out = [f"Monitoring report {report['since']} .. {report['until']} "
           f"(check list {report['check_list_version']}; {report['n_alarms']} alarms)", ""]
    for severity in catalog.SEVERITIES:
        items = report["by_severity"][severity]
        out.append(f"{severity.upper()} alarms")
        if items:
            out.append(_table(["check", "title", "alarms", "dyads", "fallback used"],
                              [[i["item_id"], i["title"], i["n_alarms"], i["n_dyads"], ", ".join(i["fallback_used"])]
                               for i in items]))
        else:
            out.append("  (none)")
        out.append("")
    out.append("Fallbacks (Monitoring-Spec §7)")
    fb_rows = []
    for fallback, per in sorted(report["by_fallback"].items()):
        total = sum(per.values())
        fb_rows.append([fallback, total, ", ".join(f"{k}: {v}" for k, v in sorted(per.items())) or "-"])
    out.append(_table(["fallback", "uses", "by agent"], fb_rows))
    out.append("")
    out.append("Error codes (Monitoring-Spec §6)")
    code_rows = [[code, sum(per.values()), ", ".join(f"{k}: {v}" for k, v in sorted(per.items()))]
                 for code, per in sorted(report["by_code"].items())]
    out.append(_table(["code", "answers", "by agent"], code_rows) if code_rows else "  (none)")
    out.append("")
    out.append("EB fallback rate (C1)")
    eb_rows = [[agent, v["updates"], v["fallbacks"], "-" if v["rate"] is None else f"{v['rate']:.2f}"]
               for agent, v in sorted(report["eb_fallback_rate"].items())]
    out.append(_table(["agent", "updates", "fallbacks", "rate"], eb_rows) if eb_rows else "  (no updates)")
    if report["red_alarms_without_note"]:
        out.append("")
        out.append(f"Red alarms without a note: {len(report['red_alarms_without_note'])} "
                   f"(add one with `flask monitor note <event_id> \"...\"`)")
    return "\n".join(out)


def build_status(now: datetime.datetime) -> dict:
    """This week's update, alarms of the last 24 h, the last monitoring run, the last host pulls."""
    from app.clock import format_detroit
    from app.models import MonitorAlarm, MonitorPull, MonitorRun
    from app.monitoring.checks.summary import update_week_state
    from app.monitoring.checks.system import _pull_time
    from app.study_calendar import week_monday

    start = now - datetime.timedelta(hours=24)
    alarms = (
        MonitorAlarm.query.filter(MonitorAlarm.raised_at >= start, MonitorAlarm.raised_at <= now)
        .order_by(MonitorAlarm.raised_at.asc(), MonitorAlarm.id.asc())
        .all()
    )
    last_run = MonitorRun.query.order_by(MonitorRun.id.desc()).first()
    last_pull = MonitorPull.query.order_by(MonitorPull.id.desc()).first()
    ok_pulls = [p for p in MonitorPull.query.filter(MonitorPull.status == 200).all() if _pull_time(p) is not None]
    last_ok = max(ok_pulls, key=lambda p: (_pull_time(p), p.id)) if ok_pulls else None
    return {
        "now": format_detroit(now, "seconds"),
        "update": update_week_state(week_monday(now)),
        "alarms_24h": {
            "counts": dict(collections.Counter(a.severity for a in alarms)),
            "items": [{"event_id": a.event_id, "item_id": a.item_id, "severity": a.severity, "owner": a.owner,
                       "raised_at": format_detroit(a.raised_at, "seconds"), "message": a.message} for a in alarms],
        },
        "last_run": None if last_run is None else {
            "run_id": last_run.id, "kind": last_run.kind, "status": last_run.status,
            "logical_now": last_run.logical_now, "started_at": last_run.started_at,
            "finished_at": last_run.finished_at, "n_alarms": last_run.n_alarms,
        },
        "last_pull": None if last_pull is None else {
            "rid": last_pull.rid, "status": last_pull.status, "created_at": last_pull.created_at,
            "returned_count": last_pull.returned_count,
        },
        "last_successful_pull": None if last_ok is None else {
            "rid": last_ok.rid, "at": format_detroit(_pull_time(last_ok), "seconds"),
            "returned_count": last_ok.returned_count,
        },
        "n_alarms_total": db.session.query(MonitorAlarm.id).count(),
    }


def format_status(status: dict) -> str:
    update = status["update"]
    lines = [f"Monitoring status at {status['now']}", ""]
    lines.append(f"This week's update ({update['study_week_monday']}): {update['status'] or 'no call yet'}"
                 + (f", ready at {update['ready_at']}" if update.get("ready_at") else "")
                 + (", C7 raised" if update.get("c7_raised") else ""))
    for call in update["calls"]:
        lines.append(f"  {call['rid']}: {call['status']} (requested {call['request_timestamp']}, "
                     f"{call['duration_seconds']} s)")
    counts = status["alarms_24h"]["counts"]
    lines.append("")
    lines.append(f"Alarms in the last 24 h: {counts.get('red', 0)} red, {counts.get('yellow', 0)} yellow, "
                 f"{counts.get('green', 0)} green")
    for a in status["alarms_24h"]["items"]:
        lines.append(f"  [{a['severity']}/{a['owner']}] {a['raised_at']} {a['item_id']}: {a['message']} "
                     f"({a['event_id']})")
    lines.append("")
    run = status["last_run"]
    lines.append("Last monitoring run: " + ("none" if run is None else
                 f"#{run['run_id']} {run['kind']} {run['status']} at {run['logical_now']} "
                 f"(finished {run['finished_at']}, {run['n_alarms']} new alarms)"))
    ok = status["last_successful_pull"]
    lines.append("Last successful host pull: " + ("never" if ok is None else f"{ok['at']} ({ok['rid']})"))
    pull = status["last_pull"]
    if pull is not None and (ok is None or pull["rid"] != ok["rid"]):
        lines.append(f"Last host pull: status {pull['status']} at {pull['created_at']} ({pull['rid']})")
    return "\n".join(lines)


__all__ = ["build_report", "format_report", "build_status", "format_status"]
