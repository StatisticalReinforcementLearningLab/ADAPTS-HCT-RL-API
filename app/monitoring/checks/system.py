"""
The server and monitoring itself (Monitoring-Spec §5.E): E1, E5, E6.

E1 and E6 are "raised when a problem starts, not on every run" (M4): E1 at
each saved reading that crosses its limit after a reading below it, E6 once
per gap of more than 24 hours without a successful ``GET /monitor/alarms``.
Both work only from saved data (E1 from the readings each run stores in
``monitor_runs.detail``; E6 from ``monitor_pulls``), so a rerun over a past
period finds the same start events.
"""

from __future__ import annotations

import datetime
import os
import shutil

from app.extensions import db
from app.monitoring import catalog
from app.monitoring.checks.common import CheckContext
from app.monitoring.findings import Finding, dedup_key

# --------------------------------------------------------------------- E1


def _memory_pct() -> float | None:
    try:
        info = {}
        with open("/proc/meminfo") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                info[key.strip()] = float(rest.strip().split()[0])
        total = info.get("MemTotal")
        available = info.get("MemAvailable")
        if not total or available is None:
            return None
        return round(100.0 * (total - available) / total, 3)
    except (OSError, ValueError, IndexError):
        return None


def _disk_path() -> str:
    from flask import current_app

    from app import _sqlite_file_path  # noqa: PLC0415 - foundation helper

    path = _sqlite_file_path(current_app.config.get("SQLALCHEMY_DATABASE_URI"))
    if path:
        return os.path.dirname(os.path.abspath(path)) or "."
    return os.getcwd()


def _disk_pct(path: str) -> float | None:
    try:
        usage = shutil.disk_usage(path)
        return round(100.0 * usage.used / usage.total, 3) if usage.total else None
    except OSError:
        return None


def _db_connections() -> tuple[int | None, int | None]:
    """``(connections, max_connections)`` for PostgreSQL (``pg_stat_activity``); ``(None, None)`` otherwise."""
    from sqlalchemy import text

    if db.engine.dialect.name != "postgresql":
        return None, None
    try:
        with db.engine.connect() as conn:
            used = conn.execute(text("SELECT count(*) FROM pg_stat_activity")).scalar()
            limit = conn.execute(text("SHOW max_connections")).scalar()
        return int(used), int(limit)
    except Exception:  # noqa: BLE001 - an unreadable counter is "n/a", not a crash
        return None, None


def read_server_resources() -> dict:
    """
    One E1 reading: memory (``/proc/meminfo``), disk of the database
    directory (``shutil.disk_usage``) and database connections (PostgreSQL
    ``pg_stat_activity`` against ``max_connections``; ``None`` for SQLite).
    """
    path = _disk_path()
    used, limit = _db_connections()
    return {
        "memory_pct": _memory_pct(),
        "disk_pct": _disk_pct(path),
        "disk_path": path,
        "db_connections": used,
        "db_max_connections": limit,
        "db_conn_pct": round(100.0 * used / limit, 3) if used is not None and limit else None,
        "db_dialect": db.engine.dialect.name,
    }


def _readings(until: datetime.datetime) -> list:
    from app.models import MonitorRun

    return (
        MonitorRun.query.filter(MonitorRun.logical_now <= until, MonitorRun.detail.isnot(None))
        .order_by(MonitorRun.logical_now.asc(), MonitorRun.id.asc())
        .all()
    )


def check_e1(ctx: CheckContext) -> list[Finding]:
    """E1: a saved reading above its limit (memory 85%, disk 90%, DB connections 90%) after one at or below it."""
    out: list[Finding] = []
    previous: dict = {}
    for run in _readings(ctx.window_end):
        reading = (run.detail or {}).get("e1")
        if not isinstance(reading, dict):
            continue
        in_window = run.logical_now is not None and run.logical_now > ctx.window_start
        for metric, limit in catalog.E1_LIMITS.items():
            value = reading.get(metric)
            if not isinstance(value, (int, float)):
                continue
            before = previous.get(metric)
            if in_window and value > limit and (before is None or before <= limit):
                out.append(Finding(
                    "E1",
                    f"Server resource {metric} at {value:.1f}% (limit {limit:g}%).",
                    dedup_key("E1", metric, run.id),
                    at=run.logical_now,
                    detail={"metric": metric, "reading": reading, "run_id": run.id},
                    value=float(value),
                    threshold=float(limit),
                ))
            previous[metric] = value
    return out


# --------------------------------------------------------------------- E6


def _pull_time(pull) -> datetime.datetime | None:
    """Logical time of a pull: its window end (``window_start + 24 h``)."""
    if pull.window_start is None:
        return None
    return pull.window_start + datetime.timedelta(hours=24)


def check_e6(ctx: CheckContext) -> list[Finding]:
    """
    E6: no successful ``GET /monitor/alarms`` for 24 hours (``monitor_pulls``).
    Raised once when a gap starts (at the last success + 24 h), with the
    first monitoring run as the reference before any pull.
    """
    from sqlalchemy import func

    from app.models import MonitorPull, MonitorRun

    gap = datetime.timedelta(hours=catalog.E6_GAP_HOURS)
    pulls = [p for p in MonitorPull.query.filter(MonitorPull.status == 200).all() if _pull_time(p) is not None]
    pulls.sort(key=lambda p: (_pull_time(p), p.id))
    pulls = [p for p in pulls if _pull_time(p) <= ctx.window_end]
    out: list[Finding] = []

    def gap_start(start: datetime.datetime, nxt: datetime.datetime | None, key, last_rid):
        start_of_gap = start + gap
        if not (ctx.window_start < start_of_gap <= ctx.window_end):
            return
        if nxt is not None and nxt <= start_of_gap:
            return
        out.append(Finding(
            "E6",
            "The host has not collected alarms (GET /monitor/alarms) for 24 hours.",
            key,
            at=start_of_gap,
            detail={"last_successful_pull": start if last_rid else None, "last_pull_rid": last_rid,
                    "gap_hours": catalog.E6_GAP_HOURS},
            threshold=float(catalog.E6_GAP_HOURS),
        ))

    first_run = db.session.query(func.min(MonitorRun.logical_now)).scalar()
    if first_run is not None:
        first_pull = _pull_time(pulls[0]) if pulls else None
        if first_pull is None or first_pull > first_run:
            gap_start(first_run, first_pull, dedup_key("E6", "never", first_run), None)
    for i, pull in enumerate(pulls):
        nxt = _pull_time(pulls[i + 1]) if i + 1 < len(pulls) else None
        gap_start(_pull_time(pull), nxt, dedup_key("E6", pull.rid), pull.rid)
    return out


# --------------------------------------------------------------------- E5


def e5_finding(last: datetime.datetime, now: datetime.datetime, n_caught_up: int) -> Finding:
    """E5: monitoring found a gap of more than 15 minutes since its last run and reran the missed period."""
    minutes = (now - last).total_seconds() / 60.0
    return Finding(
        "E5",
        f"Monitoring was down for {minutes:.0f} minutes; every scheduled check was run again over the missed period.",
        dedup_key("E5", last, now),
        at=now,
        detail={"last_run": last, "restarted_at": now, "gap_minutes": round(minutes, 2),
                "alarms_raised_in_catch_up": n_caught_up},
        value=round(minutes, 2),
        threshold=float(catalog.E5_GAP_MINUTES),
    )


__all__ = ["read_server_resources", "check_e1", "check_e6", "e5_finding"]
