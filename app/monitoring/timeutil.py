"""
Monitoring clocks (implementation contract D4, D20; Monitoring-Spec §2.2, §3).

Monitoring works on the study clock (``America/Detroit``, naive local
datetimes):

- :func:`monitoring_now` is "now" for the monitoring loop and for
  ``GET /monitor/alarms``. With ``CLOCK_MODE = "wall"`` it is the server's
  wall clock. With ``CLOCK_MODE = "host"`` (simulations that compress the
  calendar) it is :func:`app.clock.study_now` (the latest host
  ``request_timestamp``) or the latest ``monitor_runs.logical_now`` if a
  monitoring run was evaluated at a later logical time, so that alarms a run
  raised at, e.g., Monday 05:45 are inside the window of a pull made right
  after it.
- :func:`alarm_time` is the ``raised_at`` written for an alarm whose logical
  event time is known: the logical time in host mode; the server wall clock
  in wall mode (a host timestamp that is off must never push an alarm out of
  the host's 24-hour pull window).
- :func:`server_time_for` maps a logical time to the server time at which the
  API had seen everything up to it. ``group_history.recorded_at`` and other
  ``created_at`` columns are server times; in host mode they cannot be
  compared with logical times directly (Monitoring-Spec §9.6 "as of the time
  being checked").
"""

from __future__ import annotations

import datetime

from app import clock


def is_host_mode() -> bool:
    return clock.clock_mode() == clock.CLOCK_MODE_HOST


def _latest_run_logical_now() -> datetime.datetime | None:
    from sqlalchemy import func

    from app.extensions import db
    from app.models import MonitorRun

    return db.session.query(func.max(MonitorRun.logical_now)).scalar()


def monitoring_now() -> datetime.datetime:
    """The study-clock "now" used by monitoring (see the module docstring). Requires an app context."""
    if not is_host_mode():
        return clock.server_now()
    now = clock.study_now()
    try:
        latest = _latest_run_logical_now()
    except Exception:  # noqa: BLE001 - the window still works from the host clock alone
        latest = None
    if latest is not None and latest > now:
        return latest
    return now


def alarm_time(logical: datetime.datetime | None) -> datetime.datetime:
    """``raised_at`` for an alarm about an event at study time ``logical`` (see the module docstring)."""
    if not is_host_mode():
        return clock.server_now()
    if logical is not None:
        return logical
    return monitoring_now()


def server_time_for(logical: datetime.datetime | None) -> datetime.datetime | None:
    """
    Server time by which the API had received every host call stamped at or
    before ``logical``: ``logical`` itself in wall mode; in host mode the
    latest ``created_at`` / ``timestamp`` among actions, uploads and update
    requests stamped at or before ``logical`` (the earliest later one when
    there is none; ``None`` when the tables are empty, meaning "no cutoff").
    """
    if logical is None or not is_host_mode():
        return logical
    from sqlalchemy import func

    from app.extensions import db
    from app.models import Action, DataUpload, ModelUpdateRequests

    created = {Action: Action.timestamp, DataUpload: DataUpload.created_at, ModelUpdateRequests: ModelUpdateRequests.created_at}
    best = None
    for model, column in created.items():
        value = (
            db.session.query(func.max(column)).filter(model.request_timestamp <= logical).scalar()
        )
        if value is not None and (best is None or value > best):
            best = value
    if best is not None:
        return best
    for model, column in created.items():
        value = db.session.query(func.min(column)).filter(model.request_timestamp > logical).scalar()
        if value is not None and (best is None or value < best):
            best = value
    return best


def day_start(d: datetime.date) -> datetime.datetime:
    return datetime.datetime.combine(d, datetime.time.min)


def parse_hhmm(text: str) -> datetime.time:
    hour, minute = (int(part) for part in str(text).strip().split(":"))
    return datetime.time(hour, minute)


__all__ = [
    "is_host_mode",
    "monitoring_now",
    "alarm_time",
    "server_time_for",
    "day_start",
    "parse_hhmm",
]
