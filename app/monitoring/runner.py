"""
The monitoring program (Monitoring-Spec §2, §2.3, §8.3; implementation contract D20).

:func:`run_monitoring` is one run, at a logical time ``now`` (default
:func:`app.monitoring.timeutil.monitoring_now`):

1. a ``monitor_runs`` row is written first (the "still alive" row, §2.3);
   ``kind`` is ``regular`` (the 5-minute run), ``check`` (one item) or
   ``rerun`` (an explicit ``since``);
2. a regular run saves the E1 server readings and, at each D1 time, the
   export-manifest reading in its ``detail``;
3. F-U4: updates running longer than ``MONITOR_UPDATE_TIMEOUT_MIN`` are marked
   failed (``"timeout"``) and their week ``stuck``;
4. every check whose schedule falls in ``(window_start, now]`` runs at its own
   scheduled logical time, in time order (a run after an outage therefore
   catches up every missed daily and weekly check); every-run checks run once
   over the window;
5. a regular run that finds a gap of more than 15 minutes since the previous
   run raises E5 after the catch-up.

A check that crashes is caught and recorded as E7; the run goes on. Alarms
are deduplicated, so running the same window twice raises nothing new.

The window of a regular run starts at the latest ``logical_now`` of the
previous regular or rerun runs. A simulator driving the host clock can pass
``since`` (its previous ``now``) to run exactly the elapsed period without
an E5 check.
"""

from __future__ import annotations

import datetime
import logging
import time

from app.extensions import db
from app.monitoring import catalog
from app.monitoring.alarms import emit, report_internal_error
from app.monitoring.checks import EVENT, EVERY_RUN, REGISTRY, WEEKLY
from app.monitoring.checks.common import CheckContext
from app.monitoring.findings import dedup_key, jsonable
from app.monitoring.timeutil import alarm_time, monitoring_now

logger = logging.getLogger(__name__)

KIND_REGULAR = "regular"
KIND_CHECK = "check"
KIND_RERUN = "rerun"
RUN_ITEMS = tuple(REGISTRY)


def _to_local(value):
    from app.clock import to_study_local

    return None if value is None else to_study_local(value)


def last_run_time(exclude_id: int | None = None) -> datetime.datetime | None:
    """The latest ``logical_now`` of the regular and rerun runs (the start of the next regular window)."""
    from sqlalchemy import func

    from app.models import MonitorRun

    q = db.session.query(func.max(MonitorRun.logical_now)).filter(MonitorRun.kind.in_((KIND_REGULAR, KIND_RERUN)))
    if exclude_id is not None:
        q = q.filter(MonitorRun.id != exclude_id)
    return q.scalar()


def _run_every() -> datetime.timedelta:
    from flask import current_app

    return datetime.timedelta(seconds=int(current_app.config.get("MONITOR_RUN_EVERY_SECONDS", 300)))


def _window_start(now, since, item, last) -> datetime.datetime:
    if since is not None:
        return since
    if item is not None:
        schedule = REGISTRY[item].schedule
        if schedule.kind in (WEEKLY, EVENT):
            return now - datetime.timedelta(days=7)
        return now - datetime.timedelta(days=1)
    if last is not None and last < now:
        return last
    return now - _run_every()


def _tasks(items, window_start, now):
    tasks = []
    for item in items:
        schedule = item.schedule
        if schedule.kind in (EVERY_RUN, EVENT):
            tasks.append((now, item.order, item, window_start, now))
        else:
            period = schedule.period()
            for t in schedule.occurrences(window_start, now):
                tasks.append((t, item.order, item, t - period, t))
    tasks.sort(key=lambda task: (task[0], task[1]))
    return tasks


def _after_finding(item_id: str, finding) -> None:
    """Bookkeeping that follows a finding (``update_weeks`` for C11 and C7)."""
    from app.monitoring.checks import update as update_checks

    monday = (finding.detail or {}).get("study_week_monday")
    if item_id not in ("C7", "C11") or monday is None:
        return
    monday = datetime.date.fromisoformat(str(monday)[:10])
    if item_id == "C7":
        update_checks.record_c7(monday)
    else:
        update_checks.ensure_week(monday)
        db.session.commit()


def run_monitoring(
    now: datetime.datetime | None = None,
    since: datetime.datetime | None = None,
    item: str | None = None,
) -> dict:
    """
    One monitoring run (module docstring). ``item`` restricts the run to one
    check of :data:`REGISTRY` (``"codes"`` re-raises the error-code alarms).
    Returns a JSON-ready summary: run id, kind, window, the dedup keys of
    every finding, the new alarms, per-item counts, errors and timeouts.
    Raises ``ValueError`` for an unknown ``item``.
    """
    from app.clock import server_now
    from app.models import MonitorRun
    from app.monitoring.checks import records, system
    from app.monitoring.checks import update as update_checks

    if item is not None and item not in REGISTRY:
        raise ValueError(f"unknown monitoring item {item!r}; expected one of {', '.join(RUN_ITEMS)}")
    now = _to_local(now) if now is not None else monitoring_now()
    since = _to_local(since)
    kind = KIND_CHECK if item is not None else (KIND_RERUN if since is not None else KIND_REGULAR)
    last = last_run_time()
    window_start = min(_window_start(now, since, item, last), now)

    run = MonitorRun(kind=kind, status="ok", window_start=window_start, window_end=now, item_id=item,
                     logical_now=now, detail={})
    db.session.add(run)
    db.session.commit()
    run_id = run.id

    detail: dict = {"items": {}, "errors": [], "timeouts": [], "readings": {}}
    findings_keys: list[str] = []
    new_alarms: list[dict] = []

    def emit_all(item_id, findings):
        for finding in findings:
            findings_keys.append(finding.dedup_key)
            row = emit(finding, source=catalog.SOURCE_SCHEDULED, run_id=run_id)
            if row is not None:
                new_alarms.append({"event_id": row.event_id, "item_id": row.item_id, "severity": row.severity,
                                   "dedup_key": row.dedup_key})
            try:
                _after_finding(item_id, finding)
            except Exception as exc:  # noqa: BLE001
                report_internal_error(f"bookkeeping after {item_id}", exc, run_id=run_id)

    # 2. readings saved with the run (E1 every regular run; D1 at its times).
    if kind == KIND_REGULAR:
        try:
            detail["e1"] = system.read_server_resources()
        except Exception as exc:  # noqa: BLE001
            report_internal_error("E1 readings", exc, run_id=run_id)
    if kind in (KIND_REGULAR, KIND_CHECK) and (item is None or item == "D1"):
        d1_times = REGISTRY["D1"].schedule.occurrences(window_start, now)
        if d1_times:
            try:
                reading = records.read_export_manifest()
                detail["readings"]["D1"] = {t.isoformat(): reading for t in d1_times}
            except Exception as exc:  # noqa: BLE001
                report_internal_error("D1 readings", exc, run_id=run_id)
    run = db.session.get(MonitorRun, run_id)
    run.detail = jsonable(detail)
    db.session.commit()

    # 3. F-U4 timeouts.
    try:
        detail["timeouts"] = update_checks.mark_timeouts(now)
    except Exception as exc:  # noqa: BLE001
        report_internal_error("F-U4 timeouts", exc, run_id=run_id)

    # 4. the checks, in logical time order.
    items = [REGISTRY[item]] if item is not None else list(REGISTRY.values())
    for t, _order, entry, ws, we in _tasks(items, window_start, now):
        ctx = CheckContext(now=t, window_start=ws, window_end=we, run_id=run_id, kind=kind)
        try:
            findings = entry.fn(ctx)
        except Exception as exc:  # noqa: BLE001 - E7, and the run goes on
            db.session.rollback()
            detail["errors"].append({"item": entry.item_id, "at": t.isoformat(), "error": type(exc).__name__})
            report_internal_error(
                f"check {entry.item_id}", exc, key=dedup_key("E7", entry.item_id, t), raised_at=alarm_time(t),
                run_id=run_id,
            )
            continue
        detail["items"][entry.item_id] = detail["items"].get(entry.item_id, 0) + len(findings)
        emit_all(entry.item_id, findings)

    # 5. E5 after the catch-up.
    gap = None if last is None else now - last
    if kind == KIND_REGULAR and gap is not None and gap > datetime.timedelta(minutes=catalog.E5_GAP_MINUTES):
        try:
            finding = system.e5_finding(last, now, len(new_alarms))
            detail["e5"] = {"last_run": last.isoformat(), "gap_minutes": finding.detail["gap_minutes"]}
            emit_all("E5", [finding])
        except Exception as exc:  # noqa: BLE001
            report_internal_error("E5", exc, run_id=run_id)

    run = db.session.get(MonitorRun, run_id)
    run.finished_at = server_now()
    run.n_alarms = len(new_alarms)
    run.status = "error" if detail["errors"] else "ok"
    detail["n_findings"] = len(findings_keys)
    run.detail = jsonable(detail)
    db.session.commit()
    return jsonable({
        "run_id": run_id,
        "kind": kind,
        "logical_now": now,
        "window_start": window_start,
        "window_end": now,
        "status": run.status,
        "n_alarms": len(new_alarms),
        "alarms": new_alarms,
        "findings": findings_keys,
        "items": detail["items"],
        "errors": detail["errors"],
        "timeouts": detail["timeouts"],
    })


def loop(max_runs: int | None = None, sleep=time.sleep) -> int:
    """
    ``flask monitor loop``: a regular run every ``MONITOR_RUN_EVERY_SECONDS``
    (started and restarted by systemd or supervisord, Monitoring-Spec §2.3).
    A failing run is logged and the loop goes on. Returns the number of runs.
    """
    from flask import current_app

    every = int(current_app.config.get("MONITOR_RUN_EVERY_SECONDS", 300))
    n = 0
    while max_runs is None or n < max_runs:
        started = time.monotonic()
        try:
            run_monitoring()
        except Exception:  # noqa: BLE001 - the loop must survive any run
            logger.exception("Monitoring run failed.")
            try:
                db.session.rollback()
            except Exception:  # noqa: BLE001
                pass
        finally:
            db.session.remove()
        n += 1
        if max_runs is not None and n >= max_runs:
            break
        sleep(max(0.0, every - (time.monotonic() - started)))
    return n


__all__ = ["run_monitoring", "loop", "last_run_time", "KIND_REGULAR", "KIND_CHECK", "KIND_RERUN", "RUN_ITEMS"]
