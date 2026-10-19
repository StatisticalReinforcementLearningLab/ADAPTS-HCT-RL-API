"""
On-server monitoring of the ADAPTS-HCT RL API (Monitoring-Spec; implementation
contract §3.1, D17, D20).

Monitoring is part of the API code (Monitoring-Spec §2, M1). ``init_app`` is
called by ``create_app`` without a guard: if monitoring cannot load, the API
refuses to start. It registers

- ``GET /api/v1/monitor/alarms`` and the deprecated alias
  ``/api/v1/monitor/events`` (:mod:`app.monitoring.blueprint`; API-Spec §2.5);
- the after-response error-code logger for every ``/api/v1/*`` answer
  (:mod:`app.monitoring.hooks`; Monitoring-Spec §6);
- the ``flask monitor`` command group (:mod:`app.monitoring.cli`; §8.3).

Package layout:

- :mod:`~app.monitoring.catalog` — the check list (§5) and error codes (§6);
- :mod:`~app.monitoring.codes` — answer -> code mapping (§6, D17, D19);
- :mod:`~app.monitoring.alarms` — writing alarms, dedup, notes, E7;
- :mod:`~app.monitoring.checks` — one pure function per check, with schedules;
- :mod:`~app.monitoring.hooks` — the checks that run inside the API;
- :mod:`~app.monitoring.runner` — the monitoring program (catch-up, E5, F-U4);
- :mod:`~app.monitoring.report` — ``status`` and ``report`` tables.

The routes call the hooks below only through ``try/except`` wrappers that log
and continue; the hooks themselves also never raise.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

from app.monitoring.catalog import CHECK_LIST_VERSION

if TYPE_CHECKING:  # pragma: no cover
    from app.models import MonitorAlarm


def init_app(app) -> None:
    """
    Register the ``/api/v1/monitor/alarms`` blueprint (and the
    ``/monitor/events`` alias), the ``flask monitor`` CLI group and the
    after-response error-code logger. Raises if monitoring cannot load.
    """
    from app.monitoring.blueprint import monitor_blueprint
    from app.monitoring.checks import REGISTRY  # every check is imported now: a broken one stops start-up (M1)
    from app.monitoring.cli import monitor_cli
    from app.monitoring.hooks import after_response

    if "adapts_monitoring" in app.extensions:
        return
    app.register_blueprint(monitor_blueprint, url_prefix="/api/v1")
    app.after_request(after_response)
    app.cli.add_command(monitor_cli)
    app.extensions["adapts_monitoring"] = {"check_list_version": CHECK_LIST_VERSION, "checks": sorted(REGISTRY)}


def raise_alarm(
    item_id: str,
    *,
    message: str,
    detail: dict | None = None,
    source: str,
    group_id=None,
    decision_type=None,
    decision_idx=None,
    rid=None,
    value=None,
    threshold=None,
    fallback_used=None,
    raised_at: datetime.datetime | None = None,
    dedup_key: str | None = None,
    severity: str | None = None,
    owner: str | None = None,
    run_id: int | None = None,
) -> "MonitorAlarm | None":
    """Persist one alarm (severity/owner default from the catalog). Never raises (E7 on failure)."""
    from app.monitoring import alarms

    return alarms.raise_alarm(
        item_id,
        message=message,
        detail=detail,
        source=source,
        group_id=group_id,
        decision_type=decision_type,
        decision_idx=decision_idx,
        rid=rid,
        value=value,
        threshold=threshold,
        fallback_used=fallback_used,
        raised_at=raised_at,
        dedup_key=dedup_key,
        severity=severity,
        owner=owner,
        run_id=run_id,
    )


def check_action_probability(pi: float, *, learned: bool, context: dict) -> bool:
    """A13 inline check inside /action: ``True`` iff ``pi`` is a valid probability."""
    from app.monitoring import hooks

    return hooks.check_action_probability(pi, learned=learned, context=context)


def after_action_committed(action_id: int) -> None:
    """A8, A15, A17 and per-decision C9 after a decision is committed. Never raises."""
    from app.monitoring import hooks

    hooks.after_action_committed(action_id)


def on_update_received(update_rid: str) -> None:
    """``update_weeks`` bookkeeping and B11 when an /update is accepted. Never raises."""
    from app.monitoring import hooks

    hooks.on_update_received(update_rid)


def on_update_finished(update_rid: str, summary: dict) -> None:
    """C1, C3 (from the update's rows), C10, L1-L3, B5, C9 and ``update_weeks`` after an update. Never raises."""
    from app.monitoring import hooks

    hooks.on_update_finished(update_rid, summary)


def on_startup_settings_mismatch(detail: dict) -> None:
    """C9: record the red alarm before the API refuses to start (Hyperparameters.md §2.3)."""
    from app.monitoring import hooks

    hooks.on_startup_settings_mismatch(detail)


def run_monitoring(
    now: datetime.datetime | None = None,
    since: datetime.datetime | None = None,
    item: str | None = None,
) -> dict:
    """One scheduled monitoring run (catches up missed periods, E5); used by the loop, CLI and simulator."""
    from app.monitoring import runner

    return runner.run_monitoring(now=now, since=since, item=item)


__all__ = [
    "CHECK_LIST_VERSION",
    "init_app",
    "raise_alarm",
    "check_action_probability",
    "after_action_committed",
    "on_update_received",
    "on_update_finished",
    "on_startup_settings_mismatch",
    "run_monitoring",
]
