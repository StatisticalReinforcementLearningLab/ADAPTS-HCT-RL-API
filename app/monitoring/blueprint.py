"""
``GET /api/v1/monitor/alarms`` — the host collects alarms (API-Spec §2.5,
§5.10, §5.11; Monitoring-Spec §6.6, §8.1). The former
``/api/v1/monitor/events`` is a deprecated alias to the same handler.

Every call returns every alarm whose ``raised_at`` falls in the 24 hours
before the study clock's "now" (:func:`app.monitoring.timeutil.monitoring_now`;
contract D4/D20), all severities and both owners, oldest first. Each alarm
carries ``event_id``, ``item_id``, ``severity``, ``owner``, ``raised_at``
(``America/Detroit`` with its UTC offset), ``message`` and ``detail``.
There is no paging, no query parameter and no heartbeat field; overlapping
pulls are expected and deduplicated by the host on ``event_id``.

Answers: ``200 Success`` ``Returning <n> alarms.``; ``503 Service
Unavailable`` ``Monitoring store unavailable; retry with backoff.`` when the
alarms cannot be read; ``500 Internal Error`` ``Internal server error.``. On
failure ``count`` and ``alarms`` are ``null``. Every call, failures
included, writes one ``monitor_pulls`` row keyed by its ``rid`` (E6 reads
them); the answer's code is logged by the after-response step.
"""

from __future__ import annotations

import datetime
import logging

from flask import Blueprint, request
from sqlalchemy.exc import SQLAlchemyError

from app.extensions import db
from app.monitoring.alarms import alarm_feed_record, report_internal_error

logger = logging.getLogger(__name__)

monitor_blueprint = Blueprint("monitor", __name__)

WINDOW = datetime.timedelta(hours=24)
MSG_STORE = "Monitoring store unavailable; retry with backoff."
MSG_INTERNAL = "Internal server error."


def _record_pull(rid: str, status: int, window_start, returned_count) -> None:
    """One ``monitor_pulls`` row per call (API-Spec §5.11); a failure is logged (E7 if possible)."""
    from app.clock import server_now
    from app.models import MonitorPull

    try:
        db.session.rollback()
        db.session.add(MonitorPull(rid=rid, status=status, window_start=window_start,
                                   returned_count=returned_count, created_at=server_now()))
        db.session.commit()
    except Exception as exc:  # noqa: BLE001 - the answer still goes out
        db.session.rollback()
        report_internal_error("monitor_pulls", exc)


def read_alarm_window(now: datetime.datetime) -> list:
    """Alarms with ``now - 24 h <= raised_at <= now``, oldest first."""
    from app.models import MonitorAlarm

    return (
        MonitorAlarm.query.filter(MonitorAlarm.raised_at >= now - WINDOW, MonitorAlarm.raised_at <= now)
        .order_by(MonitorAlarm.raised_at.asc(), MonitorAlarm.id.asc())
        .all()
    )


@monitor_blueprint.route("/monitor/alarms", methods=["GET"])
@monitor_blueprint.route("/monitor/events", methods=["GET"])  # deprecated alias
def get_alarms():
    """Collect the alarms of the last 24 hours (API-Spec §2.5)."""
    from app.monitoring.timeutil import monitoring_now
    from app.routes.envelope import envelope, new_rid

    rid = new_rid()
    window_start = None
    try:
        if request.path.endswith("/monitor/events"):
            logger.warning("[Monitor] /monitor/events is deprecated; use /monitor/alarms.")
        try:
            now = monitoring_now()
            window_start = now - WINDOW
            alarms = [alarm_feed_record(row) for row in read_alarm_window(now)]
        except SQLAlchemyError:
            logger.error("[Monitor] Alarm store unreadable (rid=%s).", rid, exc_info=True)
            db.session.rollback()
            _record_pull(rid, 503, window_start, None)
            return envelope(503, "Service Unavailable", MSG_STORE, rid, count=None, alarms=None)
        _record_pull(rid, 200, window_start, len(alarms))
        return envelope(200, "Success", f"Returning {len(alarms)} alarms.", rid, count=len(alarms), alarms=alarms)
    except Exception:  # noqa: BLE001 - last resort of this endpoint
        logger.error("[Monitor] Unexpected error (rid=%s).", rid, exc_info=True)
        db.session.rollback()
        _record_pull(rid, 500, window_start, None)
        return envelope(500, "Internal Error", MSG_INTERNAL, rid, count=None, alarms=None)


__all__ = ["monitor_blueprint", "get_alarms", "read_alarm_window", "WINDOW"]
