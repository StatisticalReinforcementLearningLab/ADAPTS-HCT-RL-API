"""
Writing alarms (Monitoring-Spec §4, §9.1, M4; API-Spec §5.10).

- :func:`raise_alarm` writes one ``monitor_alarms`` row with the catalog's
  severity and owner (unless overridden by an escalation), the check-list
  version in force and a fresh ``event_id``. With a ``dedup_key`` it writes
  nothing when a row with that key exists, so a rerun over the same window,
  the runner after an inline hook, or a second process never duplicate a
  finding. It never raises: a failure is logged and recorded as E7.
- "Raise when a problem starts" (E1, E6) is expressed through the dedup key
  of the start event (see the checks).
- Rows are never deleted and never changed except to add a note
  (:func:`add_note`).
- :func:`report_internal_error` records E7 for a monitoring step that crashed;
  it is itself guarded and only logs when even that fails.

No participant data enters an alarm (§2.2): checks put only coded IDs,
numbers and thresholds in ``detail``; exception texts are reduced to the
exception type and location (:func:`describe_exception`).
"""

from __future__ import annotations

import datetime
import logging
import traceback
import uuid

from sqlalchemy.exc import IntegrityError

from app.extensions import db
from app.monitoring import catalog
from app.monitoring.findings import MAX_DEDUP_LEN, Finding, dedup_key as make_key, finite_or_none, jsonable

logger = logging.getLogger(__name__)

_MAX_MESSAGE = 2000
_MAX_ID = 255


def _clip(value, limit: int = _MAX_ID):
    if value is None:
        return None
    text = str(value)
    return text[:limit]


def _decision_idx(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if -(2**31) <= value < 2**31 else None


def alarm_exists(key: str | None) -> bool:
    """Whether an alarm with this ``dedup_key`` exists."""
    from app.models import MonitorAlarm

    if key is None:
        return False
    return db.session.query(MonitorAlarm.id).filter(MonitorAlarm.dedup_key == key).first() is not None


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
):
    """
    Persist one alarm and return the new ``MonitorAlarm`` row, or ``None``
    when nothing was written (an alarm with ``dedup_key`` exists, or the
    write failed). Never raises; a failure is recorded as E7 (implementation
    contract §3.1).
    """
    from app.models import MonitorAlarm
    from app.monitoring.timeutil import monitoring_now

    try:
        entry = catalog.lookup(item_id)
        sev = severity or entry.severity
        own = owner or entry.owner
        if sev not in catalog.SEVERITIES:
            raise ValueError(f"invalid severity {sev!r} for {item_id}")
        if own not in catalog.OWNERS:
            raise ValueError(f"invalid owner {own!r} for {item_id}")
        if source not in catalog.SOURCES:
            raise ValueError(f"invalid source {source!r} for {item_id}")
        if dedup_key is not None and len(dedup_key) > MAX_DEDUP_LEN:
            dedup_key = make_key(dedup_key)
        if dedup_key is not None and alarm_exists(dedup_key):
            return None
        row = MonitorAlarm(
            event_id=str(uuid.uuid4()),
            item_id=item_id,
            severity=sev,
            owner=own,
            source=source,
            message=str(message)[:_MAX_MESSAGE],
            check_list_version=catalog.CHECK_LIST_VERSION,
            raised_at=raised_at if raised_at is not None else monitoring_now(),
            detail=jsonable(detail or {}),
            run_id=run_id,
            group_id=_clip(group_id),
            decision_type=_clip(decision_type),
            decision_idx=_decision_idx(decision_idx),
            rid=_clip(rid),
            value=finite_or_none(value),
            threshold=finite_or_none(threshold),
            fallback_used=_clip(fallback_used),
            dedup_key=dedup_key,
        )
        db.session.add(row)
        db.session.commit()
        return row
    except IntegrityError:
        # A concurrent writer recorded the same finding first (unique dedup_key).
        _rollback()
        return None
    except Exception as exc:  # noqa: BLE001 - an alarm write never breaks its caller
        _rollback()
        logger.error("Could not write alarm %s: %s", item_id, describe_exception(exc))
        if item_id != "E7":
            report_internal_error(f"raise_alarm({item_id})", exc, raised_at=raised_at, run_id=run_id)
        return None


def emit(finding: Finding, *, source: str, run_id: int | None = None, raised_at: datetime.datetime | None = None):
    """Raise the alarm of one :class:`Finding` (``raised_at`` defaults to :func:`alarm_time` of its event)."""
    from app.monitoring.timeutil import alarm_time

    return raise_alarm(
        finding.item_id,
        message=finding.message,
        detail=finding.detail,
        source=source,
        group_id=finding.group_id,
        decision_type=finding.decision_type,
        decision_idx=finding.decision_idx,
        rid=finding.rid,
        value=finding.value,
        threshold=finding.threshold,
        fallback_used=finding.fallback_used,
        raised_at=raised_at if raised_at is not None else alarm_time(finding.at),
        dedup_key=finding.dedup_key,
        severity=finding.severity,
        owner=finding.owner,
        run_id=run_id,
    )


def describe_exception(exc: BaseException) -> str:
    """``TypeName at module.py:123 in func`` — the exception type and where it was raised, never its text."""
    tb = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ is not None else []
    if tb:
        frame = tb[-1]
        where = f"{frame.filename.rsplit('/', 1)[-1]}:{frame.lineno} in {frame.name}"
        return f"{type(exc).__name__} at {where}"
    return type(exc).__name__


def sanitize_error(text) -> str | None:
    """
    A stored error message reduced for an alarm: the first line, cut before
    any ``[SQL:`` / ``[parameters:`` part (database errors echo bound values),
    at most 200 characters.
    """
    if text is None:
        return None
    first = str(text).strip().splitlines()[0] if str(text).strip() else ""
    for marker in ("[SQL:", "[parameters:", "(Background on this error"):
        if marker in first:
            first = first.split(marker, 1)[0].rstrip()
    return first[:200]


def report_internal_error(
    where: str,
    exc: BaseException,
    *,
    key: str | None = None,
    raised_at: datetime.datetime | None = None,
    run_id: int | None = None,
) -> None:
    """
    E7 (Monitoring-Spec §5.E): a monitoring step crashed; the error was caught
    and did not affect any API call. Guarded: if E7 itself cannot be written,
    only the log remains.
    """
    logger.error("Monitoring step %s crashed: %s", where, describe_exception(exc), exc_info=exc)
    try:
        _rollback()
        from app.monitoring.timeutil import monitoring_now

        when = raised_at if raised_at is not None else monitoring_now()
        raise_alarm(
            "E7",
            message=f"A monitoring step crashed ({where}): {type(exc).__name__}. No API call was affected.",
            detail={"where": where, "error": describe_exception(exc)},
            source=catalog.SOURCE_SCHEDULED,
            raised_at=when,
            dedup_key=key,
            run_id=run_id,
        )
    except Exception:  # noqa: BLE001 - last line of defence
        _rollback()
        logger.exception("E7 for %s could not be recorded.", where)


def add_note(event_id: str, note: str, by: str | None = None):
    """
    Add a note on what was done about an alarm (Monitoring-Spec §8.3, §9.1).
    Earlier notes are kept: a new note is appended with its time and author.
    Returns the row; raises ``LookupError`` for an unknown ``event_id``.
    """
    from app.clock import format_detroit, server_now
    from app.models import MonitorAlarm

    text = (note or "").strip()
    if not text:
        raise ValueError("a note must not be empty.")
    row = MonitorAlarm.query.filter_by(event_id=event_id).first()
    if row is None:
        raise LookupError(f"no alarm with event_id {event_id}")
    now = server_now()
    author = (by or "").strip() or "unknown"
    if row.note:
        row.note = f"{row.note}\n[{format_detroit(now, 'seconds')} {author}] {text}"
    else:
        row.note = text
    row.note_by = author[:_MAX_ID]
    row.noted_at = now
    db.session.commit()
    return row


def alarm_feed_record(row) -> dict:
    """One element of ``alarms`` in ``GET /monitor/alarms`` (API-Spec §2.5; Monitoring-Spec §8.1)."""
    from app.clock import format_detroit

    return {
        "event_id": row.event_id,
        "item_id": row.item_id,
        "severity": row.severity,
        "owner": row.owner,
        "raised_at": format_detroit(row.raised_at, "seconds"),
        "message": row.message,
        "detail": row.detail if isinstance(row.detail, dict) else {},
    }


def alarm_full_record(row) -> dict:
    """Every column of an alarm (``flask monitor show``)."""
    from app.clock import format_detroit

    out = {}
    for column in row.__table__.columns:
        value = getattr(row, column.key)
        if isinstance(value, datetime.datetime):
            value = format_detroit(value, "seconds")
        elif isinstance(value, datetime.date):
            value = value.isoformat()
        out[column.key] = value
    return out


def _rollback() -> None:
    try:
        db.session.rollback()
    except Exception:  # noqa: BLE001
        logger.exception("Session rollback failed in monitoring.")


__all__ = [
    "raise_alarm",
    "emit",
    "alarm_exists",
    "add_note",
    "report_internal_error",
    "describe_exception",
    "sanitize_error",
    "alarm_feed_record",
    "alarm_full_record",
]
