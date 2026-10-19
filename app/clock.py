"""
Study clock for ADAPTS-HCT (implementation contract D4; API-Spec §8; Monitoring-Spec §3, §10).

The study runs in one fixed timezone, ``America/Detroit``, including its
daylight-saving transitions. Conventions:

- A **naive** timestamp (no UTC offset) means Detroit local wall time. This
  holds for host request timestamps and for every study-calendar column in the
  database (``request_timestamp``, ``created_at``, ...), which are stored naive.
- An **aware** timestamp is converted to Detroit local time and then made naive.
- During the autumn fall-back hour a naive Detroit time is ambiguous; it is
  stored as received and rendered with ``fold=0`` (the first occurrence, EDT).

Two clocks exist:

- :func:`server_now` is the server's wall clock in Detroit local time. It
  stamps ``created_at`` columns and the envelope ``server_timestamp``.
- :func:`study_now` is the clock the study calendar logic runs on (monitoring
  loop, ``/monitor/alarms``). In production (``CLOCK_MODE = "wall"``) it is
  :func:`server_now`. In simulation (``CLOCK_MODE = "host"``) the host's
  logical clock drives the study, so it is the latest host ``request_timestamp``
  seen in ``actions``, ``data_uploads`` and ``model_update_requests``
  (contract D4, D20).
"""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

STUDY_TZ_NAME = "America/Detroit"
STUDY_TZ = ZoneInfo(STUDY_TZ_NAME)

CLOCK_MODE_WALL = "wall"
CLOCK_MODE_HOST = "host"
CLOCK_MODES = (CLOCK_MODE_WALL, CLOCK_MODE_HOST)


def to_study_local(value: str | datetime.datetime) -> datetime.datetime:
    """
    Normalize a timestamp to a naive Detroit-local ``datetime`` (contract D4).

    ``value`` may be an ISO-8601 string (``"2026-11-02T03:00:10"``,
    ``"2026-11-02T08:00:10Z"``, ``"2026-11-02T03:00:10-05:00"``, a space in
    place of ``T`` is accepted) or a ``datetime``. Naive input is taken as
    Detroit local time and returned unchanged; aware input is converted. A
    bare ISO date string is local midnight.

    Raises ``ValueError`` for anything that is not a parseable timestamp
    (including numbers, booleans, ``date`` objects and ``None``).
    """
    if isinstance(value, datetime.datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("timestamp is empty.")
        # A bare date ("2026-11-02") parses as local midnight, as the routes
        # accepted before this module existed (the host contract is frozen).
        try:
            dt = datetime.datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"timestamp {value!r} is not ISO-8601.") from exc
    else:
        raise ValueError(
            f"timestamp must be an ISO-8601 string or datetime, got {type(value).__name__}."
        )

    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt
    return dt.astimezone(STUDY_TZ).replace(tzinfo=None)


def localize(dt: datetime.datetime) -> datetime.datetime:
    """Attach the study timezone to a naive Detroit-local datetime (aware input is converted)."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=STUDY_TZ)
    return dt.astimezone(STUDY_TZ)


def to_utc(dt: datetime.datetime) -> datetime.datetime:
    """Aware UTC datetime for a naive Detroit-local (or aware) datetime."""
    return localize(dt).astimezone(datetime.timezone.utc)


def server_now() -> datetime.datetime:
    """The server's wall clock as a naive Detroit-local datetime (contract D4)."""
    return datetime.datetime.now(STUDY_TZ).replace(tzinfo=None)


def format_detroit(dt: datetime.datetime, timespec: str = "auto") -> str:
    """
    ISO-8601 rendering with the Detroit UTC offset, e.g.
    ``2026-11-02T03:00:10-05:00`` (contract D4). Naive input is Detroit local.
    """
    return localize(dt).isoformat(timespec=timespec)


def clock_mode(app=None) -> str:
    """The configured ``CLOCK_MODE`` of ``app`` (default: the current app), ``"wall"`` outside an app."""
    if app is None:
        from flask import current_app, has_app_context

        if not has_app_context():
            return CLOCK_MODE_WALL
        app = current_app
    mode = str(app.config.get("CLOCK_MODE", CLOCK_MODE_WALL) or CLOCK_MODE_WALL).lower()
    return mode if mode in CLOCK_MODES else CLOCK_MODE_WALL


def latest_host_timestamp() -> datetime.datetime | None:
    """
    Latest host-stamped ``request_timestamp`` across ``actions``,
    ``data_uploads`` and ``model_update_requests``; ``None`` when all are
    empty. Requires an application context.
    """
    from sqlalchemy import func

    from app.extensions import db
    from app.models import Action, DataUpload, ModelUpdateRequests

    latest: datetime.datetime | None = None
    for model in (Action, DataUpload, ModelUpdateRequests):
        value = db.session.query(func.max(model.request_timestamp)).scalar()
        if value is not None and (latest is None or value > latest):
            latest = value
    return latest


def study_now() -> datetime.datetime:
    """
    The study's logical "now" (contract D4/D20): the latest host
    ``request_timestamp`` when ``CLOCK_MODE == "host"`` (falling back to
    :func:`server_now` when nothing has been received yet), else
    :func:`server_now`.
    """
    if clock_mode() == CLOCK_MODE_HOST:
        latest = latest_host_timestamp()
        if latest is not None:
            return latest
    return server_now()
