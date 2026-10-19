"""
Common response envelope shared by every endpoint (API-Spec §2 preamble;
implementation contract §3.2).

Every response — success or failure, on every endpoint — carries the same
five always-present fields: ``status`` (int HTTP code, echoed in the body),
``title`` (stable outcome label), ``message`` (human-readable detail),
``server_timestamp`` (ISO-8601 server time, Detroit local; ``app.clock``), and
``rid`` (the per-call request id the API assigns on receipt, never null). Each
endpoint adds its own fields on top; on failure a field is null only when the
failure makes its value unknown, and every value the API still knows is echoed.

Helpers shared by the routes also live here: :func:`parse_timestamp` (contract
D4), :func:`read_json_body`, :func:`call_monitoring` (Monitoring-Spec §8 hooks
are called only through a wrapper that logs and continues) and
:func:`is_store_error` (contract D12: a database that cannot be read or written).
"""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import Any

from flask import jsonify, request
from sqlalchemy.exc import DBAPIError, OperationalError

from app.clock import server_now, to_study_local

logger = logging.getLogger(__name__)

#: Message for a ``timestamp`` that is not a string (Monitoring-Spec §6.3).
TIMESTAMP_TYPE_MESSAGE = "timestamp must be a string or datetime object."


def new_rid() -> str:
    """Fresh per-call request id. Assigned on receipt, before any write."""
    return uuid.uuid4().hex


def envelope(
    status: int,
    title: str,
    message: str,
    rid: str,
    **fields,
):
    """
    Build the common envelope plus any endpoint-specific ``fields`` and return
    a ``(response, status)`` tuple. The HTTP status equals the ``status`` field
    in the body so a client may branch on either.
    """
    body = {
        "status": status,
        "title": title,
        "message": message,
        "server_timestamp": server_now().isoformat(),
        "rid": rid,
    }
    body.update(fields)
    return jsonify(body), status


def read_json_body() -> Any:
    """The parsed JSON request body, or ``None`` for an empty or non-JSON body (never raises)."""
    return request.get_json(silent=True)


def timestamp_problem(value: Any) -> str | None:
    """
    Validation message for a request ``timestamp`` value, or ``None`` when it
    parses (contract D4). A non-string is ``timestamp must be a string or
    datetime object.`` (Monitoring-Spec §6.3); a string that is not an ISO-8601
    timestamp is ``timestamp must be an ISO-8601 timestamp; got '<value>'.``.
    """
    if not isinstance(value, (str, datetime.datetime)):
        return TIMESTAMP_TYPE_MESSAGE
    try:
        to_study_local(value)
    except ValueError:
        return f"timestamp must be an ISO-8601 timestamp; got {value!r}."
    return None


def parse_timestamp(value: Any) -> datetime.datetime:
    """Naive Detroit-local datetime of a request ``timestamp`` (contract D4); raises ``ValueError``."""
    return to_study_local(value)


def is_store_error(exc: BaseException) -> bool:
    """
    ``True`` when ``exc`` means the database cannot be read or written
    (contract D12 ``AC-503-STORE``; API-Spec §2.1/§2.3 ``503``): an
    ``OperationalError`` or a driver-level error with the connection lost.
    """
    if isinstance(exc, OperationalError):
        return True
    return isinstance(exc, DBAPIError) and bool(getattr(exc, "connection_invalidated", False))


def call_monitoring(hook_name: str, *args, default=None, **kwargs):
    """
    Call ``app.monitoring.<hook_name>(*args, **kwargs)`` (implementation
    contract §3.1) inside ``try/except``: a monitoring failure is logged and
    never changes the API's answer (Monitoring-Spec §6.9 rule 1). Returns the
    hook's result, or ``default`` when it raised.
    """
    try:
        from app import monitoring

        hook = getattr(monitoring, hook_name)
        return hook(*args, **kwargs)
    except Exception:  # noqa: BLE001 - monitoring must never break a call
        logger.exception("Monitoring hook %s failed; continuing.", hook_name)
        return default
