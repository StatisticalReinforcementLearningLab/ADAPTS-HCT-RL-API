"""
Common response envelope shared by every endpoint (API-Spec §2 preamble).

Every response — success or failure, on every endpoint — carries the same
five always-present fields: ``status`` (int HTTP code, echoed in the body),
``title`` (stable outcome label), ``message`` (human-readable detail),
``server_timestamp`` (ISO-8601 server time), and ``rid`` (the per-call request
id the API assigns on receipt, never null). Each endpoint adds its own fields
on top; on failure a field is null only when the failure makes its value
unknown, and every value the API still knows is echoed.
"""

from __future__ import annotations

import datetime
import uuid

from flask import jsonify


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
        "server_timestamp": datetime.datetime.now().isoformat(),
        "rid": rid,
    }
    body.update(fields)
    return jsonify(body), status
