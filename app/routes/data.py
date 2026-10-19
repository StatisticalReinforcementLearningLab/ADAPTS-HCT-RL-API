"""
``POST /api/v1/upload_data`` — a full flat snapshot of a dyad's latest values
(API-Spec §2.3, §4.1; Monitoring-Spec §6.4, §7 F-U1; implementation contract
D4, D18).

Answers (envelope only, no endpoint-specific fields):

- ``201 Success`` ``Snapshot accepted.``
- ``400 Invalid Parameter``: the first failing check, with the exact
  Monitoring-Spec §6.4 / §6.4.1 message. Kinds: ``PARAM`` (top-level field),
  ``UNKNOWN``, ``MISSINGFIELD``, ``TYPE``, ``RANGE``, ``INCONSISTENT`` (code
  ``UD-400-<kind>``, written by the after-response hook, contract D17). Every
  rejected body is preserved verbatim in ``upload_quarantine`` (F-U1, D18) and
  never written to a log.
- ``404 Unknown Group`` ``group_id <id> is not registered.``
- ``503 Service Unavailable`` ``Database write failed; retry.``
- ``500 Internal Error`` ``Internal server error.``
"""

from __future__ import annotations

import hashlib
import logging

from flask import Blueprint, current_app, request

from app.clock import server_now
from app.extensions import db
from app.models import DataUpload, Group, UploadQuarantine
from app.protocol import (
    KIND_PARAM,
    SnapshotIssue,
    check_snapshot,
    check_snapshot_consistency,
)
from app.routes.envelope import (
    envelope,
    is_store_error,
    new_rid,
    parse_timestamp,
    read_json_body,
    timestamp_problem,
)

logger = logging.getLogger(__name__)

data_blueprint = Blueprint("data", __name__)

#: Longest identifier stored from a request (String(255) columns).
_MAX_ID_LEN = 255


def check_request(body) -> SnapshotIssue | None:
    """
    Field-level validation of an ``/upload_data`` body (API-Spec §2.3;
    Monitoring-Spec §6.4). Top-level fields first (``PARAM``: ``group_id is
    required.`` / ``group_id must be a string.`` / ``timestamp is required.`` /
    ``data is required.`` / ``data must be a dictionary.``; a timestamp that does
    not parse is also ``PARAM``), then the snapshot itself
    (:func:`app.protocol.check_snapshot`). Cross-field consistency needs the
    registered dyad and is checked by the route after the group lookup.
    """
    if not isinstance(body, dict) or "group_id" not in body:
        return SnapshotIssue(KIND_PARAM, "group_id", "group_id is required.")
    if not isinstance(body["group_id"], str):
        return SnapshotIssue(KIND_PARAM, "group_id", "group_id must be a string.")
    if "timestamp" not in body:
        return SnapshotIssue(KIND_PARAM, "timestamp", "timestamp is required.")
    problem = timestamp_problem(body["timestamp"])
    if problem is not None:
        return SnapshotIssue(KIND_PARAM, "timestamp", problem)
    if "data" not in body:
        return SnapshotIssue(KIND_PARAM, "data", "data is required.")
    return check_snapshot(body["data"])


def check_fields(data) -> tuple[bool, str]:
    """Compatibility wrapper of :func:`check_request`: ``(ok, message)``."""
    issue = check_request(data)
    return (True, "") if issue is None else (False, issue.message)


def _quarantine(rid: str, raw: bytes, body, issue: SnapshotIssue) -> None:
    """
    F-U1 (contract D18): keep the verbatim rejected body with its rid,
    readable group_id, validation message, fingerprint and size. A failure to
    write is logged (without the body) and does not change the 400 answer.
    """
    group_id = None
    if isinstance(body, dict) and isinstance(body.get("group_id"), str):
        group_id = body["group_id"][:_MAX_ID_LEN]
    try:
        db.session.add(
            UploadQuarantine(
                rid=rid,
                raw_body=raw,
                validation_error=issue.message,
                group_id=group_id,
                payload_sha256=hashlib.sha256(raw).hexdigest(),
                payload_bytes=len(raw),
                received_at=server_now(),
            )
        )
        db.session.commit()
    except Exception as exc:  # noqa: BLE001 - data preservation must not break the answer
        db.session.rollback()
        # Only the exception type: a database error message would carry the
        # bound parameters, i.e. the rejected body.
        logger.error(
            "[Upload Data] Could not quarantine rejected upload rid=%s (%d bytes): %s",
            rid,
            len(raw),
            type(exc).__name__,
        )


def _reject(rid: str, raw: bytes, body, issue: SnapshotIssue):
    """400 with the issue's message, after quarantining the body."""
    _quarantine(rid, raw, body, issue)
    # Kind only: an unknown field's name and the rejected values come from the body.
    logger.info("[Upload Data] Rejected rid=%s kind=%s", rid, issue.kind)
    return envelope(400, "Invalid Parameter", issue.message, rid)


@data_blueprint.route("/upload_data", methods=["POST"])
def upload_data():
    """
    Append a full flat snapshot of a dyad's latest values (API-Spec §2.3).

    Append-only: every accepted call writes a new ``data_uploads`` row with the
    data as posted, the request timestamp in Detroit local time (contract D4)
    and the call's ``rid``. ``/action`` reads the latest row at decision time;
    ``/update`` walks the timeline to derive rewards.
    """
    rid = new_rid()
    try:
        raw = request.get_data(cache=True) or b""
        body = read_json_body()

        issue = check_request(body)
        if issue is not None:
            return _reject(rid, raw, body, issue)

        group_id = body["group_id"]
        request_timestamp = parse_timestamp(body["timestamp"])

        try:
            group = Group.query.filter_by(group_id=group_id).first()
        except Exception as exc:  # noqa: BLE001
            db.session.rollback()
            if is_store_error(exc):
                logger.error("[Upload Data] Group lookup failed: %s", type(exc).__name__)
                return envelope(503, "Service Unavailable", "Database write failed; retry.", rid)
            raise
        if group is None:
            return envelope(404, "Unknown Group", f"group_id {group_id} is not registered.", rid)

        consent_start = group.group_info["consent_start_date"]
        issue = check_snapshot_consistency(
            body["data"],
            consent_start,
            request_timestamp,
            noon_hour=int(current_app.config.get("UPLOAD_SLOT_NOON_HOUR", 12)),
        )
        if issue is not None:
            return _reject(rid, raw, body, issue)

        db.session.add(
            DataUpload(
                group_id=group_id,
                data=body["data"],
                request_timestamp=request_timestamp,
                rid=rid,
            )
        )
        try:
            db.session.commit()
        except Exception as exc:  # noqa: BLE001
            db.session.rollback()
            logger.error("[Upload Data] Commit failed: %s", type(exc).__name__)
            return envelope(503, "Service Unavailable", "Database write failed; retry.", rid)

        logger.info("[Upload Data] Snapshot stored for group: %s rid=%s", group_id, rid)
        return envelope(201, "Success", "Snapshot accepted.", rid)

    except Exception:  # noqa: BLE001 - last resort for this endpoint
        db.session.rollback()
        logger.error("[Upload Data] Unexpected error rid=%s", rid, exc_info=True)
        return envelope(500, "Internal Error", "Internal server error.", rid)
