"""
``POST /api/v1/register_group`` (deprecated alias ``/api/v1/add_group``) —
register a dyad or update its consent window (API-Spec §2.1; Monitoring-Spec
§6.2, §9.6).

Answers (common envelope plus ``group_id``, ``member_list``,
``consent_start_date``, ``consent_end_date``):

- ``201 Success`` ``Group registered successfully.`` (new dyad) or
  ``Group consent window updated.`` (re-registration with the same members;
  the consent dates are overwritten, the first-registration members stand).
- ``400 Invalid Parameter``: ``Request body is required.`` /
  ``<field> is required.`` / ``<date field> must be a date in YYYY-MM-DD
  format; got <value>.``; the offending field is echoed as ``null``.
- ``409 Member Mismatch``: re-registration with different members; the
  submitted ``member_list`` is echoed.
- ``503 Service Unavailable`` ``Database write failed; retry with backoff.``
- ``500 Internal Error`` ``Internal server error.``

Every successful call writes the ``groups`` row (with this call's ``rid``) and
appends one ``group_history`` row in the same transaction (Monitoring-Spec
§9.6), so monitoring can read consent dates as they were at any time.
"""

from __future__ import annotations

import datetime
import json
import logging
import re

from flask import Blueprint, request

from app.extensions import db
from app.models import Group, GroupHistory
from app.routes.envelope import envelope, is_store_error, new_rid, read_json_body

logger = logging.getLogger(__name__)

group_blueprint = Blueprint("group", __name__)

# the user in the ADAPTS-HCT study is a group with two participants

_DATE_FIELDS = ("consent_start_date", "consent_end_date")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_date(value) -> datetime.date | None:
    """Strict ``YYYY-MM-DD`` (API-Spec §2.1); ``None`` when ``value`` is not such a date."""
    if not isinstance(value, str) or not _DATE_RE.match(value):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def check_fields(data) -> tuple[bool, str, str | None]:
    """
    Validate the /register_group request (API-Spec §2.1; Monitoring-Spec §6.2).

    Returns ``(ok, message, offending_field)``. ``offending_field`` names the
    single malformed field (echoed as ``null``) or is ``None`` for whole-body
    problems. ``consent_start_date`` may be any day of the week.
    """
    if not isinstance(data, dict):
        return False, "Request body is required.", None
    if "group_id" not in data:
        return False, "group_id is required.", "group_id"
    if not isinstance(data["group_id"], str):
        return False, "group_id must be a string.", "group_id"
    if "member_list" not in data:
        return False, "member_list is required.", "member_list"
    if not isinstance(data["member_list"], list):
        return False, "member_list must be a list.", "member_list"
    for field in _DATE_FIELDS:
        if field not in data:
            return False, f"{field} is required.", field
        if _parse_date(data[field]) is None:
            return (
                False,
                f"{field} must be a date in YYYY-MM-DD format; got {data[field]!r}.",
                field,
            )
    return True, "", None


def _echo(data, null_field: str | None = None) -> dict:
    """Echo the request fields, nulling the one that caused the failure."""
    data = data if isinstance(data, dict) else {}
    echo = {
        "group_id": data.get("group_id"),
        "member_list": data.get("member_list"),
        "consent_start_date": data.get("consent_start_date"),
        "consent_end_date": data.get("consent_end_date"),
    }
    if null_field in echo:
        echo[null_field] = None
    return echo


def _member_key(members) -> list[str]:
    """Order-insensitive comparison key for a member list (a repeat in another order matches)."""
    return sorted(json.dumps(m, sort_keys=True) for m in (members or []))


def _format_members(members) -> str:
    """``[cp_007, aya_007]`` as in the API-Spec §2.1 409 example."""
    return "[" + ", ".join(str(m) for m in (members or [])) + "]"


def _db_unavailable(rid: str, data):
    return envelope(
        503,
        "Service Unavailable",
        "Database write failed; retry with backoff.",
        rid,
        **_echo(data),
    )


@group_blueprint.route("/register_group", methods=["POST"])
@group_blueprint.route("/add_group", methods=["POST"])  # deprecated alias
def register_group():
    """
    Registers a dyad, or re-registers an existing one (API-Spec §2.1).

    Re-registration is an idempotent upsert of the consent window only: the
    ``consent_start_date`` / ``consent_end_date`` are overwritten from the
    request (``201``). A re-registration whose ``member_list`` differs from the
    recorded members (as a multiset; order does not matter) is rejected with
    ``409 Member Mismatch`` rather than silently ignored. Warm-up is decided
    server-side at /action time, not here.
    """
    rid = new_rid()
    data = None
    try:
        if request.path.endswith("/add_group"):
            logger.warning("[Group] /add_group is deprecated; use /register_group.")

        data = read_json_body()

        ok, message, field = check_fields(data)
        if not ok:
            return envelope(400, "Invalid Parameter", message, rid, **_echo(data, field))

        group_id = data["group_id"]

        try:
            existing_group = Group.query.filter_by(group_id=group_id).first()
        except Exception as exc:  # noqa: BLE001
            db.session.rollback()
            if is_store_error(exc):
                logger.error("[Group] Lookup failed: %s", type(exc).__name__)
                return _db_unavailable(rid, data)
            raise

        if existing_group is not None:
            recorded_members = (existing_group.group_info or {}).get("member_list")
            if _member_key(data["member_list"]) != _member_key(recorded_members):
                return envelope(
                    409,
                    "Member Mismatch",
                    f"group_id {group_id} is already registered with members "
                    f"{_format_members(recorded_members)}; membership cannot be "
                    f"changed via /register_group.",
                    rid,
                    **_echo(data),
                )
            # Reassign group_info (not in-place mutation) so SQLAlchemy detects
            # the change on this JSON column. The recorded members stand.
            updated_info = dict(existing_group.group_info)
            updated_info["consent_start_date"] = data["consent_start_date"]
            updated_info["consent_end_date"] = data["consent_end_date"]
            existing_group.group_info = updated_info
            existing_group.rid = rid
            members = recorded_members
            success_message = "Group consent window updated."
        else:
            members = data["member_list"]
            db.session.add(
                Group(
                    group_id=group_id,
                    group_info={
                        "member_list": members,
                        "consent_start_date": data["consent_start_date"],
                        "consent_end_date": data["consent_end_date"],
                    },
                    rid=rid,
                )
            )
            success_message = "Group registered successfully."

        db.session.add(
            GroupHistory(
                group_id=group_id,
                member_list=members,
                consent_start_date=data["consent_start_date"],
                consent_end_date=data["consent_end_date"],
                rid=rid,
            )
        )
        try:
            db.session.commit()
        except Exception as exc:  # noqa: BLE001
            db.session.rollback()
            logger.error("[Group] Commit failed: %s", type(exc).__name__)
            return _db_unavailable(rid, data)

        logger.info("[Group] %s %s rid=%s", success_message, group_id, rid)
        # member_list echoes the members recorded at first registration (§2.1).
        return envelope(201, "Success", success_message, rid, **{**_echo(data), "member_list": members})

    except Exception:  # noqa: BLE001 - last resort for this endpoint
        db.session.rollback()
        logger.error("[Group] Unexpected error rid=%s", rid, exc_info=True)
        return envelope(500, "Internal Error", "Internal server error.", rid, **_echo(data))


# Backward-compatible symbol alias for importers of the old handler name.
add_group = register_group
