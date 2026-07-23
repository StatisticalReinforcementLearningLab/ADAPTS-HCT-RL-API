import datetime
import logging
from flask import Blueprint, request
from app.models import Group
from app.extensions import db
from app.routes.envelope import envelope, new_rid

group_blueprint = Blueprint("group", __name__)

# the user in the ADAPTS-HCT study is a group with two participants

_DATE_FIELDS = ("consent_start_date", "consent_end_date")


def check_fields(data: dict) -> tuple[bool, str, str | None]:
    """
    Validate the /register_group request (API-Spec §2.1).

    Returns ``(ok, message, offending_field)`` — ``offending_field`` names the
    single malformed field (echoed as ``null`` per the §2 null-discipline) or
    is ``None`` for whole-body problems.

    Date fields are checked for ``YYYY-MM-DD`` format only. The spec's
    "consent_start_date must be a Monday" rule is documented but not enforced
    here (the bundled simulator recruits on non-Mondays); see API-Spec §2.1.
    """
    if data is None:
        return False, "Request body is required.", None
    if "group_id" not in data:
        return False, "group_id is required.", "group_id"
    if "member_list" not in data:
        return False, "member_list is required.", "member_list"
    for field in _DATE_FIELDS:
        if field not in data:
            return False, f"{field} is required.", field
        try:
            datetime.date.fromisoformat(str(data[field]))
        except (ValueError, TypeError):
            return (
                False,
                f"{field} must be a date in YYYY-MM-DD format; got {data[field]!r}.",
                field,
            )
    return True, "", None


def _echo(data: dict | None, null_field: str | None = None) -> dict:
    """Echo the parsed request fields, nulling the one that caused the failure."""
    data = data or {}
    echo = {
        "group_id": data.get("group_id"),
        "member_list": data.get("member_list"),
        "consent_start_date": data.get("consent_start_date"),
        "consent_end_date": data.get("consent_end_date"),
    }
    if null_field in echo:
        echo[null_field] = None
    return echo


@group_blueprint.route("/register_group", methods=["POST"])
@group_blueprint.route("/add_group", methods=["POST"])  # deprecated alias
def register_group():
    """
    Registers a dyad, or re-registers an existing one (API-Spec §2.1).

    Re-registration is an idempotent upsert of the consent window only: the
    ``consent_start_date`` / ``consent_end_date`` are overwritten from the
    request (returns ``201``). A re-registration whose ``member_list`` differs
    from the recorded members is rejected with ``409 Member Mismatch`` rather
    than silently ignored.

    The canonical path is ``/register_group``; ``/add_group`` is a deprecated
    alias. Warm-up is decided server-side at /action time, not here.
    """
    rid = new_rid()
    try:
        if request.path.endswith("/add_group"):
            logging.warning("[Group] /add_group is deprecated; use /register_group.")

        data = request.get_json(silent=True)

        ok, message, field = check_fields(data)
        if not ok:
            return envelope(400, "Invalid Parameter", message, rid, **_echo(data, field))

        group_id = data["group_id"]

        existing_group = Group.query.filter_by(group_id=group_id).first()
        if existing_group:
            recorded_members = (existing_group.group_info or {}).get("member_list")
            if data["member_list"] != recorded_members:
                return envelope(
                    409,
                    "Member Mismatch",
                    f"group_id {group_id} is already registered with members "
                    f"{recorded_members}; membership cannot be changed via "
                    f"/register_group.",
                    rid,
                    **_echo(data),
                )

            # Reassign group_info (not in-place mutation) so SQLAlchemy detects
            # the change on this JSON column.
            updated_info = dict(existing_group.group_info)
            updated_info["consent_start_date"] = data["consent_start_date"]
            updated_info["consent_end_date"] = data["consent_end_date"]
            existing_group.group_info = updated_info
            existing_group.rid = rid
            try:
                db.session.commit()
            except Exception as exc:
                db.session.rollback()
                logging.exception(exc)
                return envelope(
                    503,
                    "Service Unavailable",
                    "Database write failed; retry with backoff.",
                    rid,
                    **_echo(data),
                )

            logging.info(f"[Group] Consent window updated: {group_id}")
            return envelope(
                201,
                "Success",
                "Group consent window updated.",
                rid,
                **_echo(data),
            )

        # New registration.
        group_info = {
            "member_list": data["member_list"],
            "consent_start_date": data["consent_start_date"],
            "consent_end_date": data["consent_end_date"],
        }
        new_group = Group(group_id=group_id, group_info=group_info, rid=rid)
        db.session.add(new_group)
        try:
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logging.exception(exc)
            return envelope(
                503,
                "Service Unavailable",
                "Database write failed; retry with backoff.",
                rid,
                **_echo(data),
            )

        logging.info(f"[Group] Group registered: {group_id}")
        return envelope(
            201,
            "Success",
            "Group registered successfully.",
            rid,
            **_echo(data),
        )

    except Exception as e:
        logging.error(f"[Group] Error: {e}")
        logging.exception(e)
        return envelope(500, "Internal Error", "Internal server error.", rid)


# Backward-compatible symbol alias for importers of the old handler name.
add_group = register_group
