import logging
from flask import Blueprint, request, jsonify
from app.models import Group
from app.extensions import db

group_blueprint = Blueprint("group", __name__)

# the user in the ADAPTS-HCT study is a group with two participants


def check_fields(data: dict) -> tuple[bool, str]:
    """
    Check if the required fields are present in the data.
    """
    if not data or "group_id" not in data:
        return False, "group_id is required."

    if "member_list" not in data:
        return False, "member_list is required."

    if "consent_start_date" not in data:
        return False, "consent_start_date is required."

    if "consent_end_date" not in data:
        return False, "consent_end_date is required."

    return True, ""


@group_blueprint.route("/register_group", methods=["POST"])
@group_blueprint.route("/add_group", methods=["POST"])  # deprecated alias
def register_group():
    """
    Registers a dyad, or re-registers an existing one (API-Spec §2.1).

    Re-registration is an update, not an error: when the ``group_id`` already
    exists, the consent window (``consent_start_date`` / ``consent_end_date``)
    is overwritten from the request and ``member_list`` is left unchanged. A
    repeat call therefore returns ``201`` (idempotent upsert), not ``400``.

    The canonical path is ``/register_group``; ``/add_group`` is kept as a
    deprecated alias for the existing host contract.

    Warm-up is not a host concern: the API decides it at /action time from the
    cohort size and the dyad's cp_message decision count (§2.2). There is no
    ``warmup`` request field.
    """
    try:
        if request.path.endswith("/add_group"):
            logging.warning(
                "[Group] /add_group is deprecated; use /register_group."
            )

        data = request.get_json()

        # Check if the required fields are present
        fields_present, error_message = check_fields(data)
        if not fields_present:
            return jsonify({"status": "failed", "message": error_message}), 400

        # Extract the data
        group_id = data["group_id"]

        # Re-registration: update the consent window in place, keep member_list.
        existing_group = Group.query.filter_by(group_id=group_id).first()
        if existing_group:
            # Reassign group_info (not in-place mutation) so SQLAlchemy detects
            # the change on this JSON column.
            updated_info = dict(existing_group.group_info)
            updated_info["consent_start_date"] = data["consent_start_date"]
            updated_info["consent_end_date"] = data["consent_end_date"]
            existing_group.group_info = updated_info
            db.session.commit()

            logging.info(f"[Group] Consent window updated: {group_id}")

            return (
                jsonify(
                    {
                        "status": "success",
                        "group_id": group_id,
                        "message": "Group consent window updated.",
                    }
                ),
                201,
            )

        # New registration.
        group_info = {
            "member_list": data["member_list"],
            "consent_start_date": data["consent_start_date"],
            "consent_end_date": data["consent_end_date"],
        }
        new_group = Group(group_id=group_id, group_info=group_info)
        db.session.add(new_group)
        db.session.commit()

        # Log the group addition
        logging.info(f"[Group] Group registered: {group_id}")

        return (
            jsonify(
                {
                    "status": "success",
                    "group_id": group_id,
                    "message": "Group registered successfully.",
                }
            ),
            201,
        )

    except Exception as e:
        logging.error(f"[Group] Error: {e}")
        # Log the stack trace
        logging.exception(e)
        return jsonify({"status": "failed", "message": "Internal server error."}), 500


# Backward-compatible symbol alias for importers of the old handler name.
add_group = register_group
