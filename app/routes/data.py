import datetime
import logging
from flask import Blueprint, request, jsonify
from app.models import Group, DataUpload
from app.extensions import db
from app.protocol import validate_snapshot

data_blueprint = Blueprint("data", __name__)


def check_fields(data: dict) -> tuple[bool, str]:
    """
    Check the required envelope of a /upload_data call (API-Spec §3.3).

    Each upload is a flat full snapshot: there is no context/outcome
    distinction, no decision_type, and no decision_idx — every variable in
    the field dictionary (§5.1) must be present in `data` (use "miss" / null
    to mark an unobservable value).
    """
    if not data or "group_id" not in data:
        return False, "group_id is required."

    if not isinstance(data["group_id"], str):
        return False, "group_id must be a string."

    if "timestamp" not in data:
        return False, "timestamp is required."

    if "data" not in data:
        return False, "data is required."

    return validate_snapshot(data["data"])


@data_blueprint.route("/upload_data", methods=["POST"])
def upload_data(data: dict | None = None):
    """
    Append a full flat snapshot of a dyad's latest values (API-Spec §3.3).

    Append-only: every call writes a new `data_uploads` row. The "current
    value of field X for dyad Y" is `data.X` from the most recent row. /action
    reads the latest row at decision time; /update walks the timeline to
    derive outcomes and rewards.
    """
    try:
        if data is None:
            data = request.get_json()

        # Check if the required fields are present
        fields_present, error_message = check_fields(data)
        if not fields_present:
            return jsonify({"status": "failed", "message": error_message}), 400

        # Extract the group_id
        group_id = data["group_id"]

        # Check if the group exists
        group = Group.query.filter_by(group_id=group_id).first()
        if not group:
            return jsonify({"status": "failed", "message": "Group not found."}), 404

        request_timestamp = data["timestamp"]
        if isinstance(request_timestamp, str):
            request_timestamp = datetime.datetime.fromisoformat(request_timestamp)

        upload = DataUpload(
            group_id=group_id,
            data=data["data"],
            request_timestamp=request_timestamp,
        )
        db.session.add(upload)
        db.session.commit()

        logging.info(f"[Upload Data] Snapshot stored for group: {group_id}")

        return jsonify({"status": "success", "message": "Data uploaded successfully."}), 201

    except Exception as e:
        # Log the error
        logging.error(f"[Upload Data] Error: {e}")
        logging.exception(e)
        return jsonify({"error": "Internal Server Error"}), 500
