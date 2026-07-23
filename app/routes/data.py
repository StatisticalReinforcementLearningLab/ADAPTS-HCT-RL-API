import datetime
import logging
from flask import Blueprint, request
from app.models import Group, DataUpload
from app.extensions import db
from app.protocol import validate_snapshot
from app.routes.envelope import envelope, new_rid

data_blueprint = Blueprint("data", __name__)


def check_fields(data: dict) -> tuple[bool, str]:
    """
    Check the required envelope of a /upload_data call (API-Spec §2.3).

    Each upload is a flat full snapshot: there is no context/outcome
    distinction, no decision_type, and no decision_idx — every variable in
    the field dictionary (§4.1) must be present in `data` (use "miss" / null
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
    Append a full flat snapshot of a dyad's latest values (API-Spec §2.3).

    Append-only: every call writes a new `data_uploads` row. The "current
    value of field X for dyad Y" is `data.X` from the most recent row. /action
    reads the latest row at decision time; /update walks the timeline to
    derive outcomes and rewards. The response is the common envelope only (the
    always-present `rid` is persisted on the row).
    """
    rid = new_rid()
    try:
        if data is None:
            data = request.get_json(silent=True)

        ok, error_message = check_fields(data)
        if not ok:
            return envelope(400, "Invalid Parameter", error_message, rid)

        group_id = data["group_id"]

        group = Group.query.filter_by(group_id=group_id).first()
        if not group:
            return envelope(
                404, "Unknown Group",
                f"group_id {group_id} is not registered.", rid,
            )

        request_timestamp = data["timestamp"]
        if isinstance(request_timestamp, str):
            request_timestamp = datetime.datetime.fromisoformat(request_timestamp)

        upload = DataUpload(
            group_id=group_id,
            data=data["data"],
            request_timestamp=request_timestamp,
            rid=rid,
        )
        db.session.add(upload)
        try:
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logging.exception(exc)
            return envelope(
                503, "Service Unavailable",
                "Database write failed; retry.", rid,
            )

        logging.info(f"[Upload Data] Snapshot stored for group: {group_id}")
        return envelope(201, "Success", "Snapshot accepted.", rid)

    except Exception as e:
        logging.error(f"[Upload Data] Error: {e}")
        logging.exception(e)
        return envelope(500, "Internal Error", "Internal server error.", rid)
