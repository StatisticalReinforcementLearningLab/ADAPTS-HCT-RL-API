import logging
from flask import Blueprint, request, jsonify
from app.models import Dyad
from app.extensions import db

dyad_blueprint = Blueprint("dyad", __name__)

# the user in the ADAPTS-HCT study is a dyad with two participants

def check_fields(data: dict) -> tuple[bool, str]:
    """
    Check if the required fields are present in the data.
    """
    if not data or "dyad_id" not in data:
        return False, "dyad_id is required."

    return True, ""


@dyad_blueprint.route("/add_dyad", methods=["POST"])
def add_dyad():
    """
    Adds a new dyad to the database.
    """
    try:
        data = request.get_json()

        # Check if the required fields are present
        fields_present, error_message = check_fields(data)
        if not fields_present:
            return jsonify({"status": "failed", "message": error_message}), 400

        # Extract the data
        dyad_id = data["dyad_id"]

        dyad_info = {
            "cp_id": data["cp_id"],
            "aya_id": data["aya_id"],
            "consent_start_date": data["consent_start_date"],
            "consent_end_date": data["consent_end_date"],
            "AM_MTW": data["AM_MTW"],
            "PM_MTW": data["PM_MTW"],
            "AM_CPW": data["AM_CPW"],
            "PM_CPW": data["PM_CPW"],
        }

        # Check if the user already exists
        existing_dyad = Dyad.query.filter_by(dyad_id=dyad_id).first()
        if existing_dyad:
            return jsonify({"status": "failed", "message": "Dyad already exists."}), 400

        # Add new dyad
        new_dyad = Dyad(dyad_id=dyad_id, dyad_info=dyad_info)
        db.session.add(new_dyad)
        db.session.commit()

        # Log the dyad addition
        logging.info(f"[Dyad] Dyad added: {dyad_id}")

        return jsonify({"status": "success", "dyad_id": dyad_id, "message": "Dyad added successfully."}), 201

    except Exception as e:
        logging.error(f"[Dyad] Error: {e}")
        # Log the stack trace
        logging.exception(e)
        return jsonify({"status": "failed", "message": "Internal server error."}), 500
