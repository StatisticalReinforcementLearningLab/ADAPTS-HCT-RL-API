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


@group_blueprint.route("/add_group", methods=["POST"])
def add_group():
    """
    Adds a new group to the database.
    """
    try:
        data = request.get_json()

        # Check if the required fields are present
        fields_present, error_message = check_fields(data)
        if not fields_present:
            return jsonify({"status": "failed", "message": error_message}), 400

        # Extract the data
        group_id = data["group_id"]

        group_info = {
            "member_list": data["member_list"],
            "consent_start_date": data["consent_start_date"],
            "consent_end_date": data["consent_end_date"],
        }

        # Check if the user already exists
        existing_group = Group.query.filter_by(group_id=group_id).first()
        if existing_group:
            return jsonify({"status": "failed", "message": "Group already exists."}), 400

        # Add new group
        new_group = Group(group_id=group_id, group_info=group_info)
        db.session.add(new_group)
        db.session.commit()

        # Log the group addition
        logging.info(f"[Group] Group added: {group_id}")

        return jsonify({"status": "success", "group_id": group_id, "message": "Group added successfully."}), 201

    except Exception as e:
        logging.error(f"[Group] Error: {e}")
        # Log the stack trace
        logging.exception(e)
        return jsonify({"status": "failed", "message": "Internal server error."}), 500

