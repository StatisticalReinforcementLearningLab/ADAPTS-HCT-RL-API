import datetime
import logging
import uuid
import requests
from threading import Thread
from flask import Blueprint, current_app, request, jsonify
from app.models import Group, ModelParameters, StudyData, ModelUpdateRequests
from app.algorithms.base import RLAlgorithm
from app.extensions import db

data_blueprint = Blueprint("data", __name__)

# Example data field:
example_study_data = {
    "group_id": "example_group_001",
    "decision_idx": 0,
    "decision_type": "aya_message",
    "timestamp": "2024-01-01T12:00:00Z",
    "data": {
        "context": {
            "cur_var": 25,
            "past3_vars": [24.5, 23.0, 22.5]
        },
        "action": 1,
        "action_prob": 0.65,
        "state": [25, 24.5, 23.0, 22.5],
    }
}



def check_fields(data: dict) -> tuple[bool, str]:
    """
    Check if the required fields are present in the data.
    """
    if not data or "group_id" not in data:
        return False, "group_id is required."

    if "decision_idx" not in data:
        return False, "decision_idx is required."
    if "decision_type" not in data:
        return False, "decision_type is required."
    
    if not isinstance(data["decision_type"], str):
        return False, "decision_type must be a string."
    
    if data["decision_type"] not in ["aya_message", "cp_message", "dyad_game"]:
        return False, "Invalid decision_type. Must be 'aya_message', 'cp_message', or 'dyad_game'."
    

    if "timestamp" not in data:
        return False, "timestamp is required."

    if "data" not in data:
        return False, "data is required."

    group_data = data["data"]

    if not group_data or "context" not in group_data:
        return False, "context is required."


    if "cur_var" not in group_data["context"]:
        return False, "Invalid context. cur_var is required."

    if "past3_vars" not in group_data["context"]:
        return False, "Invalid context. past3_vars is required."

    if "action" not in group_data:
        return False, "action is required."

    if "action_prob" not in group_data:
        return False, "action_prob is required."
    
    if "state" not in group_data:
        return False, "state is required."
    return True, ""


# @data_blueprint.route("/upload_data", methods=["POST"])
def upload_data(data: dict):
    """
    Uploads interaction data for a specific group, along with
    the action sent and the timestamp (and associated metadata).
    """
    try:
        # data = request.get_json()

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

        # Extract the decision index
        decision_idx = data["decision_idx"]

        # Check if the decision index already exists
        study_data = StudyData.query.filter_by(
            group_id=group_id, decision_idx=decision_idx
        ).first()
        if study_data:
            return (
                jsonify(
                    {"status": "failed", "message": "Decision index already exists."}
                ),
                400,
            )

        # Extract the rest of the data
        request_timestamp = data["timestamp"]
        group_data = data["data"]
        context = group_data["context"]
        action = group_data["action"]
        action_prob = group_data["action_prob"]
        state = group_data["state"]
        outcome = group_data["outcome"]
        decision_type = data["decision_type"]
        # Get the RL algorithm
        rl_algorithm = current_app.rl_algorithm

        # Create the reward based on the outcome
        status, reward = rl_algorithm.make_reward(group_id, state, action, outcome)

        if not status:
            return jsonify({"status": "failed", "message": "Reward creation failed."}), 400

        # Save the data to the database
        study_data = StudyData(
            group_id=group_id,
            decision_idx=decision_idx,
            action=action,
            action_prob=action_prob,
            state=state,
            raw_context=context,
            outcome=outcome,
            reward=reward,
            request_timestamp=request_timestamp,
        )
        db.session.add(study_data)
        db.session.commit()

        # Log the completion
        logging.info(f"[Upload Data] Data uploaded for group: {group_id}")

        return jsonify({"status": "success", "message": "Data uploaded successfully."}), 201

    except Exception as e:
        # Log the error
        logging.error(f"[Upload Data] Error: {e}")
        logging.exception(e)
        return jsonify({"error": "Internal Server Error"}), 500
