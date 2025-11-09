import datetime
import logging
import uuid
import requests
from threading import Thread
from flask import Blueprint, current_app, request, jsonify
from app.models import Dyad, ModelParameters, StudyData, ModelUpdateRequests
from app.algorithms.base import RLAlgorithm
from app.extensions import db

data_blueprint = Blueprint("data", __name__)


def check_fields(data: dict) -> tuple[bool, str]:
    """
    Check if the required fields are present in the data.
    """
    if not data or "dyad_id" not in data:
        return False, "dyad_id is required."

    if "decision_idx" not in data:
        return False, "decision_idx is required."

    if "timestamp" not in data:
        return False, "timestamp is required."

    if "data" not in data:
        return False, "data is required."

    dyad_data = data["data"]

    if not dyad_data or "context" not in dyad_data:
        return False, "context is required."


    if "cur_var" not in dyad_data["context"]:
        return False, "Invalid context. cur_var is required."

    if "past3_vars" not in dyad_data["context"]:
        return False, "Invalid context. past3_vars is required."

    if "action" not in dyad_data:
        return False, "action is required."

    if "action_prob" not in dyad_data:
        return False, "action_prob is required."
    
    if "state" not in dyad_data:
        return False, "state is required."

    if "outcome" not in dyad_data:
        return False, "outcome is required."

    if "clicks" not in dyad_data["outcome"]:
        return False, "Invalid outcome. Clicks is required."

    return True, ""


@data_blueprint.route("/upload_data", methods=["POST"])
def upload_data():
    """
    Uploads interaction data for a specific dyad, along with
    the action sent and the timestamp (and associated metadata).
    """
    try:
        data = request.get_json()

        # Check if the required fields are present
        fields_present, error_message = check_fields(data)
        if not fields_present:
            return jsonify({"status": "failed", "message": error_message}), 400

        # Extract the dyad_id
        dyad_id = data["dyad_id"]

        # Check if the dyad exists
        dyad = Dyad.query.filter_by(dyad_id=dyad_id).first()
        if not dyad:
            return jsonify({"status": "failed", "message": "Dyad not found."}), 404

        # Extract the decision index
        decision_idx = data["decision_idx"]

        # Check if the decision index already exists
        study_data = StudyData.query.filter_by(
            dyad_id=dyad_id, decision_idx=decision_idx
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
        dyad_data = data["data"]
        context = dyad_data["context"]
        action = dyad_data["action"]
        action_prob = dyad_data["action_prob"]
        state = dyad_data["state"]
        outcome = dyad_data["outcome"]

        # Get the RL algorithm
        rl_algorithm = current_app.rl_algorithm

        # Create the reward based on the outcome
        status, reward = rl_algorithm.make_reward(dyad_id, state, action, outcome)

        if not status:
            return jsonify({"status": "failed", "message": "Reward creation failed."}), 400

        # Save the data to the database
        study_data = StudyData(
            dyad_id=dyad_id,
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
        logging.info(f"[Upload Data] Data uploaded for dyad: {dyad_id}")

        return jsonify({"status": "success", "message": "Data uploaded successfully."}), 201

    except Exception as e:
        # Log the error
        logging.error(f"[Upload Data] Error: {e}")
        logging.exception(e)
        return jsonify({"error": "Internal Server Error"}), 500
