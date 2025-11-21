import logging
import datetime
import uuid
from flask import Blueprint, request, jsonify, current_app
from app.extensions import db
from app.models import Group, Action, ModelParameters, StudyData
from app.routes.data import upload_data

action_blueprint = Blueprint("action", __name__)


def check_fields(data: dict) -> tuple[bool, str]:
    """
    Check if the required fields are present in the data.
    """
    if not data or "group_id" not in data or "timestamp" not in data:
        return False, "group_id and timestamp are required."

    if not isinstance(data["group_id"], str):
        return False, "group_id must be a string."

    if not isinstance(data["timestamp"], str) and not isinstance(
        data["timestamp"], datetime.datetime
    ):
        return False, "timestamp must be a string or datetime object."

    if "decision_idx" not in data:
        return False, "decision_idx is required."
    
    if "decision_type" not in data:
        return False, "decision_type is required."
    
    if not isinstance(data["decision_type"], str):
        return False, "decision_type must be a string."
    
    if data["decision_type"] not in ["aya_message", "cp_message", "dyad_game"]:
        return False, "Invalid decision_type. Must be 'aya_message', 'cp_message', or 'dyad_game'."
    
    if not isinstance(data["decision_idx"], int):
        return False, "decision_idx must be an integer."

    if "context" not in data:
        return False, "context is required."

    if not isinstance(data["context"], dict):
        return False, "context must be a dictionary."
    
    if "cur_var" not in data["context"]:
        return False, "Invalid context. cur_var is required."
    
    if "past3_vars" not in data["context"]:
        return False, "Invalid context. past3_vars is required."

    return True, ""


@action_blueprint.route("/action", methods=["POST"])
def request_action():
    """
    Requests an action for a specific group based on context.
    """
    try:
        data = request.get_json()

        # Check if the required fields are present
        fields_present, error_message = check_fields(data)
        if not fields_present:
            return jsonify({"status": "failed", "message": error_message}), 400

        # Extract the data
        group_id = data["group_id"]
        decision_idx = data["decision_idx"] # RL won't use. For checking or validation
        context = data["context"]
        decision_type = data["decision_type"]
        request_timestamp = data["timestamp"]
        received_timestamp_iso = datetime.datetime.now().isoformat()

        # Check if the group exists in the database
        group = Group.query.filter_by(group_id=group_id).first()
        if not group:
            return jsonify({"status": "failed", "message": "Group not found."}), 404

        # Check if decision_idx does not exist in the study data for the group
        study_data = StudyData.query.filter_by(
            group_id=group_id, decision_idx=decision_idx
        ).first()
        if study_data:
            return (
                jsonify(
                    {"status": "failed", "message": "Decision index already exists for this group."}
                ),
                400,
            )

        # Get the RL algorithm
        rl_algorithm = current_app.rl_algorithm

        # Make the state
        status, state = rl_algorithm.make_state(context)
        if not status:
            return jsonify({"status": "failed", "message": state}), 400

        # Get the latest model parameters from the database
        model_parameters = ModelParameters.query.order_by(
            ModelParameters.timestamp.desc()
        ).first()

        # Check if the model parameters exist
        if not model_parameters:
            return (
                jsonify({"status": "failed", "message": "Model parameters not found."}),
                404,
            )

        # Extract the model parameters, in this case, the probability
        probability = model_parameters.probability_of_action

        # Get the action, action selection probability, and random state
        # used to generate the action
        action, prob, random_state = rl_algorithm.get_action(
            group_id, state, {"probability": probability}, decision_type, decision_idx
        )

        rid = str(uuid.uuid4())[:8]

        # Save the action to the action database
        new_action = Action(
            group_id=group_id,
            action=action,
            rid=rid,
            state=state,
            decision_idx=decision_idx,
            decision_type=decision_type,
            raw_context=context,
            action_prob=prob,
            random_state=random_state,
            model_parameters_id=model_parameters.id,
            request_timestamp=request_timestamp,
            timestamp=received_timestamp_iso,
        )

        # Save the action to the database
        db.session.add(new_action)
        db.session.commit()

        # Upload the data to the database
        study_data = {
            "group_id": group_id,
            "decision_idx": decision_idx,
            "decision_type": decision_type,
            "timestamp": received_timestamp_iso,
            "data": {
                "context": context,
                "action": action,
                "action_prob": prob,
                "state": state,
            },
        }
        status, message = upload_data(study_data)
        if not status:
            return jsonify({"status": "failed", "message": message}), 500

        return (
            jsonify(
                {
                    "status": "success",
                    "message": "Action requested successfully.",
                    "group_id": group_id,
                    "state": state,
                    "action": action,
                    "action_prob": prob,
                    "timestamp": received_timestamp_iso,
                    "rid": rid,
                }
            ),
            201,
        )

    except Exception as e:
        # Log the exception
        logging.error(f"[Action] Error: {e}")
        logging.exception(e)
        return jsonify({"status": "failed", "message": "Internal server error."}), 500
