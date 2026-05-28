import logging
import datetime
import uuid
from flask import Blueprint, request, jsonify, current_app
from app.extensions import db
from app.models import Group, Action, ModelParameters
from app.protocol import validate_context, validate_decision_type

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

    valid_type, error_message = validate_decision_type(data["decision_type"])
    if not valid_type:
        return False, error_message
    
    if not isinstance(data["decision_idx"], int):
        return False, "decision_idx must be an integer."

    if "context" not in data:
        return False, "context is required."

    if not isinstance(data["context"], dict):
        return False, "context must be a dictionary."

    return validate_context(data["decision_type"], data["context"])


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
        if isinstance(request_timestamp, str):
            request_timestamp = datetime.datetime.fromisoformat(request_timestamp)
        received_timestamp = datetime.datetime.now()

        # Check if the group exists in the database
        group = Group.query.filter_by(group_id=group_id).first()
        if not group:
            return jsonify({"status": "failed", "message": "Group not found."}), 404

        # Check that (group_id, decision_type, decision_idx) is not already used.
        # Idempotency key is per-(dyad, decision_type): the three agents have
        # independent decision counters.
        action_row = Action.query.filter_by(
            group_id=group_id, decision_type=decision_type, decision_idx=decision_idx
        ).first()
        if action_row:
            return (
                jsonify(
                    {
                        "status": "failed",
                        "message": "Decision index already exists for this (group, decision_type).",
                    }
                ),
                400,
            )

        # Get the RL algorithm
        rl_algorithm = current_app.rl_algorithm

        # Make the state. The decision_type and group_id are passed alongside
        # the schema fields so the algorithm can look up per-dyad
        # standardization baselines (main.tex §3) before building features.
        context_with_type = {
            **context,
            "decision_type": decision_type,
            "group_id": group_id,
        }
        status, state = rl_algorithm.make_state(context_with_type)
        if not status:
            return jsonify({"status": "failed", "message": state}), 400

        # Get the latest "policy" row (the bootstrap / non-snapshot row).
        # Empirical-Bayes snapshot rows live in the same table; filter them
        # out so the action FK keeps pointing at the policy config row.
        model_parameters = (
            ModelParameters.query.filter(ModelParameters.snapshot_type.is_(None))
            .order_by(ModelParameters.timestamp.desc())
            .first()
        )

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
            timestamp=received_timestamp,
        )

        # Save the action to the database
        db.session.add(new_action)
        db.session.commit()

        return (
            jsonify(
                {
                    "status": "success",
                    "message": "Action requested successfully.",
                    "group_id": group_id,
                    "state": state,
                    "action": action,
                    "action_prob": prob,
                    "timestamp": received_timestamp.isoformat(),
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
