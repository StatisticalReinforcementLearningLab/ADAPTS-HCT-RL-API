import datetime
import logging
from flask import Blueprint, current_app, request, jsonify
from app.models import Group, Action, StudyData
from app.extensions import db
from app.protocol import validate_context, validate_decision_type, validate_outcome

data_blueprint = Blueprint("data", __name__)

# Example data field:
example_study_data = {
    "group_id": "example_group_001",
    "decision_idx": 0,
    "decision_type": "aya_message",
    "timestamp": "2024-01-01T12:00:00Z",
    "data": {
        "context": {
            "slot": "am",
            "agent_decision_index": 1,
            "day_in_study": 1,
            "week_in_study": 1,
            "prior_med_adherence": "miss",
            "aya_diary": {"mood": "miss", "physical": "miss"},
            "relationship_quality_cp": "miss",
            "relationship_quality_aya": "miss",
            "aya_app_engagement": 1,
            "aya_app_burden": 0.0,
            "aya_missing_rate_7d": 1.0,
            "current_game_on": 0,
        },
        "action": 1,
        "action_prob": 0.65,
        "state": [1.0],
        "outcome": {
            "med_adherence": 1,
            "prompted_by_message": True,
        },
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

    valid_type, error_message = validate_decision_type(data["decision_type"])
    if not valid_type:
        return False, error_message

    if "timestamp" not in data:
        return False, "timestamp is required."

    if "data" not in data:
        return False, "data is required."

    group_data = data["data"]

    if not group_data or "context" not in group_data:
        return False, "context is required."


    valid_context, error_message = validate_context(data["decision_type"], group_data["context"])
    if not valid_context:
        return False, error_message

    if "action" not in group_data:
        return False, "action is required."

    if "action_prob" not in group_data:
        return False, "action_prob is required."
    
    if "state" not in group_data:
        return False, "state is required."

    if "outcome" not in group_data:
        return False, "outcome is required."

    valid_outcome, error_message = validate_outcome(data["decision_type"], group_data["outcome"])
    if not valid_outcome:
        return False, error_message

    return True, ""


@data_blueprint.route("/upload_data", methods=["POST"])
def upload_data(data: dict | None = None):
    """
    Uploads interaction data for a specific group, along with
    the action sent and the timestamp (and associated metadata).
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

        # Extract the decision index
        decision_idx = data["decision_idx"]

        action_row = Action.query.filter_by(group_id=group_id, decision_idx=decision_idx).first()
        if not action_row:
            return (
                jsonify(
                    {"status": "failed", "message": "Associated action not found for this decision index."}
                ),
                404,
            )

        # Extract the rest of the data
        request_timestamp = data["timestamp"]
        if isinstance(request_timestamp, str):
            request_timestamp = datetime.datetime.fromisoformat(request_timestamp)
        group_data = data["data"]
        decision_type = data["decision_type"]
        context = {**group_data["context"], "decision_type": decision_type}
        action = group_data["action"]
        action_prob = group_data["action_prob"]
        state = group_data["state"]
        outcome = {**group_data["outcome"], "decision_type": decision_type}
        # Get the RL algorithm
        rl_algorithm = current_app.rl_algorithm

        # Create the reward based on the outcome
        status, reward = rl_algorithm.make_reward(group_id, state, action, outcome)

        if not status:
            return jsonify({"status": "failed", "message": "Reward creation failed."}), 400

        existing = StudyData.query.filter_by(group_id=group_id, decision_idx=decision_idx).first()
        if existing is None:
            study_data = StudyData(
                group_id=group_id,
                decision_idx=decision_idx,
                decision_type=decision_type,
                action=action,
                action_prob=action_prob,
                state=state,
                raw_context=context,
                outcome=outcome,
                reward=reward,
                request_timestamp=request_timestamp,
            )
            db.session.add(study_data)
        else:
            existing.action = action
            existing.action_prob = action_prob
            existing.state = state
            existing.raw_context = context
            existing.outcome = outcome
            existing.reward = reward
            existing.request_timestamp = request_timestamp
        db.session.commit()

        # Log the completion
        logging.info(f"[Upload Data] Data uploaded for group: {group_id}")

        return jsonify({"status": "success", "message": "Data uploaded successfully."}), 201

    except Exception as e:
        # Log the error
        logging.error(f"[Upload Data] Error: {e}")
        logging.exception(e)
        return jsonify({"error": "Internal Server Error"}), 500
