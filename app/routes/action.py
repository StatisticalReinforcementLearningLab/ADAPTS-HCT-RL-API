import logging
import datetime
import uuid
from flask import Blueprint, request, jsonify, current_app
from app.extensions import db
from app.models import Group, Action, ModelParameters, DataUpload
from app.protocol import validate_decision_type, project_snapshot

action_blueprint = Blueprint("action", __name__)


def check_fields(data: dict) -> tuple[bool, str]:
    """
    Validate the (context-free) /action request envelope (API-Spec §3.2).

    Context is no longer sent: the API reads the dyad's most recent
    /upload_data snapshot and projects the subset the requested decision_type
    needs.
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

    if not isinstance(data["decision_idx"], int):
        return False, "decision_idx must be an integer."

    if "decision_type" not in data:
        return False, "decision_type is required."

    if not isinstance(data["decision_type"], str):
        return False, "decision_type must be a string."

    return validate_decision_type(data["decision_type"])


def _evaluate_warmup(group_id: str, decision_type: str) -> tuple[bool, str | None]:
    """
    Server-side warm-up gate (API-Spec §3.2): a decision is purely randomized
    iff the cohort has fewer than WARMUP_COHORT_MIN_DYADS registered dyads, or
    this dyad has had fewer than WARMUP_WEEK1_CP_DECISIONS cp_message
    decisions (its first active week — cp_message fires once per active day,
    so its count is a shared day clock for all three agents).
    """
    cohort_min = int(current_app.config.get("WARMUP_COHORT_MIN_DYADS", 5))
    week1_cp = int(current_app.config.get("WARMUP_WEEK1_CP_DECISIONS", 6))

    n_reg = Group.query.count()
    if n_reg < cohort_min:
        return True, "cohort"

    cp_count = Action.query.filter_by(
        group_id=group_id, decision_type="cp_message"
    ).count()
    if cp_count < week1_cp:
        return True, "week1"

    return False, None


def _draw_warmup_action() -> tuple[int, dict]:
    """Bernoulli(0.5) warm-up draw, from the deterministic buffer when the
    active algorithm has one (preserves reproducibility), else a plain draw."""
    sampler = getattr(current_app, "sampler", None)
    if sampler is not None:
        cursor_start = sampler.cursor()
        action = int(sampler.draw_bernoulli(0.5))
        cursor_end = sampler.cursor()
        return action, {
            "mode": "warmup",
            "sampler_cursor_start": cursor_start,
            "sampler_cursor_end": cursor_end,
        }
    import random as _random

    return int(_random.random() < 0.5), {"mode": "warmup"}


@action_blueprint.route("/action", methods=["POST"])
def request_action():
    """
    Request an action for a dyad (API-Spec §3.2). Context is pulled from the
    dyad's latest uploaded snapshot, not the request body.
    """
    try:
        data = request.get_json()

        fields_present, error_message = check_fields(data)
        if not fields_present:
            return jsonify({"status": "failed", "message": error_message}), 400

        group_id = data["group_id"]
        decision_idx = data["decision_idx"]
        decision_type = data["decision_type"]
        request_timestamp = data["timestamp"]
        if isinstance(request_timestamp, str):
            request_timestamp = datetime.datetime.fromisoformat(request_timestamp)
        received_timestamp = datetime.datetime.now()

        # Check if the group exists in the database
        group = Group.query.filter_by(group_id=group_id).first()
        if not group:
            return jsonify({"status": "failed", "message": "Group not found."}), 404

        # Idempotency: (group_id, decision_type, decision_idx) is per-agent.
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

        # Pull the dyad's most recent uploaded snapshot; 409 if none yet.
        latest_upload = (
            DataUpload.query.filter_by(group_id=group_id)
            .order_by(DataUpload.request_timestamp.desc(), DataUpload.id.desc())
            .first()
        )
        if latest_upload is None:
            return (
                jsonify(
                    {
                        "status": "failed",
                        "message": "No /upload_data received for this group yet.",
                    }
                ),
                409,
            )

        # Project the subset this decision_type needs (§5.2). Recorded on the
        # action so the decision is reproducible even if later uploads
        # overwrite individual fields; also seeds warm-up rows into the fit.
        raw_context = project_snapshot(decision_type, latest_upload.data, decision_idx)

        # Get the latest "policy" row (non-snapshot); EB snapshot rows live in
        # the same table and are filtered out.
        model_parameters = (
            ModelParameters.query.filter(ModelParameters.snapshot_type.is_(None))
            .order_by(ModelParameters.timestamp.desc())
            .first()
        )
        if not model_parameters:
            return (
                jsonify({"status": "failed", "message": "Model parameters not found."}),
                404,
            )

        rl_algorithm = current_app.rl_algorithm

        # Server-side warm-up gate.
        is_warmup, warmup_reason = _evaluate_warmup(group_id, decision_type)

        if is_warmup:
            action, random_state = _draw_warmup_action()
            random_state["warmup_reason"] = warmup_reason
            prob = 0.5
            state = None
        else:
            context_with_meta = {
                **raw_context,
                "decision_type": decision_type,
                "group_id": group_id,
            }
            status, state = rl_algorithm.make_state(context_with_meta)
            if not status:
                return jsonify({"status": "failed", "message": state}), 400

            probability = model_parameters.probability_of_action
            action, prob, random_state = rl_algorithm.get_action(
                group_id, state, {"probability": probability}, decision_type, decision_idx
            )

        rid = str(uuid.uuid4())[:8]

        new_action = Action(
            group_id=group_id,
            action=action,
            rid=rid,
            state=state,
            decision_idx=decision_idx,
            decision_type=decision_type,
            raw_context=raw_context,
            action_prob=prob,
            is_warmup=is_warmup,
            warmup_reason=warmup_reason,
            random_state=random_state,
            model_parameters_id=model_parameters.id,
            request_timestamp=request_timestamp,
            timestamp=received_timestamp,
        )

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
                    "warmup": is_warmup,
                    "warmup_reason": warmup_reason,
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
