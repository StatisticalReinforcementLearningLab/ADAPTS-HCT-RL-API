import logging
import datetime
from flask import Blueprint, request, current_app
from app.extensions import db
from app.models import Group, Action, ModelParameters, DataUpload
from app.protocol import validate_decision_type, project_snapshot
from app.routes.envelope import envelope, new_rid

action_blueprint = Blueprint("action", __name__)


def check_fields(data: dict) -> tuple[bool, str, str | None]:
    """
    Validate the (context-free) /action request envelope (API-Spec §2.2).

    Returns ``(ok, message, offending_field)`` — the offending field is echoed
    as ``null`` per the §2 null-discipline. Context is not sent: the API reads
    the dyad's most recent /upload_data snapshot and projects the subset the
    requested decision_type needs.
    """
    if data is None:
        return False, "Request body is required.", None

    if "group_id" not in data:
        return False, "group_id is required.", "group_id"
    if not isinstance(data["group_id"], str):
        return False, "group_id must be a string.", "group_id"

    if "timestamp" not in data:
        return False, "timestamp is required.", "timestamp"
    if not isinstance(data["timestamp"], (str, datetime.datetime)):
        return False, "timestamp must be a string or datetime object.", "timestamp"

    if "decision_idx" not in data:
        return False, "decision_idx is required.", "decision_idx"
    if not isinstance(data["decision_idx"], int) or isinstance(data["decision_idx"], bool):
        return False, "decision_idx must be an integer.", "decision_idx"

    if "decision_type" not in data:
        return False, "decision_type is required.", "decision_type"
    if not isinstance(data["decision_type"], str):
        return False, "decision_type must be a string.", "decision_type"

    ok, message = validate_decision_type(data["decision_type"])
    if not ok:
        return False, message, "decision_type"
    return True, "", None


def _echo(data: dict | None, null_field: str | None = None) -> dict:
    """Echo the parsed request identifiers, nulling the failure's cause."""
    data = data or {}
    echo = {
        "group_id": data.get("group_id"),
        "decision_type": data.get("decision_type"),
        "decision_idx": data.get("decision_idx"),
    }
    if null_field in echo:
        echo[null_field] = None
    return echo


def _null_outputs() -> dict:
    """The un-producible decision outputs, all null (used on every failure)."""
    return {
        "action": None,
        "action_prob": None,
        "warmup": None,
        "state": None,
        "model_theta": None,
        "model_cov": None,
        "eta": None,
    }


def _evaluate_warmup(group_id: str, decision_type: str) -> tuple[bool, str | None]:
    """
    Server-side warm-up gate (API-Spec §2.2): a decision is purely randomized
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


def _replay(action_row: Action):
    """
    Idempotent replay of an already-committed decision (API-Spec §2.2). Returns
    the originally minted outputs verbatim with the original rid; a lost
    response can never desync host and server.
    """
    dp = action_row.decision_params or {}
    return envelope(
        200,
        "Duplicate Decision",
        f"Returning the existing action for ("
        f"{action_row.group_id}, {action_row.decision_type}, "
        f"{action_row.decision_idx}).",
        action_row.rid,
        group_id=action_row.group_id,
        decision_type=action_row.decision_type,
        decision_idx=action_row.decision_idx,
        action=action_row.action,
        action_prob=action_row.action_prob,
        warmup=bool(action_row.is_warmup),
        state=action_row.state,
        model_theta=dp.get("theta"),
        model_cov=dp.get("cov"),
        eta=dp.get("eta"),
    )


@action_blueprint.route("/action", methods=["POST"])
def request_action():
    """
    Request an action for a dyad (API-Spec §2.2). Context is pulled from the
    dyad's latest uploaded snapshot, not the request body.
    """
    rid = new_rid()
    try:
        data = request.get_json(silent=True)

        ok, message, field = check_fields(data)
        if not ok:
            return envelope(
                400, "Invalid Parameter", message, rid,
                **_echo(data, field), **_null_outputs(),
            )

        group_id = data["group_id"]
        decision_idx = data["decision_idx"]
        decision_type = data["decision_type"]
        request_timestamp = data["timestamp"]
        if isinstance(request_timestamp, str):
            request_timestamp = datetime.datetime.fromisoformat(request_timestamp)
        received_timestamp = datetime.datetime.now()

        # Unknown group -> 404 (group_id is the offender, echoed as null).
        group = Group.query.filter_by(group_id=group_id).first()
        if not group:
            return envelope(
                404, "Unknown Group",
                f"group_id {group_id} is not registered.", rid,
                **_echo(data, "group_id"), **_null_outputs(),
            )

        # Idempotency: a repeated (group_id, decision_type, decision_idx)
        # replays the original decision (200), it is not an error.
        existing = Action.query.filter_by(
            group_id=group_id, decision_type=decision_type, decision_idx=decision_idx
        ).first()
        if existing:
            return _replay(existing)

        # Pull the dyad's most recent uploaded snapshot; 409 if none yet.
        latest_upload = (
            DataUpload.query.filter_by(group_id=group_id)
            .order_by(DataUpload.request_timestamp.desc(), DataUpload.id.desc())
            .first()
        )
        if latest_upload is None:
            return envelope(
                409, "No State Available",
                f"No /upload_data has been received for {group_id}; "
                "send a snapshot first.", rid,
                **_echo(data), **_null_outputs(),
            )

        # Project the subset this decision_type needs. Recorded on the action
        # so the decision is reproducible even if later uploads overwrite
        # individual fields; also seeds warm-up rows into the fit.
        raw_context = project_snapshot(decision_type, latest_upload.data, decision_idx)

        # Latest "policy" row (non-snapshot); EB snapshot rows are filtered out.
        model_parameters = (
            ModelParameters.query.filter(ModelParameters.snapshot_type.is_(None))
            .order_by(ModelParameters.timestamp.desc())
            .first()
        )
        if not model_parameters:
            return envelope(
                503, "Model Not Initialized",
                f"The learner for {decision_type} is not loaded; the "
                "deployment is not ready to serve decisions.", rid,
                **_echo(data), **_null_outputs(),
            )

        rl_algorithm = current_app.rl_algorithm

        # Server-side warm-up gate.
        is_warmup, warmup_reason = _evaluate_warmup(group_id, decision_type)

        decision_params = None
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
                # Context is server-projected, not host input, so a make_state
                # failure is an internal error rather than a bad request.
                return envelope(
                    500, "Internal Error",
                    f"Learner could not build the state: {state}", rid,
                    **_echo(data), **_null_outputs(),
                )

            probability = model_parameters.probability_of_action
            action, prob, random_state = rl_algorithm.get_action(
                group_id, state, {"probability": probability}, decision_type, decision_idx
            )
            # θ / Σ / η the learner scored, for the response + replay; kept out
            # of random_state (which is for the sample-buffer cursors).
            if isinstance(random_state, dict):
                decision_params = random_state.pop("decision_params", None)

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
            decision_params=decision_params,
            random_state=random_state,
            model_parameters_id=model_parameters.id,
            request_timestamp=request_timestamp,
            timestamp=received_timestamp,
        )
        db.session.add(new_action)
        try:
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logging.exception(exc)
            return envelope(
                500, "Internal Error",
                "Failed to persist the action.", rid,
                **_echo(data), **_null_outputs(),
            )

        dp = decision_params or {}
        return envelope(
            201, "Success", "Action requested successfully.", rid,
            group_id=group_id,
            decision_type=decision_type,
            decision_idx=decision_idx,
            action=action,
            action_prob=prob,
            warmup=is_warmup,
            state=state,
            model_theta=dp.get("theta"),
            model_cov=dp.get("cov"),
            eta=dp.get("eta"),
        )

    except Exception as e:
        logging.error(f"[Action] Error: {e}")
        logging.exception(e)
        return envelope(
            500, "Internal Error",
            "Learner raised an exception while sampling the action.", rid,
            **_echo(request.get_json(silent=True)), **_null_outputs(),
        )
