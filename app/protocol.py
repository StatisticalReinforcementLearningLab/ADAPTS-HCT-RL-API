from __future__ import annotations

from typing import Any


DECISION_TYPES = ("aya_message", "cp_message", "dyad_game")
MISSING_TOKEN = "miss"
DEFAULT_DIARY_ITEMS = ("mood", "physical")


# Context schemas follow Table 2 in Study_Design/main.tex. The previous
# `notifications_48h` (AYA/CP) and `notifications_week_*` (REL) fields are
# dropped — app-burden terms subsume those raw counts — and the REL game
# schema gains per-role diary summaries (`aya_diary_summary`, `cp_diary_summary`).
CONTEXT_SCHEMAS: dict[str, dict[str, Any]] = {
    "aya_message": {
        "slot": "slot",
        "agent_decision_index": "positive_int",
        "day_in_study": "positive_int",
        "week_in_study": "positive_int",
        "prior_med_adherence": "binary_or_miss",
        "aya_diary": "diary_block",
        "relationship_quality_cp": "float_or_miss",
        "relationship_quality_aya": "float_or_miss",
        "aya_app_engagement": "engagement",
        "aya_app_burden": "nonneg_float",
        "aya_missing_rate_7d": "unit_interval",
        "current_game_on": "binary",
    },
    "cp_message": {
        "agent_decision_index": "positive_int",
        "day_in_study": "positive_int",
        "week_in_study": "positive_int",
        "cp_diary": "diary_block",
        "cp_app_engagement": "engagement",
        "cp_app_burden": "nonneg_float",
        "cp_missing_rate_7d": "unit_interval",
        "relationship_quality_cp": "float_or_miss",
        "relationship_quality_aya": "float_or_miss",
        "current_game_on": "binary_or_miss",
    },
    "dyad_game": {
        "agent_decision_index": "positive_int",
        "week_in_study": "positive_int",
        "relationship_quality_aya": "float_or_miss",
        "relationship_quality_cp": "float_or_miss",
        "aya_app_engagement": "engagement",
        "cp_app_engagement": "engagement",
        "aya_app_burden": "nonneg_float",
        "cp_app_burden": "nonneg_float",
        "prior_game_action": "binary_or_miss",
        "aya_diary_summary": "unit_interval",
        "cp_diary_summary": "unit_interval",
    },
}


OUTCOME_SCHEMAS: dict[str, dict[str, Any]] = {
    "aya_message": {
        "med_adherence": "binary_or_miss",
        "prompted_by_message": "bool",
    },
    "cp_message": {
        "daily_diary_completed": "bool",
        "daily_diary_score": "nonneg_float",
    },
    "dyad_game": {
        "weekly_survey_completed": "bool",
        "weekly_relationship_score": "nonneg_float",
    },
}


def is_missing(value: Any) -> bool:
    return value is None or value == MISSING_TOKEN


def validate_decision_type(decision_type: str) -> tuple[bool, str]:
    if decision_type not in DECISION_TYPES:
        return (
            False,
            "Invalid decision_type. Must be 'aya_message', 'cp_message', or 'dyad_game'.",
        )
    return True, ""


def validate_context(decision_type: str, context: dict[str, Any]) -> tuple[bool, str]:
    valid_type, message = validate_decision_type(decision_type)
    if not valid_type:
        return False, message

    if not isinstance(context, dict):
        return False, "context must be a dictionary."

    schema = CONTEXT_SCHEMAS[decision_type]
    for field_name, field_type in schema.items():
        if field_name not in context:
            return False, f"Invalid context for {decision_type}. {field_name} is required."
        value = context[field_name]
        valid, error_message = _validate_field(field_type, value)
        if not valid:
            return False, f"Invalid context for {decision_type}. {field_name}: {error_message}"

    return True, ""


def validate_outcome(decision_type: str, outcome: dict[str, Any]) -> tuple[bool, str]:
    valid_type, message = validate_decision_type(decision_type)
    if not valid_type:
        return False, message

    if not isinstance(outcome, dict):
        return False, "outcome must be a dictionary."

    schema = OUTCOME_SCHEMAS[decision_type]
    for field_name, field_type in schema.items():
        if field_name not in outcome:
            return False, f"Invalid outcome for {decision_type}. {field_name} is required."
        value = outcome[field_name]
        valid, error_message = _validate_field(field_type, value)
        if not valid:
            return False, f"Invalid outcome for {decision_type}. {field_name}: {error_message}"

    return True, ""


def encode_state(
    decision_type: str,
    context: dict[str, Any],
    baselines: dict[str, dict[str, float]] | None = None,
) -> list[float]:
    """Pre-action feature vector u(s): intercept + per-variable [I, v*I]."""
    from app.feature_builder import ProtocolRLFeatureBuilder

    return (
        ProtocolRLFeatureBuilder(decision_type)
        .base_vector(context, baselines=baselines)
        .tolist()
    )


def compute_reward(decision_type: str, action: int, outcome: dict[str, Any]) -> float:
    """
    Scalar reward per main.tex §4.

    - aya_message: 4-tier {0,1,2,3} ordinal — {no usable report, non-adherent,
      reported after prompt, unprompted}.
    - cp_message: average daily-diary wellbeing score if diary completed, else 0.
    - dyad_game: average weekly relationship-quality score if survey completed,
      else 0.
    """
    if decision_type == "aya_message":
        adherence = outcome["med_adherence"]
        if is_missing(adherence):
            return 0.0  # no usable self-report for the evaluation cycle
        if int(adherence) == 0:
            return 1.0  # AYA reports not taking the medication
        # Reported adherent
        if outcome["prompted_by_message"] or int(action) == 1:
            return 2.0  # adherent after a supportive message
        return 3.0  # unprompted adherent dose

    if decision_type == "cp_message":
        if not outcome["daily_diary_completed"]:
            return 0.0
        return float(outcome["daily_diary_score"])

    if not outcome["weekly_survey_completed"]:
        return 0.0
    return float(outcome["weekly_relationship_score"])


def agent_index_for_context(decision_type: str, context: dict[str, Any]) -> int:
    schema = CONTEXT_SCHEMAS[decision_type]
    if "agent_decision_index" in schema:
        return int(context["agent_decision_index"])
    raise KeyError(f"agent_decision_index missing from {decision_type} context")


def _validate_field(field_type: str, value: Any) -> tuple[bool, str]:
    if field_type == "slot":
        if value not in {"am", "pm"}:
            return False, "must be 'am' or 'pm'."
        return True, ""

    if field_type == "positive_int":
        if not isinstance(value, int) or value <= 0:
            return False, "must be a positive integer."
        return True, ""

    if field_type == "nonneg_int":
        if not isinstance(value, int) or value < 0:
            return False, "must be a non-negative integer."
        return True, ""

    if field_type == "nonneg_float":
        if not isinstance(value, (int, float)) or float(value) < 0:
            return False, "must be a non-negative number."
        return True, ""

    if field_type == "unit_interval":
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            return False, "must be a number between 0 and 1."
        return True, ""

    if field_type == "binary":
        if value not in {0, 1}:
            return False, "must be 0 or 1."
        return True, ""

    if field_type == "binary_or_miss":
        if is_missing(value):
            return True, ""
        if value not in {0, 1}:
            return False, "must be 0, 1, or 'miss'."
        return True, ""

    if field_type == "float_or_miss":
        if is_missing(value):
            return True, ""
        if not isinstance(value, (int, float)):
            return False, "must be numeric or 'miss'."
        return True, ""

    if field_type == "engagement":
        if value not in {1, 2, 3, 4}:
            return False, "must be one of 1, 2, 3, or 4."
        return True, ""

    if field_type == "diary_block":
        if not isinstance(value, dict):
            return False, "must be a dictionary with diary items."
        for item in DEFAULT_DIARY_ITEMS:
            if item not in value:
                return False, f"missing diary item '{item}'."
            item_value = value[item]
            if is_missing(item_value):
                continue
            if not isinstance(item_value, (int, float)):
                return False, f"'{item}' must be numeric or 'miss'."
        return True, ""

    if field_type == "bool":
        if not isinstance(value, bool):
            return False, "must be true or false."
        return True, ""

    return False, f"unsupported schema field type '{field_type}'."
