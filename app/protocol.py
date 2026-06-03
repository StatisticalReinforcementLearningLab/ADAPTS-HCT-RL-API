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
        "cp_diary_mood": "float_or_miss",
        "cp_app_engagement": "engagement",
        "cp_app_burden": "nonneg_float",
        "cp_missing_rate_7d": "unit_interval",
        "relationship_quality_cp": "float_or_miss",
        "relationship_quality_aya": "float_or_miss",
        "current_game_on": "binary_or_miss",
    },
    # dyad_game state is the five tailoring variables only (engagement /
    # burden / prior action). week_in_study (γ=0 → no time trend), the two
    # relationship_quality_* fields (they are the outcome, not state), and the
    # two diary_summary fields are not read by the learner — see
    # ProtocolRLFeatureBuilder._specs_for("dyad_game").
    "dyad_game": {
        "agent_decision_index": "positive_int",
        "week_in_study": "positive_int",
        "aya_app_engagement": "engagement",
        "cp_app_engagement": "engagement",
        "aya_app_burden": "nonneg_float",
        "cp_app_burden": "nonneg_float",
        "prior_game_action": "binary_or_miss",
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


# ---------------------------------------------------------------------------
# Flat snapshot contract (API-Spec §5.1). /upload_data carries a full snapshot
# of every variable below; the host does not tag context vs. outcome. The
# learner projects the subset each decision_type needs at /action time (§5.2)
# and reads the outcome fields at /update time (§5.3).
# ---------------------------------------------------------------------------
SNAPSHOT_SCHEMA: dict[str, str] = {
    # bookkeeping
    "day_in_study": "positive_int",
    "week_in_study": "positive_int",
    "slot": "slot",
    # AYA-side
    "aya_diary_mood": "float_or_miss",
    "aya_diary_physical": "float_or_miss",
    "aya_app_engagement": "engagement_or_miss",
    "aya_app_burden": "nonneg_float",
    "aya_missing_rate_7d": "unit_interval",
    "previous_med_adherence": "binary_or_miss",
    "prompted_by_message": "bool",
    # CP-side
    "cp_diary_mood": "float_or_miss",
    "cp_app_engagement": "engagement_or_miss",
    "cp_app_burden": "nonneg_float",
    "cp_missing_rate_7d": "unit_interval",
    "daily_diary_completed": "bool",
    "daily_diary_score": "nonneg_float",
    # dyad-level
    "relationship_quality_aya": "float_or_miss",
    "relationship_quality_cp": "float_or_miss",
    "current_game_on": "binary_or_miss",
    "prior_game_action": "binary_or_miss",
    "aya_diary_summary": "unit_interval_or_miss",
    "cp_diary_summary": "unit_interval_or_miss",
    "weekly_survey_completed": "bool",
    "weekly_relationship_score": "nonneg_float",
}


def validate_snapshot(data: dict[str, Any]) -> tuple[bool, str]:
    """Validate a full flat /upload_data snapshot (every key required)."""
    if not isinstance(data, dict):
        return False, "data must be a dictionary."
    for field_name in data:
        if field_name not in SNAPSHOT_SCHEMA:
            return False, f"unknown field '{field_name}'."
    for field_name, field_type in SNAPSHOT_SCHEMA.items():
        if field_name not in data:
            return False, f"{field_name} is required (full snapshot)."
        valid, error_message = _validate_field(field_type, data[field_name])
        if not valid:
            return False, f"{field_name}: {error_message}"
    return True, ""


def _engagement_or_default(value: Any, default: int = 1) -> int:
    """Engagement bucket, substituting a default when missing (warm-up week 1).

    The learner treats engagement as always-observed; this keeps the stored
    raw_context numeric so warm-up rows can still seed the fit at /update.
    """
    return default if is_missing(value) else int(value)


def _binary_or_default(value: Any, default: int = 0) -> int:
    return default if is_missing(value) else int(value)


def project_snapshot(
    decision_type: str, snapshot: dict[str, Any], decision_idx: int
) -> dict[str, Any]:
    """
    Project the flat snapshot to the per-agent context the feature builder
    consumes (API-Spec §5.2). The API owns the per-(dyad, decision_type)
    counter, so agent_decision_index is derived from decision_idx rather than
    sent by the host. Bookkeeping (day/week/slot) is always carried for the
    /update timeline reward derivation (§5.3); the feature builder ignores
    keys it does not list.
    """
    agent_decision_index = int(decision_idx) + 1
    book = {
        "agent_decision_index": agent_decision_index,
        "day_in_study": int(snapshot["day_in_study"]),
        "week_in_study": int(snapshot["week_in_study"]),
        "slot": snapshot["slot"],
    }

    if decision_type == "aya_message":
        return {
            **book,
            "prior_med_adherence": snapshot["previous_med_adherence"],
            "aya_diary": {
                "mood": snapshot["aya_diary_mood"],
                "physical": snapshot["aya_diary_physical"],
            },
            "relationship_quality_cp": snapshot["relationship_quality_cp"],
            "relationship_quality_aya": snapshot["relationship_quality_aya"],
            "aya_app_engagement": _engagement_or_default(snapshot["aya_app_engagement"]),
            "aya_app_burden": snapshot["aya_app_burden"],
            "aya_missing_rate_7d": snapshot["aya_missing_rate_7d"],
            "current_game_on": _binary_or_default(snapshot["current_game_on"]),
        }

    if decision_type == "cp_message":
        return {
            **book,
            "cp_diary_mood": snapshot["cp_diary_mood"],
            "cp_app_engagement": _engagement_or_default(snapshot["cp_app_engagement"]),
            "cp_app_burden": snapshot["cp_app_burden"],
            "cp_missing_rate_7d": snapshot["cp_missing_rate_7d"],
            "relationship_quality_cp": snapshot["relationship_quality_cp"],
            "relationship_quality_aya": snapshot["relationship_quality_aya"],
            "current_game_on": snapshot["current_game_on"],
        }

    if decision_type == "dyad_game":
        return {
            **book,
            "aya_app_engagement": _engagement_or_default(snapshot["aya_app_engagement"]),
            "cp_app_engagement": _engagement_or_default(snapshot["cp_app_engagement"]),
            "aya_app_burden": snapshot["aya_app_burden"],
            "cp_app_burden": snapshot["cp_app_burden"],
            "prior_game_action": snapshot["prior_game_action"],
        }

    raise ValueError(f"unknown decision_type: {decision_type}")


def outcome_from_snapshot(
    decision_type: str, snapshot: dict[str, Any]
) -> dict[str, Any]:
    """
    Read the outcome fields the reward depends on from a later snapshot on the
    timeline (API-Spec §5.3). Mapped to the OUTCOME_SCHEMAS keys that
    `compute_reward` expects.
    """
    if decision_type == "aya_message":
        return {
            "med_adherence": snapshot["previous_med_adherence"],
            "prompted_by_message": bool(snapshot["prompted_by_message"]),
            "decision_type": "aya_message",
        }
    if decision_type == "cp_message":
        return {
            "daily_diary_completed": bool(snapshot["daily_diary_completed"]),
            "daily_diary_score": float(snapshot["daily_diary_score"]),
            "decision_type": "cp_message",
        }
    if decision_type == "dyad_game":
        return {
            "weekly_survey_completed": bool(snapshot["weekly_survey_completed"]),
            "weekly_relationship_score": float(snapshot["weekly_relationship_score"]),
            "decision_type": "dyad_game",
        }
    raise ValueError(f"unknown decision_type: {decision_type}")


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

    if field_type == "engagement_or_miss":
        if is_missing(value):
            return True, ""
        if value not in {1, 2, 3, 4}:
            return False, "must be one of 1, 2, 3, 4, or 'miss'."
        return True, ""

    if field_type == "unit_interval_or_miss":
        if is_missing(value):
            return True, ""
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            return False, "must be a number between 0 and 1, or 'miss'."
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
