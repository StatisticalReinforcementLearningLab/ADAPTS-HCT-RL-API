"""
Host data protocol of the ADAPTS-HCT RL API (API-Spec §2.3, §4; Monitoring-Spec
§6.4, §6.4.1; implementation contract D4, D5).

Two layers live here:

1. The **flat snapshot contract** of ``/upload_data`` (API-Spec §4.1):
   :data:`SNAPSHOT_SCHEMA` (the 26 fields and their encodings),
   :func:`check_snapshot` (field-level validation: unknown / missing / type /
   range) and :func:`check_snapshot_consistency` (cross-field consistency
   against the registered dyad and the upload timestamp). Each failure is a
   :class:`SnapshotIssue` whose ``kind`` selects the Monitoring-Spec §6.4 code
   (``UD-400-<kind>``) and whose ``message`` is the exact §6.4 / §6.4.1 wording.
2. The **legacy projection** used by the pre-``inf_lsvi_eb`` learners
   (contract D1: still selectable, not maintained beyond "boots and /action does
   not crash"): :data:`CONTEXT_SCHEMAS`, :func:`project_snapshot`,
   :func:`outcome_from_snapshot` and the legacy :func:`compute_reward`. The
   ``inf_lsvi_eb`` learner builds its own design state and rewards
   (``app.learner``; contract D6, D7).

The study calendar moved to :mod:`app.study_calendar` (contract D5);
``week1_start`` / ``study_week`` / ``day_in_study`` / ``week_monday`` are
re-exported here for compatibility.
"""

from __future__ import annotations

import datetime
import math
from typing import Any, NamedTuple

from app.study_calendar import day_in_study, study_week, week1_start, week_monday  # noqa: F401 (re-exports, D5)

DECISION_TYPES = ("aya_message", "cp_message", "dyad_game")
MISSING_TOKEN = "miss"
DEFAULT_DIARY_ITEMS = ("mood", "physical")


# ---------------------------------------------------------------------------
# Flat snapshot contract (API-Spec §4.1). /upload_data carries a full snapshot
# of every variable below; the host does not tag context vs. outcome. JSON
# null is the same as "miss" where "miss" is allowed and a type error
# elsewhere; JSON booleans are never accepted as numbers.
# ---------------------------------------------------------------------------
SNAPSHOT_SCHEMA: dict[str, str] = {
    # bookkeeping
    "day_in_study": "positive_int",
    "week_in_study": "nonneg_int",
    "slot": "slot",
    # AYA-side
    "aya_diary_mood": "item_1_5_or_miss",
    "aya_diary_physical": "item_1_5_or_miss",
    "aya_app_engagement": "engagement_or_miss",
    "aya_app_burden": "nonneg_float",
    "aya_missing_rate_7d": "unit_interval",
    "previous_med_adherence": "binary_or_miss",
    "prompted_by_message": "bool",
    "aya_messages_delivered_week": "nonneg_int",
    # CP-side
    "cp_diary_mood": "item_1_5_or_miss",
    "cp_app_engagement": "engagement_or_miss",
    "cp_app_burden": "nonneg_float",
    "cp_missing_rate_7d": "unit_interval",
    "daily_diary_completed": "bool",
    "cp_prompted_by_message": "bool",
    "cp_messages_delivered_week": "nonneg_int",
    # dyad-level
    "relationship_quality_aya": "score_1_5_or_miss",
    "relationship_quality_cp": "score_1_5_or_miss",
    "current_game_on": "binary_or_miss",
    "prior_game_action": "binary_or_miss",
    "aya_diary_summary": "unit_interval_or_miss",
    "cp_diary_summary": "unit_interval_or_miss",
    "weekly_survey_completed": "bool",
    "weekly_relationship_score": "nonneg_float",
}

# Validation kinds -> Monitoring-Spec §6.4 code suffix (UD-400-<kind>).
KIND_PARAM = "PARAM"
KIND_UNKNOWN = "UNKNOWN"
KIND_MISSINGFIELD = "MISSINGFIELD"
KIND_TYPE = "TYPE"
KIND_RANGE = "RANGE"
KIND_INCONSISTENT = "INCONSISTENT"
VALIDATION_KINDS = (
    KIND_PARAM,
    KIND_UNKNOWN,
    KIND_MISSINGFIELD,
    KIND_TYPE,
    KIND_RANGE,
    KIND_INCONSISTENT,
)


class SnapshotIssue(NamedTuple):
    """The first validation failure of an upload (Monitoring-Spec §6.4: only the first wrong field is reported)."""

    kind: str
    field: str | None
    message: str


class _FieldType(NamedTuple):
    """
    How one API-Spec §4.1 encoding is checked.

    ``number``: ``"int"`` (JSON integer), ``"real"`` (any finite JSON number),
    or ``None`` (not numeric). ``low``/``high``: the closed range a numeric
    value must fall in; a two-sided range is a RANGE error when violated, a
    one-sided (sign) bound is part of the type and violating it is a TYPE error
    with the §6.4.1 ending (e.g. ``must be a non-negative number.``), because
    the §6.4 RANGE wording needs both ends. ``miss``: ``"miss"`` / JSON null
    allowed. ``ending``: the §6.4.1 type-error message ending.
    """

    number: str | None
    low: float | None
    high: float | None
    miss: bool
    ending: str


FIELD_TYPES: dict[str, _FieldType] = {
    "positive_int": _FieldType("int", 1, None, False, "must be a positive integer."),
    "nonneg_int": _FieldType("int", 0, None, False, "must be a non-negative integer."),
    "nonneg_float": _FieldType("real", 0, None, False, "must be a non-negative number."),
    "unit_interval": _FieldType("real", 0, 1, False, "must be a number between 0 and 1."),
    "unit_interval_or_miss": _FieldType("real", 0, 1, True, "must be a number between 0 and 1, or 'miss'."),
    "binary": _FieldType("int", 0, 1, False, "must be 0 or 1."),
    "binary_or_miss": _FieldType("int", 0, 1, True, "must be 0, 1, or 'miss'."),
    "float_or_miss": _FieldType("real", None, None, True, "must be numeric or 'miss'."),
    "item_1_5_or_miss": _FieldType("int", 1, 5, True, "must be an integer from 1 through 5, or 'miss'."),
    "score_1_5_or_miss": _FieldType("real", 1, 5, True, "must be a number from 1 through 5, or 'miss'."),
    "engagement": _FieldType("int", 1, 4, False, "must be one of 1, 2, 3, or 4."),
    "engagement_or_miss": _FieldType("int", 1, 4, True, "must be one of 1, 2, 3, 4, or 'miss'."),
    "slot": _FieldType(None, None, None, False, "must be 'am' or 'pm'."),
    "bool": _FieldType(None, None, None, False, "must be true or false."),
}


def is_missing(value: Any) -> bool:
    """``True`` for the literal ``"miss"`` and JSON null (API-Spec §2.3)."""
    return value is None or value == MISSING_TOKEN


def format_value(value: Any) -> str:
    """Render a request value inside a validation message: strings quoted, numbers bare, JSON literals as JSON."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(value)


def _format_bound(bound: float) -> str:
    return str(int(bound)) if float(bound).is_integer() else repr(float(bound))


def check_field(field_name: str, field_type: str, value: Any) -> SnapshotIssue | None:
    """
    Check one snapshot value against its API-Spec §4.1 encoding.

    Returns ``None`` when valid; otherwise a TYPE issue
    ``"<field>: <ending>"`` (wrong JSON type, non-integer where an integer is
    required, non-finite number, bad sign, or null where "miss" is not
    allowed) or a RANGE issue ``"<field> must be between <low> and <high>;
    got <value>."`` (a number of the right kind outside a two-sided range)
    (Monitoring-Spec §6.4, §6.4.1).
    """
    spec = FIELD_TYPES[field_type]
    type_issue = SnapshotIssue(KIND_TYPE, field_name, f"{field_name}: {spec.ending}")

    if field_type == "slot":
        return None if value in ("am", "pm") else type_issue
    if field_type == "bool":
        return None if isinstance(value, bool) else type_issue

    if spec.miss and is_missing(value):
        return None
    # JSON booleans are not numbers; strings other than "miss" and null are type errors.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return type_issue
    if isinstance(value, float) and not math.isfinite(value):
        return type_issue
    if spec.number == "int" and not isinstance(value, int):
        return type_issue

    two_sided = spec.low is not None and spec.high is not None
    if two_sided:
        if not spec.low <= value <= spec.high:
            return SnapshotIssue(
                KIND_RANGE,
                field_name,
                f"{field_name} must be between {_format_bound(spec.low)} and "
                f"{_format_bound(spec.high)}; got {format_value(value)}.",
            )
        return None
    if spec.low is not None and value < spec.low:
        return type_issue
    return None


def check_snapshot(data: Any) -> SnapshotIssue | None:
    """
    Field-level validation of an /upload_data ``data`` object (API-Spec §4.1;
    Monitoring-Spec §6.4). Checks, in order, and returns the first failure:

    1. ``data`` is a dictionary (PARAM ``data must be a dictionary.``);
    2. no key outside :data:`SNAPSHOT_SCHEMA` (UNKNOWN ``unknown field '<name>'.``,
       in the order the keys were sent);
    3. every schema field present (MISSINGFIELD ``<field> is required (full snapshot).``,
       in schema order);
    4. every value of the right type and range (TYPE / RANGE, in schema order).

    Cross-field consistency needs the registered dyad and is checked separately
    by :func:`check_snapshot_consistency`.
    """
    if not isinstance(data, dict):
        return SnapshotIssue(KIND_PARAM, "data", "data must be a dictionary.")
    for field_name in data:
        if field_name not in SNAPSHOT_SCHEMA:
            return SnapshotIssue(KIND_UNKNOWN, str(field_name), f"unknown field '{field_name}'.")
    for field_name in SNAPSHOT_SCHEMA:
        if field_name not in data:
            return SnapshotIssue(
                KIND_MISSINGFIELD, field_name, f"{field_name} is required (full snapshot)."
            )
    for field_name, field_type in SNAPSHOT_SCHEMA.items():
        issue = check_field(field_name, field_type, data[field_name])
        if issue is not None:
            return issue
    return None


def slot_for_time(local_time: datetime.datetime, noon_hour: int = 12) -> str:
    """``"am"`` iff the Detroit-local hour is before ``noon_hour`` (``UPLOAD_SLOT_NOON_HOUR``), else ``"pm"``."""
    return "am" if local_time.hour < int(noon_hour) else "pm"


def check_snapshot_consistency(
    data: dict,
    consent_start: Any,
    upload_time: datetime.datetime,
    noon_hour: int = 12,
) -> SnapshotIssue | None:
    """
    Cross-field consistency of a field-valid snapshot (API-Spec §4.1;
    Monitoring-Spec §6.4 ``UD-400-INCONSISTENT``). ``upload_time`` is the
    request timestamp in Detroit local time (contract D4). Checks, in order:

    - ``day_in_study == (date - consent_start_date).days + 1``
      (``day_in_study does not match timestamp: ...``);
    - ``week_in_study == study_calendar.study_week(consent_start_date, date)``
      (``week_in_study does not match day_in_study: ...``). This is the
      API-Spec §4.1 Monday-week formula; the Monitoring-Spec §5.B wording
      ``week = floor((day - 1)/7) + 1`` predates Monday-aligned weeks and is
      not used;
    - ``slot == "am"`` iff the local hour is before ``noon_hour``
      (``slot does not match timestamp: ...``).

    Message: ``<field> does not match <other field>: expected <value>, got <value>.``
    """
    date = upload_time.date()

    expected_day = day_in_study(consent_start, date)
    if data["day_in_study"] != expected_day:
        return SnapshotIssue(
            KIND_INCONSISTENT,
            "day_in_study",
            f"day_in_study does not match timestamp: expected {format_value(expected_day)}, "
            f"got {format_value(data['day_in_study'])}.",
        )

    expected_week = study_week(consent_start, date)
    if data["week_in_study"] != expected_week:
        return SnapshotIssue(
            KIND_INCONSISTENT,
            "week_in_study",
            f"week_in_study does not match day_in_study: expected {format_value(expected_week)}, "
            f"got {format_value(data['week_in_study'])}.",
        )

    expected_slot = slot_for_time(upload_time, noon_hour)
    if data["slot"] != expected_slot:
        return SnapshotIssue(
            KIND_INCONSISTENT,
            "slot",
            f"slot does not match timestamp: expected {format_value(expected_slot)}, "
            f"got {format_value(data['slot'])}.",
        )
    return None


def validate_snapshot(data: dict[str, Any]) -> tuple[bool, str]:
    """Compatibility wrapper of :func:`check_snapshot`: ``(ok, message)`` (field-level checks only)."""
    issue = check_snapshot(data)
    return (True, "") if issue is None else (False, issue.message)


def validate_decision_type(decision_type: str) -> tuple[bool, str]:
    """``/action`` agent check (Monitoring-Spec §6.3)."""
    if decision_type not in DECISION_TYPES:
        return (
            False,
            "Invalid decision_type. Must be 'aya_message', 'cp_message', or 'dyad_game'.",
        )
    return True, ""


# ---------------------------------------------------------------------------
# Legacy learners (contract D1). Everything below serves the pre-inf_lsvi_eb
# algorithms only: their per-agent context schema (Table 2 of the retired
# Study_Design/main.tex), the projection of the flat snapshot onto it, and
# their reward. The inf_lsvi_eb learner does not read any of it.
# ---------------------------------------------------------------------------
CONTEXT_SCHEMAS: dict[str, dict[str, Any]] = {
    "aya_message": {
        "slot": "slot",
        "agent_decision_index": "positive_int",
        "day_in_study": "positive_int",
        "week_in_study": "nonneg_int",
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
        "week_in_study": "nonneg_int",
        "cp_diary_mood": "float_or_miss",
        "cp_app_engagement": "engagement",
        "cp_app_burden": "nonneg_float",
        "cp_missing_rate_7d": "unit_interval",
        "relationship_quality_cp": "float_or_miss",
        "relationship_quality_aya": "float_or_miss",
        "current_game_on": "binary_or_miss",
    },
    # dyad_game state is the five tailoring variables only (engagement /
    # burden / prior action); see ProtocolRLFeatureBuilder._specs_for("dyad_game").
    "dyad_game": {
        "agent_decision_index": "positive_int",
        "week_in_study": "nonneg_int",
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


def validate_context(decision_type: str, context: dict[str, Any]) -> tuple[bool, str]:
    """Legacy per-agent context check (used by the legacy learners' ``make_state``)."""
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
    """Legacy per-agent outcome check (used by the legacy learners' ``make_reward``)."""
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
    """Legacy pre-action feature vector u(s): intercept + per-variable [I, v*I]."""
    from app.feature_builder import ProtocolRLFeatureBuilder

    return (
        ProtocolRLFeatureBuilder(decision_type)
        .base_vector(context, baselines=baselines)
        .tolist()
    )


def compute_reward(decision_type: str, action: int, outcome: dict[str, Any]) -> float:
    """
    Legacy scalar reward (retired main.tex §4; legacy learners only). The
    ``inf_lsvi_eb`` reward is ``app.learner.compute_reward`` (contract D6).

    - aya_message: 4-tier {0,1,2,3} ordinal — {no usable report, non-adherent,
      reported after prompt, unprompted}.
    - cp_message: daily-diary score if the diary was completed, else 0.
    - dyad_game: weekly relationship score if the survey was completed, else 0.
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


def _engagement_or_default(value: Any, default: int = 1) -> int:
    """Engagement bucket, substituting level 1 when missing (API-Spec §4.1: "the learner substitutes level 1")."""
    return default if is_missing(value) else int(value)


def _binary_or_default(value: Any, default: int = 0) -> int:
    return default if is_missing(value) else int(value)


def _miss_if_null(value: Any) -> Any:
    """JSON null and "miss" are the same (API-Spec §2.3); the legacy context uses the literal."""
    return MISSING_TOKEN if value is None else value


def project_snapshot(
    decision_type: str, snapshot: dict[str, Any], decision_idx: int
) -> dict[str, Any]:
    """
    Legacy projection of the flat snapshot onto the per-agent context the
    legacy feature builder consumes. ``agent_decision_index`` is derived from
    ``decision_idx``. Bookkeeping (day/week/slot) is always carried; the
    feature builder ignores keys it does not list. Stored in
    ``actions.raw_context`` next to the ``inf_lsvi_eb`` design state so the
    legacy learners keep working (contract D7).
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
            "prior_med_adherence": _miss_if_null(snapshot["previous_med_adherence"]),
            "aya_diary": {
                "mood": _miss_if_null(snapshot["aya_diary_mood"]),
                "physical": _miss_if_null(snapshot["aya_diary_physical"]),
            },
            "relationship_quality_cp": _miss_if_null(snapshot["relationship_quality_cp"]),
            "relationship_quality_aya": _miss_if_null(snapshot["relationship_quality_aya"]),
            "aya_app_engagement": _engagement_or_default(snapshot["aya_app_engagement"]),
            "aya_app_burden": snapshot["aya_app_burden"],
            "aya_missing_rate_7d": snapshot["aya_missing_rate_7d"],
            "current_game_on": _binary_or_default(snapshot["current_game_on"]),
        }

    if decision_type == "cp_message":
        return {
            **book,
            "cp_diary_mood": _miss_if_null(snapshot["cp_diary_mood"]),
            "cp_app_engagement": _engagement_or_default(snapshot["cp_app_engagement"]),
            "cp_app_burden": snapshot["cp_app_burden"],
            "cp_missing_rate_7d": snapshot["cp_missing_rate_7d"],
            "relationship_quality_cp": _miss_if_null(snapshot["relationship_quality_cp"]),
            "relationship_quality_aya": _miss_if_null(snapshot["relationship_quality_aya"]),
            "current_game_on": _miss_if_null(snapshot["current_game_on"]),
        }

    if decision_type == "dyad_game":
        return {
            **book,
            "aya_app_engagement": _engagement_or_default(snapshot["aya_app_engagement"]),
            "cp_app_engagement": _engagement_or_default(snapshot["cp_app_engagement"]),
            "aya_app_burden": snapshot["aya_app_burden"],
            "cp_app_burden": snapshot["cp_app_burden"],
            "prior_game_action": _miss_if_null(snapshot["prior_game_action"]),
        }

    raise ValueError(f"unknown decision_type: {decision_type}")


def outcome_from_snapshot(
    decision_type: str, snapshot: dict[str, Any]
) -> dict[str, Any]:
    """
    Legacy outcome fields (the :data:`OUTCOME_SCHEMAS` keys the legacy
    :func:`compute_reward` expects) read from a later snapshot. The CP diary
    score is ``cp_diary_mood`` when the diary was completed (the snapshot no
    longer carries a separate ``daily_diary_score``).
    """
    if decision_type == "aya_message":
        return {
            "med_adherence": _miss_if_null(snapshot["previous_med_adherence"]),
            "prompted_by_message": bool(snapshot["prompted_by_message"]),
            "decision_type": "aya_message",
        }
    if decision_type == "cp_message":
        completed = bool(snapshot["daily_diary_completed"])
        mood = snapshot["cp_diary_mood"]
        score = float(mood) if completed and not is_missing(mood) else 0.0
        return {
            "daily_diary_completed": completed,
            "daily_diary_score": score,
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
    """Legacy context/outcome value check (legacy learners only)."""
    if field_type == "slot":
        if value not in {"am", "pm"}:
            return False, "must be 'am' or 'pm'."
        return True, ""

    if field_type == "positive_int":
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return False, "must be a positive integer."
        return True, ""

    if field_type == "nonneg_int":
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return False, "must be a non-negative integer."
        return True, ""

    if field_type == "nonneg_float":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0:
            return False, "must be a non-negative number."
        return True, ""

    if field_type == "unit_interval":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            return False, "must be a number between 0 and 1."
        return True, ""

    if field_type == "binary":
        if isinstance(value, bool) or value not in {0, 1}:
            return False, "must be 0 or 1."
        return True, ""

    if field_type == "binary_or_miss":
        if is_missing(value):
            return True, ""
        if isinstance(value, bool) or value not in {0, 1}:
            return False, "must be 0, 1, or 'miss'."
        return True, ""

    if field_type == "float_or_miss":
        if is_missing(value):
            return True, ""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False, "must be numeric or 'miss'."
        return True, ""

    if field_type == "engagement":
        if isinstance(value, bool) or value not in {1, 2, 3, 4}:
            return False, "must be one of 1, 2, 3, or 4."
        return True, ""

    if field_type == "engagement_or_miss":
        if is_missing(value):
            return True, ""
        if isinstance(value, bool) or value not in {1, 2, 3, 4}:
            return False, "must be one of 1, 2, 3, 4, or 'miss'."
        return True, ""

    if field_type == "unit_interval_or_miss":
        if is_missing(value):
            return True, ""
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
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
            if isinstance(item_value, bool) or not isinstance(item_value, (int, float)):
                return False, f"'{item}' must be numeric or 'miss'."
        return True, ""

    if field_type == "bool":
        if not isinstance(value, bool):
            return False, "must be true or false."
        return True, ""

    return False, f"unsupported schema field type '{field_type}'."
