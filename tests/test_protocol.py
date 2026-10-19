"""
The flat snapshot contract of ``app.protocol`` (API-Spec §4.1;
Monitoring-Spec §6.4, §6.4.1; contract D4, D5) and the legacy projection kept
for the legacy learners (contract D1, D7).
"""

from __future__ import annotations

import datetime

import pytest

from app import protocol
from app.protocol import (
    CONTEXT_SCHEMAS,
    KIND_INCONSISTENT,
    KIND_MISSINGFIELD,
    KIND_PARAM,
    KIND_RANGE,
    KIND_TYPE,
    KIND_UNKNOWN,
    SNAPSHOT_SCHEMA,
    check_field,
    check_snapshot,
    check_snapshot_consistency,
    outcome_from_snapshot,
    project_snapshot,
    slot_for_time,
    validate_context,
    validate_snapshot,
)
from tests.conftest import full_snapshot, snapshot_for

EXPECTED_SCHEMA = {
    "day_in_study": "positive_int",
    "week_in_study": "nonneg_int",
    "slot": "slot",
    "aya_diary_mood": "item_1_5_or_miss",
    "aya_diary_physical": "item_1_5_or_miss",
    "aya_app_engagement": "engagement_or_miss",
    "aya_app_burden": "nonneg_float",
    "aya_missing_rate_7d": "unit_interval",
    "previous_med_adherence": "binary_or_miss",
    "prompted_by_message": "bool",
    "aya_messages_delivered_week": "nonneg_int",
    "cp_diary_mood": "item_1_5_or_miss",
    "cp_app_engagement": "engagement_or_miss",
    "cp_app_burden": "nonneg_float",
    "cp_missing_rate_7d": "unit_interval",
    "daily_diary_completed": "bool",
    "cp_prompted_by_message": "bool",
    "cp_messages_delivered_week": "nonneg_int",
    "relationship_quality_aya": "score_1_5_or_miss",
    "relationship_quality_cp": "score_1_5_or_miss",
    "current_game_on": "binary_or_miss",
    "prior_game_action": "binary_or_miss",
    "aya_diary_summary": "unit_interval_or_miss",
    "cp_diary_summary": "unit_interval_or_miss",
    "weekly_survey_completed": "bool",
    "weekly_relationship_score": "nonneg_float",
}


# ------------------------------------------------------------------ schema


def test_snapshot_schema_is_the_26_fields_of_api_spec_4_1():
    assert SNAPSHOT_SCHEMA == EXPECTED_SCHEMA
    assert len(SNAPSHOT_SCHEMA) == 26
    assert "daily_diary_score" not in SNAPSHOT_SCHEMA


def test_full_snapshot_is_valid():
    assert check_snapshot(full_snapshot()) is None
    assert validate_snapshot(full_snapshot()) == (True, "")


def test_api_spec_example_snapshot_is_valid():
    example = {
        "day_in_study": 17, "week_in_study": 3, "slot": "am", "aya_diary_mood": 4, "aya_diary_physical": 3,
        "aya_app_engagement": 3, "aya_app_burden": 1.5, "aya_missing_rate_7d": 0.14,
        "previous_med_adherence": 1, "prompted_by_message": True, "aya_messages_delivered_week": 5,
        "cp_diary_mood": 3, "cp_app_engagement": 2, "cp_app_burden": 1.2, "cp_missing_rate_7d": 0.1,
        "cp_prompted_by_message": True, "cp_messages_delivered_week": 2, "daily_diary_completed": True,
        "relationship_quality_aya": 4, "relationship_quality_cp": 3, "current_game_on": 1,
        "prior_game_action": 0, "aya_diary_summary": 0.55, "cp_diary_summary": 0.5,
        "weekly_survey_completed": True, "weekly_relationship_score": 3.5,
    }
    assert check_snapshot(example) is None
    # API-Spec §2.3 example: Wednesday 2026-06-10 07:25 is day 17 / week 3 for a 2026-05-25 consent.
    assert check_snapshot_consistency(example, "2026-05-25", datetime.datetime(2026, 6, 10, 7, 25)) is None


@pytest.mark.parametrize(
    "data, kind, field, message",
    [
        ([1], KIND_PARAM, "data", "data must be a dictionary."),
        ({**full_snapshot(), "extra": 1}, KIND_UNKNOWN, "extra", "unknown field 'extra'."),
        ({k: v for k, v in full_snapshot().items() if k != "cp_prompted_by_message"}, KIND_MISSINGFIELD,
         "cp_prompted_by_message", "cp_prompted_by_message is required (full snapshot)."),
        (full_snapshot(slot="AM"), KIND_TYPE, "slot", "slot: must be 'am' or 'pm'."),
        (full_snapshot(cp_diary_mood=9), KIND_RANGE, "cp_diary_mood", "cp_diary_mood must be between 1 and 5; got 9."),
    ],
)
def test_check_snapshot_kinds(data, kind, field, message):
    issue = check_snapshot(data)
    assert issue == (kind, field, message)
    assert validate_snapshot(data) == (False, message)


# ------------------------------------------------------------ field types


@pytest.mark.parametrize(
    "field_type, valid, invalid_type, invalid_range",
    [
        ("positive_int", [1, 250], [0, -3, 1.0, True, "1", None, "miss"], []),
        ("nonneg_int", [0, 9], [-1, 0.5, False, None, "miss"], []),
        ("nonneg_float", [0, 0.0, 2.75, 7], [-0.01, None, "miss", True, "1.5"], []),
        ("unit_interval", [0, 1, 0.5], [None, "miss", True, "0.5"], [1.01, -0.2]),
        ("unit_interval_or_miss", [0, 1, 0.3, "miss", None], [True, "x"], [1.5, -1]),
        ("binary_or_miss", [0, 1, "miss", None], [True, False, 0.5, "1"], [2, -1]),
        ("item_1_5_or_miss", [1, 3, 5, "miss", None], [2.5, True, "3"], [0, 6, 7]),
        ("score_1_5_or_miss", [1, 2.75, 5, "miss", None], [True, "4"], [0.99, 5.01]),
        ("engagement_or_miss", [1, 4, "miss", None], [1.5, True], [0, 5]),
        ("slot", ["am", "pm"], ["AM", None, 1, "miss"], []),
        ("bool", [True, False], [0, 1, None, "true", "miss"], []),
    ],
)
def test_field_type_rules(field_type, valid, invalid_type, invalid_range):
    for value in valid:
        assert check_field("f", field_type, value) is None, value
    for value in invalid_type:
        issue = check_field("f", field_type, value)
        assert issue is not None and issue.kind == KIND_TYPE, value
        assert issue.message == f"f: {protocol.FIELD_TYPES[field_type].ending}"
    for value in invalid_range:
        issue = check_field("f", field_type, value)
        assert issue is not None and issue.kind == KIND_RANGE, value
        assert issue.message.startswith("f must be between ")
        assert issue.message.endswith(f"; got {value!r}.")


def test_non_finite_numbers_are_type_errors():
    for value in (float("nan"), float("inf"), float("-inf")):
        assert check_field("x", "float_or_miss", value).kind == KIND_TYPE
        assert check_field("x", "score_1_5_or_miss", value).kind == KIND_TYPE


def test_type_error_endings_match_monitoring_spec_6_4_1():
    expected = {
        "slot": "must be 'am' or 'pm'.",
        "positive_int": "must be a positive integer.",
        "nonneg_int": "must be a non-negative integer.",
        "nonneg_float": "must be a non-negative number.",
        "unit_interval": "must be a number between 0 and 1.",
        "binary": "must be 0 or 1.",
        "binary_or_miss": "must be 0, 1, or 'miss'.",
        "float_or_miss": "must be numeric or 'miss'.",
        "item_1_5_or_miss": "must be an integer from 1 through 5, or 'miss'.",
        "score_1_5_or_miss": "must be a number from 1 through 5, or 'miss'.",
        "engagement": "must be one of 1, 2, 3, or 4.",
        "engagement_or_miss": "must be one of 1, 2, 3, 4, or 'miss'.",
        "unit_interval_or_miss": "must be a number between 0 and 1, or 'miss'.",
        "bool": "must be true or false.",
    }
    assert {k: v.ending for k, v in protocol.FIELD_TYPES.items()} == expected


# ------------------------------------------------------- cross-field checks


def test_consistency_checks_in_order():
    ts = datetime.datetime(2026, 1, 7, 7, 25)
    good = snapshot_for("2026-01-05", ts.isoformat())
    assert check_snapshot_consistency(good, "2026-01-05", ts) is None
    bad = {**good, "day_in_study": 1, "week_in_study": 4, "slot": "pm"}
    assert check_snapshot_consistency(bad, "2026-01-05", ts) == (
        KIND_INCONSISTENT, "day_in_study", "day_in_study does not match timestamp: expected 3, got 1."
    )
    bad["day_in_study"] = 3
    assert check_snapshot_consistency(bad, "2026-01-05", ts) == (
        KIND_INCONSISTENT, "week_in_study", "week_in_study does not match day_in_study: expected 1, got 4."
    )
    bad["week_in_study"] = 1
    assert check_snapshot_consistency(bad, "2026-01-05", ts) == (
        KIND_INCONSISTENT, "slot", "slot does not match timestamp: expected 'am', got 'pm'."
    )


@pytest.mark.parametrize("weekday_offset", range(7))
def test_week_follows_monday_weeks_for_any_consent_weekday(weekday_offset):
    consent = datetime.date(2026, 1, 5) + datetime.timedelta(days=weekday_offset)
    for day in range(0, 30):
        current = consent + datetime.timedelta(days=day)
        ts = datetime.datetime.combine(current, datetime.time(19, 30))
        snap = snapshot_for(consent, ts.isoformat())
        assert check_snapshot_consistency(snap, consent, ts) is None
        week1 = consent + datetime.timedelta(days=-consent.weekday() % 7)
        expected = 0 if current < week1 else (current - week1).days // 7 + 1
        assert snap["week_in_study"] == expected
        assert snap["day_in_study"] == day + 1
        assert snap["slot"] == "pm"


def test_week_closing_upload_carries_the_new_week():
    # Monday 01:00 after week 1 of a Monday consent: day 8, week 2, slot am.
    ts = datetime.datetime(2026, 1, 12, 1, 0)
    snap = snapshot_for("2026-01-05", ts.isoformat())
    assert (snap["day_in_study"], snap["week_in_study"], snap["slot"]) == (8, 2, "am")
    assert check_snapshot_consistency(snap, "2026-01-05", ts) is None


def test_slot_boundary():
    assert slot_for_time(datetime.datetime(2026, 1, 7, 11, 59, 59)) == "am"
    assert slot_for_time(datetime.datetime(2026, 1, 7, 12, 0)) == "pm"
    assert slot_for_time(datetime.datetime(2026, 1, 7, 13, 0), noon_hour=14) == "am"


def test_calendar_reexports_are_the_study_calendar():
    from app import study_calendar

    assert protocol.week1_start is study_calendar.week1_start
    assert protocol.study_week is study_calendar.study_week
    assert protocol.day_in_study is study_calendar.day_in_study
    monday = datetime.date(2026, 1, 5)
    assert protocol.week1_start(datetime.date(2026, 1, 8)) == datetime.date(2026, 1, 12)
    assert protocol.study_week(monday, monday) == 1
    assert protocol.study_week(datetime.date(2026, 1, 8), datetime.date(2026, 1, 11)) == 0


# ------------------------------------------------------------ legacy layer


def test_project_snapshot_keeps_the_legacy_keys():
    snap = full_snapshot(day_in_study=9, week_in_study=2, slot="pm", aya_diary_mood=None, aya_app_engagement="miss",
                         current_game_on="miss", cp_diary_mood=4, prior_game_action=1)
    aya = project_snapshot("aya_message", snap, decision_idx=4)
    assert aya["agent_decision_index"] == 5
    assert (aya["day_in_study"], aya["week_in_study"], aya["slot"]) == (9, 2, "pm")
    assert aya["aya_diary"] == {"mood": "miss", "physical": "miss"}
    assert aya["aya_app_engagement"] == 1  # missing engagement -> level 1
    assert aya["current_game_on"] == 0
    assert validate_context("aya_message", aya) == (True, "")
    cp = project_snapshot("cp_message", snap, decision_idx=0)
    assert cp["cp_diary_mood"] == 4
    assert validate_context("cp_message", cp) == (True, "")
    rel = project_snapshot("dyad_game", snap, decision_idx=1)
    assert rel["prior_game_action"] == 1
    assert set(CONTEXT_SCHEMAS["dyad_game"]) <= set(rel)
    assert validate_context("dyad_game", rel) == (True, "")


def test_legacy_cp_outcome_uses_the_diary_mood():
    assert outcome_from_snapshot("cp_message", full_snapshot(daily_diary_completed=True, cp_diary_mood=4)) == {
        "daily_diary_completed": True, "daily_diary_score": 4.0, "decision_type": "cp_message"}
    assert outcome_from_snapshot("cp_message", full_snapshot(daily_diary_completed=False, cp_diary_mood=4))[
        "daily_diary_score"] == 0.0
    assert outcome_from_snapshot("aya_message", full_snapshot(previous_med_adherence=None))["med_adherence"] == "miss"


def test_legacy_reward_is_unchanged():
    assert protocol.compute_reward("aya_message", 0, {"med_adherence": 1, "prompted_by_message": False}) == 3.0
    assert protocol.compute_reward("aya_message", 1, {"med_adherence": 1, "prompted_by_message": False}) == 2.0
    assert protocol.compute_reward("aya_message", 0, {"med_adherence": 0, "prompted_by_message": False}) == 1.0
    assert protocol.compute_reward("aya_message", 0, {"med_adherence": "miss", "prompted_by_message": False}) == 0.0
    assert protocol.compute_reward("cp_message", 0, {"daily_diary_completed": True, "daily_diary_score": 3.4}) == 3.4
    assert protocol.compute_reward("dyad_game", 0, {"weekly_survey_completed": False,
                                                    "weekly_relationship_score": 4.0}) == 0.0
