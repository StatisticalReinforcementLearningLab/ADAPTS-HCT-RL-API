"""Protocol schema + reward tests aligned with Study_Design/main.tex §3-§4."""

import pytest

from app.protocol import (
    CONTEXT_SCHEMAS,
    compute_reward,
    validate_context,
    validate_outcome,
)


class TestRewardAYA:
    """4-tier {0, 1, 2, 3} per main.tex §4."""

    def test_unprompted_adherent_is_highest(self):
        r = compute_reward(
            "aya_message", 0, {"med_adherence": 1, "prompted_by_message": False}
        )
        assert r == 3.0

    def test_prompted_adherent_is_middle(self):
        r = compute_reward(
            "aya_message", 1, {"med_adherence": 1, "prompted_by_message": True}
        )
        assert r == 2.0

    def test_non_adherent_is_low(self):
        r = compute_reward(
            "aya_message", 0, {"med_adherence": 0, "prompted_by_message": False}
        )
        assert r == 1.0

    def test_no_report_is_lowest(self):
        r = compute_reward(
            "aya_message", 0, {"med_adherence": "miss", "prompted_by_message": False}
        )
        assert r == 0.0

    def test_action_sent_counts_as_prompted_even_if_outcome_flag_false(self):
        # If RL action was 1, the dose is considered prompted regardless of
        # the outcome-side prompted_by_message flag.
        r = compute_reward(
            "aya_message", 1, {"med_adherence": 1, "prompted_by_message": False}
        )
        assert r == 2.0


class TestRewardCP:
    def test_completed_diary_score(self):
        r = compute_reward(
            "cp_message", 0, {"daily_diary_completed": True, "daily_diary_score": 3.4}
        )
        assert r == pytest.approx(3.4)

    def test_incomplete_diary_zero(self):
        r = compute_reward(
            "cp_message", 1, {"daily_diary_completed": False, "daily_diary_score": 0.0}
        )
        assert r == 0.0


class TestRewardREL:
    def test_completed_survey_score(self):
        r = compute_reward(
            "dyad_game",
            1,
            {"weekly_survey_completed": True, "weekly_relationship_score": 4.2},
        )
        assert r == pytest.approx(4.2)

    def test_incomplete_survey_zero(self):
        r = compute_reward(
            "dyad_game",
            0,
            {"weekly_survey_completed": False, "weekly_relationship_score": 0.0},
        )
        assert r == 0.0


class TestContextSchemas:
    def test_aya_schema_has_no_notifications_48h(self):
        assert "notifications_48h" not in CONTEXT_SCHEMAS["aya_message"]

    def test_cp_schema_has_no_notifications_48h(self):
        assert "notifications_48h" not in CONTEXT_SCHEMAS["cp_message"]

    def test_game_schema_has_diary_summaries(self):
        assert "aya_diary_summary" in CONTEXT_SCHEMAS["dyad_game"]
        assert "cp_diary_summary" in CONTEXT_SCHEMAS["dyad_game"]

    def test_game_schema_has_no_weekly_notifications(self):
        assert "notifications_week_aya" not in CONTEXT_SCHEMAS["dyad_game"]
        assert "notifications_week_cp" not in CONTEXT_SCHEMAS["dyad_game"]

    def test_validate_context_rejects_missing_diary_summary(self):
        ctx = {
            "agent_decision_index": 1,
            "week_in_study": 1,
            "relationship_quality_aya": 4,
            "relationship_quality_cp": 3,
            "aya_app_engagement": 2,
            "cp_app_engagement": 3,
            "aya_app_burden": 0.0,
            "cp_app_burden": 0.0,
            "prior_game_action": "miss",
            # aya_diary_summary intentionally missing
            "cp_diary_summary": 0.5,
        }
        ok, msg = validate_context("dyad_game", ctx)
        assert not ok
        assert "aya_diary_summary" in msg
