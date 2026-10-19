import datetime

from tests.resource_estimate import estimate_trial_resources


def test_estimate_trial_resources_counts_follow_the_calendar():
    summary = estimate_trial_resources(
        base_date=datetime.date(2025, 1, 5),  # a Sunday: dyad_001 consents Monday 2025-01-06
        num_weeks=4,
        num_dyads=3,
        seed=42,
    )
    counts = summary["event_counts"]
    # dyad_001: 28 days x 3 + 4 games; dyad_002 (Tuesday of week 2): 20 days x 3 + 2 games;
    # dyad_003 (Wednesday of week 3): 12 days x 3 + 1 game.
    assert counts["action"] == 88 + 62 + 37
    # One upload per action plus a week-closing upload per dyad per Monday after an active week.
    assert counts["upload_data"] == counts["action"] + 4 + 3 + 2
    assert counts["register_group"] == counts["add_group"] == 3
    # /update every Monday from the first registration through the Monday that closes week 4.
    assert summary["study_calendar"]["scheduled_weekly_updates"] == 5 == counts["update"]
    assert summary["study_calendar"]["calendar_days_inclusive"] == 28
    # Fewer than 5 dyads ever complete week 1: no learning update, only the legacy policy row.
    assert summary["learning_updates"] == 0
    assert summary["row_counts"]["model_parameters"] == 1
    assert summary["traffic_profile"]["peak_weekly_actions"] == 3 * 22  # week 4: every dyad has a full week + game
    assert summary["estimated_storage_bytes"] > 0
    assert summary["logging_budget"]["total_log_bytes_est"] > 0
    assert summary["ram_profile_mb"]["peak_update_mb_est"] > 0
    assert summary["rounded_trial_storage_gb_est"] >= 0


def test_estimate_counts_learning_updates_after_the_cohort_warm_up():
    summary = estimate_trial_resources(num_weeks=12, num_dyads=8, seed=1)
    # Default enrollment (one dyad a week from 2026-11-02): the fifth dyad completes week 1
    # in calendar week 5, so the Mondays of calendar weeks 6..12 are learning updates.
    assert summary["learning_updates"] == 7
    rows = summary["model_parameter_rows"]
    assert rows["hyper"] == 7 * 3
    assert rows["local_fit"] > 0 and rows["posterior"] > 0
    assert summary["row_counts"]["policy_versions"] == 21
