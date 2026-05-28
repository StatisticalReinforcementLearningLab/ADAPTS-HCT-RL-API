import datetime

from tests.resource_estimate import estimate_trial_resources


def test_estimate_trial_resources_returns_positive_counts():
    summary = estimate_trial_resources(
        base_date=datetime.date(2025, 1, 5),
        num_weeks=4,
        num_dyads=3,
        seed=42,
    )
    assert summary["event_counts"]["action"] > 0
    assert summary["event_counts"]["upload_data"] > 0
    assert summary["row_counts"]["model_parameters"] > 0
    assert summary["estimated_storage_bytes"] > 0
    assert summary["traffic_profile"]["peak_weekly_actions"] > 0
    assert "logging_budget" in summary
    assert summary["logging_budget"]["total_log_bytes_est"] > 0
    assert "ram_profile_mb" in summary
    assert summary["study_calendar"]["calendar_days_inclusive"] > 0
    assert summary["study_calendar"]["scheduled_weekly_updates"] == 4
    assert summary["rounded_trial_storage_gb_est"] >= 0
