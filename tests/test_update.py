import time

from app.models import ModelUpdateRequests, StudyData
from tests.conftest import register_group, upload


def test_update_model_success(client):
    """
    /update is async with no callback (API-Spec §3.4): completion is observed
    by reading model_update_requests. Rewards are derived server-side from the
    data_uploads timeline.
    """
    register_group(client, "test_group_123")

    # One CP decision: pre-action upload, action, then a next-morning upload
    # that closes the outcome window.
    upload(
        client,
        "test_group_123",
        "2026-01-05T08:00:00",
        slot="am",
        day_in_study=1,
        cp_diary_mood=4,
        daily_diary_completed=True,
        daily_diary_score=4.0,
    )
    action_response = client.post(
        "/api/v1/action",
        json={
            "group_id": "test_group_123",
            "timestamp": "2026-01-05T09:00:00",
            "decision_idx": 0,
            "decision_type": "cp_message",
        },
    )
    assert action_response.status_code == 201

    # Next morning's upload supplies the CP outcome (yesterday's diary).
    upload(
        client,
        "test_group_123",
        "2026-01-06T08:00:00",
        slot="am",
        day_in_study=2,
        cp_diary_mood=4,
        daily_diary_completed=True,
        daily_diary_score=4.0,
    )

    response = client.post(
        "/api/v1/update",
        json={"timestamp": "2026-01-12T03:00:00"},
    )
    assert response.status_code == 202
    assert response.json["status"] == "processing"
    update_id = response.get_json()["update_id"]

    # Poll for completion (no callback).
    for _ in range(30):
        with client.application.app_context():
            row = ModelUpdateRequests.query.filter_by(update_id=update_id).first()
            if row is not None and row.status in ("completed", "failed"):
                break
        time.sleep(0.1)

    with client.application.app_context():
        update_row = ModelUpdateRequests.query.filter_by(update_id=update_id).first()
        assert update_row is not None
        assert update_row.status == "completed"
        assert update_row.completed_at is not None

        # The CP action got a derived reward = daily_diary_score (4.0).
        sd = StudyData.query.filter_by(
            group_id="test_group_123", decision_type="cp_message", decision_idx=0
        ).first()
        assert sd is not None
        assert sd.reward == 4.0
        assert sd.action_prob == 0.5  # warm-up decision


def test_update_requires_timestamp(client):
    response = client.post("/api/v1/update", json={})
    assert response.status_code == 400
    assert "timestamp is required" in response.json["message"]
