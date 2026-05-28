import time
from unittest.mock import patch

from app.models import ModelUpdateRequests

test_group_json = {
    "group_id": "test_group_123",
    "member_list": ["member1", "member2"],
    "consent_start_date": "2025-01-05",
    "consent_end_date": "2025-04-14",
}

test_action_json = {
    "group_id": "test_group_123",
    "timestamp": "2025-01-06T09:00:00",
    "decision_idx": 1,
    "decision_type": "aya_message",
    "context": {
        "slot": "am",
        "agent_decision_index": 1,
        "day_in_study": 2,
        "week_in_study": 1,
        "prior_med_adherence": "miss",
        "aya_diary": {"mood": "miss", "physical": "miss"},
        "relationship_quality_cp": "miss",
        "relationship_quality_aya": "miss",
        "aya_app_engagement": 1,
        "aya_app_burden": 0.0,
        "aya_missing_rate_7d": 1.0,
        "current_game_on": 0,
    },
}

def test_update_model_success(client):
    """
    Tests updating the model successfully.
    """

    client.post("/api/v1/add_group", json=test_group_json)

    action_response = client.post("/api/v1/action", json=test_action_json)
    assert action_response.status_code == 201
    action_json = action_response.get_json()

    upload_response = client.post(
        "/api/v1/upload_data",
        json={
            "group_id": "test_group_123",
            "decision_idx": 1,
            "decision_type": "aya_message",
            "timestamp": "2025-01-06T10:00:00",
            "data": {
                "context": test_action_json["context"],
                "action": action_json["action"],
                "action_prob": action_json["action_prob"],
                "state": action_json["state"],
                "outcome": {
                    "med_adherence": 1,
                    "prompted_by_message": bool(action_json["action"]),
                },
            },
        },
    )
    assert upload_response.status_code == 201

    callback_url = "http://127.0.0.1:5001/callback"
    with patch("app.routes.update.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        response = client.post(
            "/api/v1/update",
            json={
                "callback_url": callback_url,
                "timestamp": "2025-01-12T03:00:00",
            },
        )

        assert response.status_code == 202
        assert response.json["status"] == "processing"
        update_id = response.get_json()["update_id"]

        time.sleep(1.5)

        assert mock_post.called
        callback_payload = mock_post.call_args.kwargs["json"]
        assert callback_payload["update_id"] == update_id
        assert callback_payload["status"] == "completed"

    with client.application.app_context():
        update_row = ModelUpdateRequests.query.filter_by(update_id=update_id).first()
        assert update_row is not None
        assert update_row.status == "completed"
