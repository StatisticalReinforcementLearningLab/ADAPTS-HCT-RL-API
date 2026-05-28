from copy import deepcopy

from app.routes.action import check_fields


test_group_json = {
    "group_id": "test_group_123",
    "member_list": ["aya_001", "cp_001"],
    "consent_start_date": "2025-01-05",
    "consent_end_date": "2025-04-14",
}

test_data_json = {
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


def test_check_fields_missing_group_id():
    data = deepcopy(test_data_json)
    data.pop("group_id", None)
    result, error_message = check_fields(data)
    assert not result
    assert "group_id and timestamp are required." in error_message


def test_check_fields_missing_timestamp():
    data = deepcopy(test_data_json)
    data.pop("timestamp", None)
    result, error_message = check_fields(data)
    assert not result
    assert "group_id and timestamp are required." in error_message


def test_check_fields_decision_idx_not_int():
    data = deepcopy(test_data_json)
    data["decision_idx"] = "a string"
    result, error_message = check_fields(data)
    assert not result
    assert "decision_idx must be an integer." in error_message


def test_check_fields_missing_context():
    data = deepcopy(test_data_json)
    data.pop("context", None)
    result, error_message = check_fields(data)
    assert not result
    assert "context is required." in error_message


def test_check_fields_invalid_context_shape():
    data = deepcopy(test_data_json)
    data["context"].pop("slot")
    result, error_message = check_fields(data)
    assert not result
    assert "slot is required" in error_message


def test_check_fields_valid():
    result, error_message = check_fields(deepcopy(test_data_json))
    assert result
    assert error_message == ""


def test_request_action_missing_group(client):
    response = client.post("/api/v1/action", json=deepcopy(test_data_json))
    assert response.status_code == 404
    assert response.json["message"] == "Group not found."


def test_request_action_success(client):
    client.post("/api/v1/add_group", json=test_group_json)
    response = client.post("/api/v1/action", json=deepcopy(test_data_json))
    assert response.status_code == 201
    assert response.json["status"] == "success"
    assert response.json["action"] in (0, 1)
    assert isinstance(response.json["state"], list)


def test_request_action_rejects_duplicate_decision_index(client):
    client.post("/api/v1/add_group", json=test_group_json)
    first = client.post("/api/v1/action", json=deepcopy(test_data_json))
    second = client.post("/api/v1/action", json=deepcopy(test_data_json))
    assert first.status_code == 201
    assert second.status_code == 400
    assert "Decision index already exists" in second.json["message"]