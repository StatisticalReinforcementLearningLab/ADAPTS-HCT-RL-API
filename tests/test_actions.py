from copy import deepcopy

from app.routes.action import check_fields
from tests.conftest import register_group, upload


# /action is context-free: only the envelope is sent (API-Spec §3.2).
test_action_json = {
    "group_id": "test_group_123",
    "timestamp": "2026-01-06T09:00:00",
    "decision_idx": 1,
    "decision_type": "aya_message",
}


def test_check_fields_missing_group_id():
    data = deepcopy(test_action_json)
    data.pop("group_id", None)
    result, error_message = check_fields(data)
    assert not result
    assert "group_id and timestamp are required." in error_message


def test_check_fields_missing_timestamp():
    data = deepcopy(test_action_json)
    data.pop("timestamp", None)
    result, error_message = check_fields(data)
    assert not result
    assert "group_id and timestamp are required." in error_message


def test_check_fields_decision_idx_not_int():
    data = deepcopy(test_action_json)
    data["decision_idx"] = "a string"
    result, error_message = check_fields(data)
    assert not result
    assert "decision_idx must be an integer." in error_message


def test_check_fields_missing_decision_type():
    data = deepcopy(test_action_json)
    data.pop("decision_type", None)
    result, error_message = check_fields(data)
    assert not result
    assert "decision_type is required." in error_message


def test_check_fields_invalid_decision_type():
    data = deepcopy(test_action_json)
    data["decision_type"] = "not_a_real_agent"
    result, error_message = check_fields(data)
    assert not result
    assert "Invalid decision_type" in error_message


def test_check_fields_valid():
    result, error_message = check_fields(deepcopy(test_action_json))
    assert result
    assert error_message == ""


def test_request_action_missing_group(client):
    response = client.post("/api/v1/action", json=deepcopy(test_action_json))
    assert response.status_code == 404
    assert response.json["message"] == "Group not found."


def test_request_action_without_upload_returns_409(client):
    register_group(client, "test_group_123")
    response = client.post("/api/v1/action", json=deepcopy(test_action_json))
    assert response.status_code == 409
    assert "upload_data" in response.json["message"]


def test_request_action_success(client):
    register_group(client, "test_group_123")
    upload(client, "test_group_123", "2026-01-06T08:00:00")
    response = client.post("/api/v1/action", json=deepcopy(test_action_json))
    assert response.status_code == 201
    assert response.json["status"] == "success"
    assert response.json["action"] in (0, 1)
    assert "warmup" in response.json
    # First dyad (cohort < 5) -> warm-up: state is null, prob is 0.5.
    assert response.json["warmup"] is True
    assert response.json["warmup_reason"] == "cohort"
    assert response.json["state"] is None
    assert response.json["action_prob"] == 0.5


def test_request_action_rejects_duplicate_decision_index(client):
    register_group(client, "test_group_123")
    upload(client, "test_group_123", "2026-01-06T08:00:00")
    first = client.post("/api/v1/action", json=deepcopy(test_action_json))
    second = client.post("/api/v1/action", json=deepcopy(test_action_json))
    assert first.status_code == 201
    assert second.status_code == 400
    assert "Decision index already exists" in second.json["message"]
