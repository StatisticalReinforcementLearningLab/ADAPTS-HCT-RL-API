from copy import deepcopy

from app.routes.action import check_fields
from tests.conftest import register_group, upload


# /action is context-free: only the envelope is sent (API-Spec §2.2).
test_action_json = {
    "group_id": "test_group_123",
    "timestamp": "2026-01-06T09:00:00",
    "decision_idx": 1,
    "decision_type": "aya_message",
}


def test_check_fields_missing_group_id():
    data = deepcopy(test_action_json)
    data.pop("group_id", None)
    result, error_message, field = check_fields(data)
    assert not result
    assert "group_id is required." in error_message
    assert field == "group_id"


def test_check_fields_missing_timestamp():
    data = deepcopy(test_action_json)
    data.pop("timestamp", None)
    result, error_message, field = check_fields(data)
    assert not result
    assert "timestamp is required." in error_message
    assert field == "timestamp"


def test_check_fields_decision_idx_not_int():
    data = deepcopy(test_action_json)
    data["decision_idx"] = "a string"
    result, error_message, field = check_fields(data)
    assert not result
    assert "decision_idx must be an integer." in error_message
    assert field == "decision_idx"


def test_check_fields_missing_decision_type():
    data = deepcopy(test_action_json)
    data.pop("decision_type", None)
    result, error_message, field = check_fields(data)
    assert not result
    assert "decision_type is required." in error_message
    assert field == "decision_type"


def test_check_fields_invalid_decision_type():
    data = deepcopy(test_action_json)
    data["decision_type"] = "not_a_real_agent"
    result, error_message, field = check_fields(data)
    assert not result
    assert "Invalid decision_type" in error_message
    assert field == "decision_type"


def test_check_fields_valid():
    result, error_message, field = check_fields(deepcopy(test_action_json))
    assert result
    assert error_message == ""
    assert field is None


def test_request_action_missing_group(client):
    response = client.post("/api/v1/action", json=deepcopy(test_action_json))
    assert response.status_code == 404
    assert response.json["title"] == "Unknown Group"
    assert "is not registered" in response.json["message"]
    # group_id is the offender, echoed as null (§2 null-discipline).
    assert response.json["group_id"] is None
    assert response.json["rid"] is not None


def test_request_action_without_upload_returns_409(client):
    register_group(client, "test_group_123")
    response = client.post("/api/v1/action", json=deepcopy(test_action_json))
    assert response.status_code == 409
    assert response.json["title"] == "No State Available"
    assert "upload_data" in response.json["message"]


def test_request_action_success(client):
    register_group(client, "test_group_123")
    upload(client, "test_group_123", "2026-01-06T08:00:00")
    response = client.post("/api/v1/action", json=deepcopy(test_action_json))
    assert response.status_code == 201
    assert response.json["status"] == 201
    assert response.json["title"] == "Success"
    assert response.json["action"] in (0, 1)
    assert "warmup" in response.json
    # First dyad (cohort < 5) -> warm-up: state is null, prob is 0.5.
    # The gate reason stays internal (actions row), not in the response.
    assert response.json["warmup"] is True
    assert "warmup_reason" not in response.json
    assert response.json["state"] is None
    assert response.json["model_theta"] is None
    assert response.json["model_cov"] is None
    assert response.json["eta"] is None
    assert response.json["action_prob"] == 0.5
    assert response.json["rid"] is not None


def test_request_action_replays_duplicate_decision_index(client):
    """A repeated idempotency triple replays the original decision (200), not 400."""
    register_group(client, "test_group_123")
    upload(client, "test_group_123", "2026-01-06T08:00:00")
    first = client.post("/api/v1/action", json=deepcopy(test_action_json))
    second = client.post("/api/v1/action", json=deepcopy(test_action_json))
    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json["title"] == "Duplicate Decision"
    # Original outputs echoed verbatim.
    assert second.json["action"] == first.json["action"]
    assert second.json["action_prob"] == first.json["action_prob"]
    assert second.json["rid"] == first.json["rid"]
