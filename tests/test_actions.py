import pytest
from app.routes.action import check_fields, request_action
from app.models import Dyad, StudyData
from unittest.mock import patch, MagicMock
from copy import deepcopy

test_data_json = {
    "dyad_id": "test_dyad_123",
    "timestamp": "2025-01-01T12:00:00",
    "decision_idx": 0,
    "context": {
        "decision_type": "aya_message",
        "cur_var": 1,
        "past3_vars": [1, 2, 3],
    }
}

# Test check_fields for all scenarios
def test_check_fields_missing_dyad_id():
    data = deepcopy(test_data_json)
    data.pop("dyad_id", None)
    result, error_message = check_fields(data)
    assert not result
    assert "dyad_id and timestamp are required." in error_message

def test_check_fields_missing_timestamp():
    data = deepcopy(test_data_json)
    data.pop("timestamp", None)
    result, error_message = check_fields(data)
    assert not result
    assert "dyad_id and timestamp are required." in error_message

def test_check_fields_decision_idx_not_int():
    data = deepcopy(test_data_json)
    data["decision_idx"] = "a string"
    result, error_message = check_fields(data)
    assert not result
    assert "decision_idx must be an integer." in error_message

def test_check_fields_missing_decision_idx():
    data = deepcopy(test_data_json)
    data.pop("decision_idx", None)
    result, error_message = check_fields(data)
    assert not result
    assert "decision_idx is required." in error_message

def test_check_fields_missing_context():
    data = deepcopy(test_data_json)
    data.pop("context", None)
    result, error_message = check_fields(data)
    assert not result
    assert "context is required." in error_message

# def test_check_fields_missing_temperature():
#     data = {"dyad_id": "test_dyad_123", "timestamp": "2025-01-01T12:00:00", "decision_idx": 0, "context": {}}
#     result, error_message = check_fields(data)
#     assert not result
#     assert "Invalid context. Temperature is required." in error_message

# Now all the fields with wrong data types
def test_check_fields_dyad_id_not_string():
    data = deepcopy(test_data_json)
    data["dyad_id"] = 123
    result, error_message = check_fields(data)
    assert not result
    assert "dyad_id must be a string." in error_message

def test_check_fields_timestamp_not_string():
    data = deepcopy(test_data_json)
    data["timestamp"] = 123
    result, error_message = check_fields(data)
    assert not result
    assert "timestamp must be a string or datetime object." in error_message

def test_check_fields_context_not_dict():
    data = deepcopy(test_data_json)
    data["context"] = "a string"
    result, error_message = check_fields(data)
    assert not result
    assert "context must be a dictionary." in error_message

def test_check_fields_decision_type_not_in_list():
    data = deepcopy(test_data_json)
    data["context"]["decision_type"] = "a string"
    result, error_message = check_fields(data)
    assert not result
    assert "Invalid decision_type in context. Must be 'aya_message', 'cp_message', or 'dyad_game'." in error_message

def test_check_fields_valid():
    data = deepcopy(test_data_json)
    result, error_message = check_fields(data)
    print(error_message)
    assert result
    assert error_message == ""

# Test request_action for all scenarios
# Test request_action_missing_dyad
@patch("app.routes.action.Dyad.query")
def test_request_action_missing_dyad(mock_dyad_query, client):
    mock_dyad_query.filter_by.return_value.first.return_value = None
    data = deepcopy(test_data_json)
    data["dyad_id"] = "non_existent_dyad"
    response = client.post("/api/v1/action", json=data)
    assert response.status_code == 404
    assert response.json["message"] == "Dyad not found."

# Test request_action_success
def test_request_action_success(client):
    with patch("app.routes.action.Dyad.query") as mock_dyad_query, \
         patch("app.routes.action.StudyData.query") as mock_study_data_query:
        mock_dyad_query.filter_by.return_value.first.return_value = MagicMock()
        mock_study_data_query.filter_by.return_value.first.return_value = None

        data = deepcopy(test_data_json)
        response = client.post("/api/v1/action", json=data)
        print(response.json)
        assert response.status_code == 201
        assert response.json["status"] == "success"