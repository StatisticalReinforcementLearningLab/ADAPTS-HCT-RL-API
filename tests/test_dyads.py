import pytest
from app.routes.dyad import check_fields, add_dyad
from app.models import Dyad, StudyData
from unittest.mock import patch, MagicMock

test_json = {
    "dyad_id": "test_dyad_123",
    "cp_id": "test_cp_123",
    "aya_id": "test_aya_123",
    "consent_start_date": "2025-01-01",
    "consent_end_date": "2025-01-01",
    "AM_MTW": 7,
    "PM_MTW": 17,
    "AM_CPW": 7,
    "PM_CPW": 17,
}

# Test check_fields for dyad route
def test_check_fields_dyad_missing_dyad_id():
    data = {}
    result, error_message = check_fields(data)
    assert not result
    assert "dyad_id is required." in error_message

def test_check_fields_dyad_valid():
    data = {"dyad_id": "test_dyad_123"}
    result, error_message = check_fields(data)
    assert result
    assert error_message == ""

def test_add_dyad_missing_field(client):
    """
    Tests adding a dyad without the required fields.
    """
    response = client.post(
        "/api/v1/add_dyad",
        json={
            "cp_id": "test_cp_123",
            "aya_id": "test_aya_123",
            "consent_start_date": "2025-01-01",
            "consent_end_date": "2025-01-01",
            "AM_MTW": 7,
            "PM_MTW": 17,
            "AM_CPW": 7,
            "PM_CPW": 17,},  # Missing `dyad_id`
    )

    assert response.status_code == 400
    assert response.json["message"] == "dyad_id is required."



def test_add_dyad_duplicate(client):
    """
    Tests adding a duplicate dyad.
    """
    # Add the dyad for the first time
    client.post(
        "/api/v1/add_dyad",
        json=test_json,
    )

    # Attempt to add the same dyad again
    response = client.post(
        "/api/v1/add_dyad",
        json=test_json,
    )

    assert response.status_code == 400
    assert response.json["message"] == "Dyad already exists."


def test_add_dyad_success(client):
    """
    Tests adding a dyad successfully.
    """

    response = client.post(
        "/api/v1/add_dyad",
        json=test_json,
    )

    print(response.json)

    assert response.status_code == 201
    assert response.json["dyad_id"] == "test_dyad_123"
    assert response.json["message"] == "Dyad added successfully."
