import pytest
from app.routes.group import check_fields, add_group
from app.models import Group, StudyData
from unittest.mock import patch, MagicMock

test_json = {
    "group_id": "test_group_123",
    "member_list": ["member1", "member2"],
    "consent_start_date": "2025-01-01",
    "consent_end_date": "2025-01-01",
}

# Test check_fields for group route
def test_check_fields_group_missing_group_id():
    data = {}
    result, error_message = check_fields(data)
    assert not result
    assert "group_id is required." in error_message

def test_check_fields_group_valid():
    data = {
        "group_id": "test_group_123",
        "member_list": ["member1", "member2"],
        "consent_start_date": "2025-01-01",
        "consent_end_date": "2025-01-01",
    }
    result, error_message = check_fields(data)
    assert result
    assert error_message == ""

def test_add_group_missing_field(client):
    """
    Tests adding a group without the required fields.
    """
    response = client.post(
        "/api/v1/add_group",
        json={
            "member_list": ["member1", "member2"],
            "consent_start_date": "2025-01-01",
            "consent_end_date": "2025-01-01",
        },  # Missing `group_id`
    )

    assert response.status_code == 400
    assert response.json["message"] == "group_id is required."



def test_add_group_duplicate(client):
    """
    Tests adding a duplicate group.
    """
    # Add the group for the first time
    client.post(
        "/api/v1/add_group",
        json=test_json,
    )

    # Attempt to add the same group again
    response = client.post(
        "/api/v1/add_group",
        json=test_json,
    )

    assert response.status_code == 400
    assert response.json["message"] == "Group already exists."


def test_add_group_success(client):
    """
    Tests adding a group successfully.
    """

    response = client.post(
        "/api/v1/add_group",
        json=test_json,
    )

    print(response.json)

    assert response.status_code == 201
    assert response.json["group_id"] == "test_group_123"
    assert response.json["message"] == "Group added successfully."

