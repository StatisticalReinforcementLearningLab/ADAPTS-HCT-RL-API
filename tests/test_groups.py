import pytest
from app.routes.group import check_fields, register_group
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

def test_register_group_missing_field(client):
    """
    Tests registering a group without the required fields.
    """
    response = client.post(
        "/api/v1/register_group",
        json={
            "member_list": ["member1", "member2"],
            "consent_start_date": "2025-01-01",
            "consent_end_date": "2025-01-01",
        },  # Missing `group_id`
    )

    assert response.status_code == 400
    assert response.json["message"] == "group_id is required."



def test_register_group_reregister_updates_consent(client):
    """
    Re-registering an existing group updates the consent window (upsert) and
    leaves member_list unchanged. It returns 201, not 400.
    """
    # Register the group for the first time
    first = client.post(
        "/api/v1/register_group",
        json=test_json,
    )
    assert first.status_code == 201

    # Re-register the same group with a new consent window and a different
    # member_list, which should be ignored.
    response = client.post(
        "/api/v1/register_group",
        json={
            "group_id": "test_group_123",
            "member_list": ["someone_else"],
            "consent_start_date": "2025-02-01",
            "consent_end_date": "2025-05-01",
        },
    )

    assert response.status_code == 201
    assert response.json["message"] == "Group consent window updated."

    # The persisted consent dates changed; member_list is unchanged.
    grp = Group.query.filter_by(group_id="test_group_123").first()
    assert grp.group_info["consent_start_date"] == "2025-02-01"
    assert grp.group_info["consent_end_date"] == "2025-05-01"
    assert grp.group_info["member_list"] == ["member1", "member2"]

    # No duplicate row was created.
    assert Group.query.filter_by(group_id="test_group_123").count() == 1


def test_register_group_success(client):
    """
    Tests registering a group successfully.
    """

    response = client.post(
        "/api/v1/register_group",
        json=test_json,
    )

    print(response.json)

    assert response.status_code == 201
    assert response.json["group_id"] == "test_group_123"
    assert response.json["message"] == "Group registered successfully."


def test_add_group_alias_still_works(client):
    """
    The deprecated /add_group path maps to the same handler.
    """
    response = client.post(
        "/api/v1/add_group",
        json=test_json,
    )

    assert response.status_code == 201
    assert response.json["group_id"] == "test_group_123"
