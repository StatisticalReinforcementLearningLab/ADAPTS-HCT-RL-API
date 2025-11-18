test_group_json = {
    "group_id": "test_group_123",
    "member_list": ["member1", "member2"],
    "consent_start_date": "2025-01-01",
    "consent_end_date": "2025-01-01",
}

def test_upload_data_success(client):
    """
    Tests uploading interaction data successfully.
    """
    # Add a group
    client.post(
        "/api/v1/add_group",
        json=test_group_json,
    )

    # Upload data
    response = client.post(
        "/api/v1/upload_data",
        json={
            "group_id": "test_group_123",
            "timestamp": "2024-01-01T12:00:00Z",
            "decision_idx": 0,
            "data": {
                "context": {"cur_var": 23, "past3_vars": [22.0, 21.0, 20.0]},
                "action": 1,
                "action_prob": 0.5,
                "state": [23, 22.0, 21.0, 20.0],
                "outcome": {"clicks": 4},
            },
        },
    )

    print(response.json)

    assert response.status_code == 201
    assert response.json["message"] == "Data uploaded successfully."


def test_upload_data_group_not_found(client):
    """
    Tests uploading data for a non-existent group.
    """
    response = client.post(
        "/api/v1/upload_data",
        json={
            "group_id": "non_existent_group",
            "timestamp": "2024-01-01T12:00:00Z",
            "decision_idx": 2,
            "data": {
                "context": {"cur_var": 30, "past3_vars": [29.0, 28.0, 27.0]},
                "action": 1,
                "action_prob": 0.5,
                "state": [30, 29.0, 28.0, 27.0],
                "outcome": {"clicks": 4},
            },
        },
    )
    print(response.json)

    assert response.status_code == 404
    assert response.json["message"] == "Group not found."
