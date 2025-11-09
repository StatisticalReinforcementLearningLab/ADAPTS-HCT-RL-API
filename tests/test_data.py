test_dyad_json = {
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

def test_upload_data_success(client):
    """
    Tests uploading interaction data successfully.
    """
    # Add a dyad
    client.post(
        "/api/v1/add_dyad",
        json=test_dyad_json,
    )

    # Upload data
    response = client.post(
        "/api/v1/upload_data",
        json={
            "dyad_id": "test_dyad_123",
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


def test_upload_data_dyad_not_found(client):
    """
    Tests uploading data for a non-existent dyad.
    """
    response = client.post(
        "/api/v1/upload_data",
        json={
            "dyad_id": "non_existent_dyad",
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
    assert response.json["message"] == "Dyad not found."
