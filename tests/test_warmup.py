"""
Warmup behavior tests:
- the /add_group route accepts a `warmup` flag and persists it on the Group
- the EB algorithm returns Bernoulli(0.5) actions for warmup dyads, ignoring
  any state input
"""

from copy import deepcopy

from app.models import Group


warmup_group_payload = {
    "group_id": "warmup_dyad_001",
    "member_list": ["aya_001", "cp_001"],
    "consent_start_date": "2025-01-05",
    "consent_end_date": "2025-04-14",
    "warmup": True,
}


standard_group_payload = {
    "group_id": "standard_dyad_001",
    "member_list": ["aya_001", "cp_001"],
    "consent_start_date": "2025-01-05",
    "consent_end_date": "2025-04-14",
}


aya_action_payload = {
    "group_id": "warmup_dyad_001",
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


def test_add_group_accepts_warmup_true(client):
    response = client.post("/api/v1/add_group", json=deepcopy(warmup_group_payload))
    assert response.status_code == 201
    assert response.json["warmup"] is True

    with client.application.app_context():
        grp = Group.query.filter_by(group_id="warmup_dyad_001").first()
        assert grp is not None
        assert grp.warmup is True


def test_add_group_defaults_warmup_false(client):
    response = client.post("/api/v1/add_group", json=deepcopy(standard_group_payload))
    assert response.status_code == 201
    assert response.json["warmup"] is False

    with client.application.app_context():
        grp = Group.query.filter_by(group_id="standard_dyad_001").first()
        assert grp is not None
        assert grp.warmup is False


def test_add_group_rejects_non_bool_warmup(client):
    bad = deepcopy(warmup_group_payload)
    bad["warmup"] = "yes"
    response = client.post("/api/v1/add_group", json=bad)
    assert response.status_code == 400
    assert "warmup" in response.json["message"]


def test_warmup_action_has_prob_one_half(client):
    client.post("/api/v1/add_group", json=deepcopy(warmup_group_payload))
    response = client.post("/api/v1/action", json=deepcopy(aya_action_payload))
    assert response.status_code == 201
    assert response.json["action"] in (0, 1)
    # Warmup contract: action prob is exactly 0.5 (purely-randomized).
    assert response.json["action_prob"] == 0.5


def test_warmup_action_distribution_is_balanced(client):
    """1000 warmup decisions should sit close to 0.5 mean."""
    client.post("/api/v1/add_group", json=deepcopy(warmup_group_payload))

    actions = []
    for i in range(1, 201):
        payload = deepcopy(aya_action_payload)
        payload["decision_idx"] = i
        payload["context"]["agent_decision_index"] = i
        r = client.post("/api/v1/action", json=payload)
        assert r.status_code == 201
        actions.append(r.json["action"])

    mean_action = sum(actions) / len(actions)
    assert abs(mean_action - 0.5) < 0.15  # generous bound for 200 trials
