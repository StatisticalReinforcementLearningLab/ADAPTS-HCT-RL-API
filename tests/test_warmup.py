"""
Server-side warm-up tests (API-Spec §3.2):
- /add_group takes no `warmup` field (the host cannot force/suppress warm-up)
- a decision is Bernoulli(0.5) while the cohort has < 5 dyads (cohort gate) or
  the dyad has had < 6 cp_message decisions (week-1 gate)
- warm-up actions report warmup=True, action_prob=0.5, state=null
"""

from app.models import Group
from tests.conftest import register_group, upload


def _aya_action(client, group_id, idx, ts="2026-01-06T09:00:00"):
    return client.post(
        "/api/v1/action",
        json={
            "group_id": group_id,
            "timestamp": ts,
            "decision_idx": idx,
            "decision_type": "aya_message",
        },
    )


def test_add_group_has_no_warmup_field(client):
    response = register_group(client, "dyad_001")
    assert response.status_code == 201
    assert "warmup" not in response.json
    with client.application.app_context():
        grp = Group.query.filter_by(group_id="dyad_001").first()
        assert grp is not None
        assert not hasattr(grp, "warmup")


def test_add_group_ignores_warmup_in_payload(client):
    # A stray warmup field is simply ignored (no longer part of the contract).
    response = client.post(
        "/api/v1/add_group",
        json={
            "group_id": "dyad_002",
            "member_list": ["a", "b"],
            "consent_start_date": "2026-01-05",
            "consent_end_date": "2026-04-15",
            "warmup": True,
        },
    )
    assert response.status_code == 201
    assert "warmup" not in response.json


def test_cohort_gate_warms_up_first_dyads(client):
    register_group(client, "dyad_001")
    upload(client, "dyad_001", "2026-01-06T08:00:00")
    r = _aya_action(client, "dyad_001", 0)
    assert r.status_code == 201
    assert r.json["warmup"] is True
    assert r.json["warmup_reason"] == "cohort"
    assert r.json["action_prob"] == 0.5
    assert r.json["state"] is None


def test_week1_gate_after_cohort_filled(client):
    # Five dyads registered -> cohort gate off. A fresh dyad with 0 CP
    # decisions is still warmed up by the week-1 gate.
    for i in range(5):
        register_group(client, f"dyad_{i:03d}")
    upload(client, "dyad_000", "2026-01-06T08:00:00")
    r = _aya_action(client, "dyad_000", 0)
    assert r.status_code == 201
    assert r.json["warmup"] is True
    assert r.json["warmup_reason"] == "week1"
    assert r.json["action_prob"] == 0.5


def test_warmup_action_distribution_is_balanced(client):
    register_group(client, "dyad_001")
    upload(client, "dyad_001", "2026-01-06T08:00:00")
    actions = []
    for i in range(200):
        r = _aya_action(client, "dyad_001", i)
        assert r.status_code == 201
        assert r.json["warmup"] is True
        actions.append(r.json["action"])
    mean_action = sum(actions) / len(actions)
    assert abs(mean_action - 0.5) < 0.15
