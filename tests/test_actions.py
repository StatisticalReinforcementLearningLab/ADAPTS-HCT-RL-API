"""
``POST /api/v1/action`` request handling (API-Spec §2.2; Monitoring-Spec §6.3;
contract D11, D12): every 400 message with the offending field echoed as null,
404, the no-state fallback, scheduled warm-up rows and the idempotent replay.
The learned path and the other fallbacks are in ``test_routes_action.py``.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from app.models import Action
from app.routes.action import check_fields
from tests.conftest import assert_envelope, post_action, register_group, upload

GID = "test_group_123"
REQUEST = {
    "group_id": GID,
    "timestamp": "2026-01-06T09:00:00",
    "decision_idx": 1,
    "decision_type": "aya_message",
}
OUTPUT_KEYS = ("group_id", "decision_type", "decision_idx", "action", "action_prob", "warmup", "state", "model_param")


def _without(key):
    data = deepcopy(REQUEST)
    data.pop(key)
    return data


def _with(**kw):
    return {**deepcopy(REQUEST), **kw}


# ------------------------------------------------------------- check_fields


def test_check_fields_valid():
    assert check_fields(deepcopy(REQUEST)) == (True, "", None)


@pytest.mark.parametrize(
    "data, message, field",
    [
        (None, "Request body is required.", None),
        ("a string", "Request body is required.", None),
        (_without("group_id"), "group_id is required.", "group_id"),
        (_with(group_id=123), "group_id must be a string.", "group_id"),
        (_without("timestamp"), "timestamp is required.", "timestamp"),
        (_with(timestamp=20260106), "timestamp must be a string or datetime object.", "timestamp"),
        (_with(timestamp="not-a-time"), "timestamp must be an ISO-8601 timestamp; got 'not-a-time'.", "timestamp"),
        (_without("decision_idx"), "decision_idx is required.", "decision_idx"),
        (_with(decision_idx="1"), "decision_idx must be an integer.", "decision_idx"),
        (_with(decision_idx=1.0), "decision_idx must be an integer.", "decision_idx"),
        (_with(decision_idx=True), "decision_idx must be an integer.", "decision_idx"),
        (_without("decision_type"), "decision_type is required.", "decision_type"),
        (_with(decision_type=3), "decision_type must be a string.", "decision_type"),
        (_with(decision_type="aya_msg"),
         "Invalid decision_type. Must be 'aya_message', 'cp_message', or 'dyad_game'.", "decision_type"),
    ],
)
def test_check_fields_errors(data, message, field):
    assert check_fields(data) == (False, message, field)


# ------------------------------------------------------------------ 400 / 404


def test_400_nulls_the_offending_field_and_outputs(client):
    register_group(client, GID)
    out = assert_envelope(client.post("/api/v1/action", json=_with(decision_type="aya_msg")), 400,
                          "Invalid Parameter",
                          "Invalid decision_type. Must be 'aya_message', 'cp_message', or 'dyad_game'.")
    assert out["decision_type"] is None
    assert out["group_id"] == GID
    assert out["decision_idx"] == 1
    for key in ("action", "action_prob", "warmup", "state", "model_param"):
        assert out[key] is None
    assert Action.query.count() == 0


def test_empty_body_is_400(client):
    response = client.post("/api/v1/action", data=b"", content_type="application/json")
    out = assert_envelope(response, 400, "Invalid Parameter", "Request body is required.")
    assert all(out[k] is None for k in OUTPUT_KEYS)


def test_unknown_group_is_404(client):
    out = assert_envelope(client.post("/api/v1/action", json=deepcopy(REQUEST)), 404, "Unknown Group",
                          f"group_id {GID} is not registered.")
    assert out["group_id"] is None
    assert out["decision_type"] == "aya_message"
    assert out["decision_idx"] == 1
    for key in ("action", "action_prob", "warmup", "state", "model_param"):
        assert out[key] is None


# ---------------------------------------------------------- no-state fallback


def test_no_upload_returns_the_no_state_fallback(client, app):
    register_group(client, GID)
    cursor = app.sampler.cursor()
    out = assert_envelope(client.post("/api/v1/action", json=deepcopy(REQUEST)), 201, "Success",
                          "Action requested successfully using the no-state fallback.")
    assert out["action"] in (0, 1)
    assert out["action_prob"] == 0.5
    assert out["warmup"] is True
    assert out["state"] is None
    assert out["model_param"] is None
    assert out["group_id"] == GID and out["decision_type"] == "aya_message" and out["decision_idx"] == 1

    row = Action.query.filter_by(rid=out["rid"]).one()
    assert row.is_warmup is True
    assert row.warmup_reason == "no_state"
    assert row.raw_context is None
    assert row.state is None
    assert row.model_parameters_id is None
    assert row.upload_id is None
    assert row.hyperparams_hash == app.hyperparams.sha256
    # Exactly one uniform, no normals (D11), and action = int(u < 0.5).
    rs = row.random_state
    assert rs["mode"] == "fallback"
    assert rs["sampler_cursor_start"] == cursor
    assert rs["sampler_cursor_end"] == {"normal": cursor["normal"], "uniform": cursor["uniform"] + 1}
    assert row.action == int(rs["u"] < 0.5)


def test_no_state_takes_precedence_over_scheduled_warmup(client):
    # Cohort warm-up is active (one dyad), but without any upload the reason is no_state.
    register_group(client, GID)
    out = assert_envelope(client.post("/api/v1/action", json=deepcopy(REQUEST)), 201)
    assert Action.query.filter_by(rid=out["rid"]).one().warmup_reason == "no_state"


# ------------------------------------------------------- scheduled warm-up


def test_scheduled_warmup_row(client, app):
    register_group(client, GID)
    up = upload(client, GID, "2026-01-06T08:00:00")
    assert up.status_code == 201
    out = assert_envelope(client.post("/api/v1/action", json=deepcopy(REQUEST)), 201, "Success",
                          "Action requested successfully.")
    assert out["warmup"] is True
    assert out["action_prob"] == 0.5
    assert out["state"] is None
    assert out["model_param"] is None
    assert "warmup_reason" not in out  # the reason stays on the actions row

    row = Action.query.filter_by(rid=out["rid"]).one()
    assert row.warmup_reason == "cohort"
    assert row.state is None
    assert row.random_state["mode"] == "warmup"
    assert row.upload_id is not None
    assert row.raw_context["upload_rid"] == up.get_json()["rid"]
    # The design state is kept so the warm-up decision enters learning (D7, D8),
    # next to the legacy projection keys.
    assert row.raw_context["design_state"]["upload_id"] == row.upload_id
    assert row.raw_context["slot"] == "am"
    assert row.raw_context["agent_decision_index"] == 2
    assert row.model_parameters_id is None and row.policy_version_id is None


def test_warmup_decision_survives_a_design_state_failure(client, monkeypatch):
    from app import learner

    def broken(*args, **kwargs):
        raise ValueError("bad snapshot")

    monkeypatch.setattr(learner, "build_design_state", broken)
    register_group(client, GID)
    upload(client, GID, "2026-01-06T08:00:00")
    out = assert_envelope(client.post("/api/v1/action", json=deepcopy(REQUEST)), 201, "Success",
                          "Action requested successfully.")
    row = Action.query.filter_by(rid=out["rid"]).one()
    assert row.warmup_reason == "cohort"
    assert "design_state" not in row.raw_context
    assert row.raw_context["design_state_error"].startswith("ValueError")


# ------------------------------------------------------------------ replay


def test_replay_returns_the_original_decision_without_a_draw(client, app):
    register_group(client, GID)
    upload(client, GID, "2026-01-06T08:00:00")
    first = assert_envelope(client.post("/api/v1/action", json=deepcopy(REQUEST)), 201)
    cursor = app.sampler.cursor()
    second = assert_envelope(
        client.post("/api/v1/action", json=_with(timestamp="2026-01-06T09:05:00")), 200, "Duplicate Decision",
        f"Returning the existing action for ({GID}, aya_message, 1).",
    )
    assert second["rid"] == first["rid"]
    for key in OUTPUT_KEYS:
        assert second[key] == first[key]
    assert app.sampler.cursor() == cursor
    assert Action.query.count() == 1


def test_same_index_for_another_agent_is_a_new_decision(client):
    register_group(client, GID)
    upload(client, GID, "2026-01-06T08:00:00")
    a = assert_envelope(post_action(client, GID, "aya_message", 1, "2026-01-06T09:00:00"), 201)
    c = assert_envelope(post_action(client, GID, "cp_message", 1, "2026-01-06T09:00:00"), 201)
    assert a["rid"] != c["rid"]
    assert Action.query.count() == 2


def test_warmup_actions_are_balanced(client):
    register_group(client, GID)
    upload(client, GID, "2026-01-06T08:00:00")
    actions = [
        assert_envelope(post_action(client, GID, "aya_message", i, "2026-01-06T09:00:00"), 201)["action"]
        for i in range(200)
    ]
    assert abs(sum(actions) / len(actions) - 0.5) < 0.15
