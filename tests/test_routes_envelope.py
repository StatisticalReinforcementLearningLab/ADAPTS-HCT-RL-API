"""
The common envelope on every answer of every endpoint (API-Spec §2 preamble;
Monitoring-Spec §6.1, §6.7; contract §3.2, D19): ``status`` equal to the HTTP
status, nonempty ``title`` / ``message`` / ``rid``, an ISO-8601
``server_timestamp``, a fresh ``rid`` per call, and the same field set for
success and failure of one endpoint.
"""

from __future__ import annotations

import datetime

from app.clock import server_now
from tests.conftest import ENVELOPE_KEYS, assert_envelope, post_action, register_group, upload

GROUP_KEYS = set(ENVELOPE_KEYS) | {"group_id", "member_list", "consent_start_date", "consent_end_date"}
ACTION_KEYS = set(ENVELOPE_KEYS) | {
    "group_id", "decision_type", "decision_idx", "action", "action_prob", "warmup", "state", "model_param",
}


def test_every_endpoint_answer_carries_the_envelope(client):
    seen_rids = []

    def check(response, status, keys):
        body = assert_envelope(response, status)
        assert set(body) == keys, body
        seen_rids.append(body["rid"])
        return body

    # /register_group: 201, 201 (update), 400, 409
    check(register_group(client, "dyad_e"), 201, GROUP_KEYS)
    check(register_group(client, "dyad_e"), 201, GROUP_KEYS)
    check(client.post("/api/v1/register_group", json={"group_id": "x"}), 400, GROUP_KEYS)
    check(register_group(client, "dyad_e", members=["someone"]), 409, GROUP_KEYS)
    check(client.post("/api/v1/add_group", json={}), 400, GROUP_KEYS)

    # /action: 201 no-state, 201 warm-up, 200 replay, 400, 404
    check(post_action(client, "dyad_e", "cp_message", 0, "2026-01-06T07:35:00"), 201, ACTION_KEYS)
    upload(client, "dyad_e", "2026-01-06T07:40:00")
    check(post_action(client, "dyad_e", "cp_message", 1, "2026-01-06T07:45:00"), 201, ACTION_KEYS)
    replay = check(post_action(client, "dyad_e", "cp_message", 1, "2026-01-06T07:45:00"), 200, ACTION_KEYS)
    seen_rids.pop()  # a replay echoes the original rid by design
    assert replay["title"] == "Duplicate Decision"
    check(client.post("/api/v1/action", json={"group_id": "dyad_e"}), 400, ACTION_KEYS)
    check(post_action(client, "dyad_nope", "cp_message", 0, "2026-01-06T07:45:00"), 404, ACTION_KEYS)

    # /upload_data: 201, 400, 404 (envelope only)
    check(upload(client, "dyad_e", "2026-01-06T19:25:00"), 201, set(ENVELOPE_KEYS))
    check(client.post("/api/v1/upload_data", json={"group_id": "dyad_e"}), 400, set(ENVELOPE_KEYS))
    check(upload(client, "dyad_nope", "2026-01-06T19:25:00"), 404, set(ENVELOPE_KEYS))

    # /update: 202, 400 (envelope only)
    check(client.post("/api/v1/update", json={"timestamp": "2026-01-12T03:00:00"}), 202, set(ENVELOPE_KEYS))
    check(client.post("/api/v1/update", json={}), 400, set(ENVELOPE_KEYS))

    # Framework answers (D19).
    check(client.get("/api/v1/nope"), 404, set(ENVELOPE_KEYS))
    check(client.get("/api/v1/action"), 405, set(ENVELOPE_KEYS))
    check(client.put("/api/v1/upload_data"), 405, set(ENVELOPE_KEYS))

    assert len(seen_rids) == len(set(seen_rids)), "every call gets a fresh rid"


def test_framework_answer_texts(client):
    assert_envelope(client.get("/api/v1/does_not_exist"), 404, "Not Found", "Unknown route.")
    assert_envelope(client.get("/api/v1/register_group"), 405, "Method Not Allowed", "Method not allowed.")
    assert_envelope(client.delete("/api/v1/update"), 405, "Method Not Allowed", "Method not allowed.")


def test_server_timestamp_is_detroit_local_server_time(client):
    before = server_now() - datetime.timedelta(seconds=1)
    body = assert_envelope(client.post("/api/v1/update", json={}), 400)
    stamp = datetime.datetime.fromisoformat(body["server_timestamp"])
    assert stamp.tzinfo is None
    assert before <= stamp <= server_now() + datetime.timedelta(seconds=1)
