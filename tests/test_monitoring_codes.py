"""
WS-M tests of the check list, the error-code list and the after-response step
(Monitoring-Spec §5, §6, §6.9; implementation contract D17, D19).

Each §6 answer that can be produced through a real request is triggered and
checked for its ``api_error_events`` row (code, endpoint, status, title,
message, rid, ids, payload fingerprint and size, never the body) and its
alarm (every code except ``AC-200-REPLAY``).
"""

from __future__ import annotations

import hashlib

import pytest
from sqlalchemy.exc import OperationalError

from app.extensions import db
from app.monitoring import catalog, codes
from tests.test_monitoring_support import (  # noqa: F401 - fixtures
    AYA,
    MON1,
    D,
    alarms,
    app_factory,
    at,
    events,
    mon_app,
    mon_client,
    post_action,
    post_register,
    post_update,
    post_upload,
    raw_json,
    snapshot,
)

SECTION5_CHECKS = {
    "A13": ("red", "rl"), "A4": ("red", "rl"), "A10": ("red", "rl"), "A6": ("yellow", "rl"),
    "A7": ("yellow", "rl"), "A16": ("yellow", "host"), "A17": ("yellow", "rl"), "A2": ("yellow", "host"),
    "A8": ("yellow", "host"), "A15": ("yellow", "host"), "A5": ("yellow", "rl"), "B11": ("yellow", "host"),
    "B5": ("yellow", "rl"), "B8": ("yellow", "host"), "B3": ("green", "host"), "C7": ("yellow", "rl"),
    "C11": ("red", "host"), "C9": ("red", "rl"), "C3": ("yellow", "rl"), "C1": ("yellow", "rl"),
    "C5": ("yellow", "rl"), "C10": ("green", "rl"), "L1": ("green", "rl"), "L2": ("green", "rl"),
    "L3": ("yellow", "rl"), "L4": ("green", "rl"), "D1": ("green", "rl"), "D3": ("green", "rl"),
    "E1": ("red", "rl"), "E5": ("yellow", "rl"), "E6": ("yellow", "host"), "E7": ("yellow", "rl"),
    "G0": ("green", "rl"),
}
SECTION6_CODES = {
    "RG-400-PARAM": ("yellow", "host"), "RG-409-MEMBER": ("red", "host"), "RG-500-INTERNAL": ("yellow", "rl"),
    "RG-503-DB": ("yellow", "rl"), "AC-201-NOSTATE": ("yellow", "host"), "AC-201-FALLBACK": ("yellow", "rl"),
    "AC-400-PARAM": ("red", "host"), "AC-404-GROUP": ("red", "host"), "AC-500-PERSIST": ("red", "rl"),
    "AC-503-STORE": ("red", "rl"), "UD-400-PARAM": ("yellow", "host"), "UD-400-UNKNOWN": ("yellow", "host"),
    "UD-400-MISSINGFIELD": ("yellow", "host"), "UD-400-TYPE": ("yellow", "host"),
    "UD-400-RANGE": ("yellow", "host"), "UD-400-INCONSISTENT": ("yellow", "host"),
    "UD-404-GROUP": ("yellow", "host"), "UD-500-INTERNAL": ("yellow", "rl"), "UD-503-DB": ("yellow", "rl"),
    "UP-400-PARAM": ("yellow", "host"), "UP-503-ENQUEUE": ("red", "rl"), "MA-500-INTERNAL": ("yellow", "rl"),
    "MA-503-STORE": ("yellow", "rl"), "XX-404-ROUTE": ("yellow", "host"), "XX-405-METHOD": ("yellow", "host"),
    "XX-500-UNCAUGHT": ("red", "rl"),
}


# ------------------------------------------------------------------ catalog


def test_catalog_has_every_section5_check_with_its_severity_and_owner():
    assert set(catalog.CHECKS) == set(SECTION5_CHECKS)
    for item, (severity, owner) in SECTION5_CHECKS.items():
        entry = catalog.lookup(item)
        assert (entry.severity, entry.owner) == (severity, owner), item
        assert entry.runs and entry.where in catalog.SOURCES


def test_catalog_has_every_section6_code_with_its_severity_owner_and_fallback():
    for code, (severity, owner) in SECTION6_CODES.items():
        entry = catalog.lookup(code)
        assert (entry.severity, entry.owner, entry.kind, entry.alarm) == (severity, owner, "code", True), code
    assert catalog.lookup("AC-200-REPLAY").alarm is False
    assert catalog.lookup("UP-500-INTERNAL").owner == "rl"
    assert catalog.lookup("AC-400-PARAM").fallback == "F-A1"
    assert catalog.lookup("AC-201-FALLBACK").fallback == "F-A2/F-A3"
    assert {catalog.lookup(f"UD-400-{k}").fallback for k in
            ("PARAM", "UNKNOWN", "MISSINGFIELD", "TYPE", "RANGE", "INCONSISTENT")} == {"F-U1"}


def test_catalog_escalation_rules_and_unmapped_family():
    for item in ("C1", "C3", "C5", "B11"):
        assert "red" in catalog.lookup(item).escalation
    entry = catalog.lookup("XX-418-UNMAPPED")
    assert (entry.severity, entry.owner, entry.http_status) == ("yellow", "rl", 418)
    with pytest.raises(KeyError):
        catalog.lookup("Z9")
    assert catalog.CHECK_LIST_VERSION == "0.2"


# ---------------------------------------------------------------- classify


@pytest.mark.parametrize("path,status,title,message,code", [
    ("/api/v1/register_group", 201, "Success", "Group registered successfully.", None),
    ("/api/v1/add_group", 201, "Success", "Group consent window updated.", None),
    ("/api/v1/register_group", 400, "Invalid Parameter", "Request body is required.", "RG-400-PARAM"),
    ("/api/v1/register_group", 400, "Invalid Parameter", "consent_end_date is required.", "RG-400-PARAM"),
    ("/api/v1/register_group", 400, "Invalid Parameter",
     "consent_start_date must be a date in YYYY-MM-DD format; got '2026/01/05'.", "RG-400-PARAM"),
    ("/api/v1/register_group", 409, "Member Mismatch",
     "group_id d1 is already registered with members [a, b]; membership cannot be changed via /register_group.",
     "RG-409-MEMBER"),
    ("/api/v1/register_group", 500, "Internal Error", "Internal server error.", "RG-500-INTERNAL"),
    ("/api/v1/register_group", 503, "Service Unavailable", "Database write failed; retry with backoff.",
     "RG-503-DB"),
    ("/api/v1/action", 201, "Success", "Action requested successfully.", None),
    ("/api/v1/action", 201, "Success", "Action requested successfully using the no-state fallback.",
     "AC-201-NOSTATE"),
    ("/api/v1/action", 201, "Success", "Action requested successfully using a fallback.", "AC-201-FALLBACK"),
    ("/api/v1/action", 200, "Duplicate Decision", "Returning the existing action for (d1, aya_message, 3).",
     "AC-200-REPLAY"),
    ("/api/v1/action", 400, "Invalid Parameter", "decision_idx must be an integer.", "AC-400-PARAM"),
    ("/api/v1/action", 400, "Invalid Parameter",
     "Invalid decision_type. Must be 'aya_message', 'cp_message', or 'dyad_game'.", "AC-400-PARAM"),
    ("/api/v1/action", 404, "Unknown Group", "group_id d9 is not registered.", "AC-404-GROUP"),
    ("/api/v1/action", 500, "Internal Error", "Failed to persist the action.", "AC-500-PERSIST"),
    ("/api/v1/action", 503, "Service Unavailable", "The decision store is unavailable; use the host fallback.",
     "AC-503-STORE"),
    ("/api/v1/action", 500, "Internal Error", "Internal server error.", "XX-500-UNCAUGHT"),
    ("/api/v1/upload_data", 201, "Success", "Snapshot accepted.", None),
    ("/api/v1/upload_data", 400, "Invalid Parameter", "data must be a dictionary.", "UD-400-PARAM"),
    ("/api/v1/upload_data", 400, "Invalid Parameter", "unknown field 'aya_mood'.", "UD-400-UNKNOWN"),
    ("/api/v1/upload_data", 400, "Invalid Parameter", "slot is required (full snapshot).", "UD-400-MISSINGFIELD"),
    ("/api/v1/upload_data", 400, "Invalid Parameter", "slot: must be 'am' or 'pm'.", "UD-400-TYPE"),
    ("/api/v1/upload_data", 400, "Invalid Parameter", "aya_diary_mood: must be an integer from 1 through 5, or "
     "'miss'.", "UD-400-TYPE"),
    ("/api/v1/upload_data", 400, "Invalid Parameter", "aya_diary_mood must be between 1 and 5; got 7.",
     "UD-400-RANGE"),
    ("/api/v1/upload_data", 400, "Invalid Parameter",
     "week_in_study does not match day_in_study: expected 1, got 2.", "UD-400-INCONSISTENT"),
    ("/api/v1/upload_data", 404, "Unknown Group", "group_id d9 is not registered.", "UD-404-GROUP"),
    ("/api/v1/upload_data", 500, "Internal Error", "Internal server error.", "UD-500-INTERNAL"),
    ("/api/v1/upload_data", 503, "Service Unavailable", "Database write failed; retry.", "UD-503-DB"),
    ("/api/v1/update", 202, "Update Accepted", "Model update started.", None),
    ("/api/v1/update", 400, "Invalid Parameter", "timestamp is required.", "UP-400-PARAM"),
    ("/api/v1/update", 500, "Internal Error", "Internal server error.", "UP-500-INTERNAL"),
    ("/api/v1/update", 503, "Service Unavailable", "Could not start the update; retry.", "UP-503-ENQUEUE"),
    ("/api/v1/monitor/alarms", 200, "Success", "Returning 0 alarms.", None),
    ("/api/v1/monitor/events", 200, "Success", "Returning 12 alarms.", None),
    ("/api/v1/monitor/alarms", 500, "Internal Error", "Internal server error.", "MA-500-INTERNAL"),
    ("/api/v1/monitor/alarms", 503, "Service Unavailable", "Monitoring store unavailable; retry with backoff.",
     "MA-503-STORE"),
    ("/api/v1/nope", 404, "Not Found", "Unknown route.", "XX-404-ROUTE"),
    ("/api/v1/action", 405, "Method Not Allowed", "Method not allowed.", "XX-405-METHOD"),
    ("/api/v1/action", 418, "Teapot", "short and stout", "XX-418-UNMAPPED"),
    ("/api/v1/upload_data", 400, "Invalid Parameter", "something new.", "XX-400-UNMAPPED"),
    ("/api/v1/action", 201, "Success", "A new success wording.", "XX-201-UNMAPPED"),
])
def test_classify_maps_every_section6_row(path, status, title, message, code):
    assert codes.classify(path, status, title, message).code == code


def test_classify_flags_wordings_section6_does_not_document_yet():
    c = codes.classify("/api/v1/action", 400, "Invalid Parameter", "timestamp must be an ISO-8601 timestamp; got 'x'.")
    assert c.code == "AC-400-PARAM" and c.documented is False
    c = codes.classify("/api/v1/register_group", 400, "Invalid Parameter", "member_list must be a list.")
    assert c.code == "RG-400-PARAM" and c.documented is False
    assert codes.classify("/api/v1/action", 400, "Invalid Parameter", "group_id is required.").documented is True


# ----------------------------------------------------- after-response step


def _assert_logged(response, code, *, alarm=True, group_id=None, decision_type=None, decision_idx=None,
                   body: bytes | None = None):
    rid = response.get_json()["rid"]
    rows = [e for e in events(code) if e.rid == rid]
    assert len(rows) == 1, (code, [(e.code, e.rid) for e in events()])
    event = rows[0]
    payload = response.get_json()
    assert event.http_status == response.status_code == payload["status"]
    assert event.title == payload["title"] and event.message == payload["message"]
    assert event.endpoint.startswith("/api/v1/")
    assert (event.group_id, event.decision_type, event.decision_idx) == (group_id, decision_type, decision_idx)
    if body is not None:
        assert event.payload_sha256 == hashlib.sha256(body).hexdigest()
        assert event.payload_bytes == len(body)
    found = [a for a in alarms(code) if a.rid == rid]
    if alarm:
        assert len(found) == 1, code
        entry = catalog.lookup(code)
        assert (found[0].severity, found[0].owner, found[0].source) == (entry.severity, entry.owner, "error_code")
        assert found[0].check_list_version == "0.2"
    else:
        assert found == []
    return event


def test_registration_answers_are_logged(mon_client, monkeypatch):
    body = raw_json({"group_id": "g1"})
    r = mon_client.post("/api/v1/register_group", data=body, content_type="application/json")
    assert r.status_code == 400
    _assert_logged(r, "RG-400-PARAM", group_id="g1", body=body)

    assert post_register(mon_client, "g1", MON1).status_code == 201
    r = post_register(mon_client, "g1", MON1, members=["x", "y"])
    assert r.status_code == 409
    _assert_logged(r, "RG-409-MEMBER", group_id="g1")

    import app.routes.group as group_route

    class Broken:
        class query:  # noqa: N801 - mimics Model.query
            @staticmethod
            def filter_by(**_):
                raise OperationalError("SELECT", {}, Exception("db down"))

    monkeypatch.setattr(group_route, "Group", Broken)
    r = post_register(mon_client, "g2", MON1)
    assert r.status_code == 503
    _assert_logged(r, "RG-503-DB", group_id="g2")
    monkeypatch.undo()

    monkeypatch.setattr(group_route, "check_fields", lambda data: 1 / 0)
    r = post_register(mon_client, "g3", MON1)
    assert r.status_code == 500
    _assert_logged(r, "RG-500-INTERNAL", group_id="g3")


def test_success_answers_are_not_logged(mon_client):
    assert post_register(mon_client, "g1", MON1).status_code == 201
    assert post_upload(mon_client, "g1", MON1, at(MON1, 7, 25)).status_code == 201
    assert mon_client.get("/api/v1/monitor/alarms").status_code == 200
    assert events() == []


def test_action_answers_are_logged(mon_client, mon_app, monkeypatch):
    r = post_action(mon_client, "nobody", AYA, 0, at(MON1, 7, 30))
    assert r.status_code == 404
    _assert_logged(r, "AC-404-GROUP", group_id="nobody", decision_type=AYA, decision_idx=0)

    body = raw_json({"group_id": "g1", "decision_type": AYA, "decision_idx": True, "timestamp": "2026-01-05T07:30"})
    r = mon_client.post("/api/v1/action", data=body, content_type="application/json")
    assert r.status_code == 400
    _assert_logged(r, "AC-400-PARAM", group_id="g1", decision_type=AYA, decision_idx=None, body=body)

    assert post_register(mon_client, "g1", MON1).status_code == 201
    r = post_action(mon_client, "g1", AYA, 0, at(MON1, 7, 30))  # no upload yet: no-state fallback
    assert r.status_code == 201 and r.get_json()["warmup"] is True
    _assert_logged(r, "AC-201-NOSTATE", group_id="g1", decision_type=AYA, decision_idx=0)

    r2 = post_action(mon_client, "g1", AYA, 1, at(MON1, 19, 30))  # second no-state: logged, no second alarm
    assert r2.status_code == 201
    assert len([e for e in events("AC-201-NOSTATE") if e.rid == r2.get_json()["rid"]]) == 1
    assert len(alarms("AC-201-NOSTATE")) == 1

    r3 = post_action(mon_client, "g1", AYA, 0, at(MON1, 7, 30))  # replay
    assert r3.status_code == 200 and r3.get_json()["rid"] == r.get_json()["rid"]
    replay = [e for e in events("AC-200-REPLAY")]
    assert len(replay) == 1 and replay[0].rid == r.get_json()["rid"]
    assert alarms("AC-200-REPLAY") == []

    import app.routes.action as action_route

    monkeypatch.setattr(action_route, "_mint", lambda decision: 1 / 0)
    r = post_action(mon_client, "g1", AYA, 5, at(MON1, 19, 31))
    assert r.status_code == 500
    _assert_logged(r, "AC-500-PERSIST", group_id="g1", decision_type=AYA, decision_idx=5)
    monkeypatch.undo()

    class Broken:
        class query:  # noqa: N801
            @staticmethod
            def filter_by(**_):
                raise OperationalError("SELECT", {}, Exception("db down"))

    monkeypatch.setattr(action_route, "Group", Broken)
    r = post_action(mon_client, "g1", AYA, 6, at(MON1, 19, 32))
    assert r.status_code == 503
    _assert_logged(r, "AC-503-STORE", group_id="g1", decision_type=AYA, decision_idx=6)


def _learned_path_app(app_factory):
    """An app where a dyad that consented long ago is past every warm-up rule (n_warm = 2, two dyads)."""
    app = app_factory(HYPERPARAMS_OVERRIDES={"warmup.n_warm": 2})
    client = app.test_client()
    for gid in ("old1", "old2"):
        assert post_register(client, gid, D(2025, 11, 3)).status_code == 201
    assert post_upload(client, "old1", D(2025, 11, 3), at(D(2026, 1, 13), 7, 25)).status_code == 201
    return app, client


def test_fallback_answer_logged_with_reason_and_alarm(app_factory):
    _, client = _learned_path_app(app_factory)
    r = post_action(client, "old1", AYA, 0, at(D(2026, 1, 13), 7, 30))  # learned path, no policy in force
    body = r.get_json()
    assert r.status_code == 201 and body["message"].endswith("using a fallback."), body
    event = _assert_logged(r, "AC-201-FALLBACK", group_id="old1", decision_type=AYA, decision_idx=0)
    alarm = [a for a in alarms("AC-201-FALLBACK") if a.rid == event.rid][0]
    assert alarm.detail["reason"] == "no_policy" and alarm.fallback_used == "F-A3"


def test_impossible_pi_raises_a13_not_the_fallback_alarm(app_factory, monkeypatch):
    from app import learner

    app, client = _learned_path_app(app_factory)

    def fake(group_id, decision_type, design_state, decision_time):
        return learner.PolicyDecision(action_prob=0.95, mu=3.0, v=0.01, model_param=[0.0], policy_version_id=1,
                                      policy_week=D(2026, 1, 12), model_parameters_id=None,
                                      hyperparams_hash=app.hyperparams.sha256)

    monkeypatch.setattr(learner, "learned_probability", fake)
    r = post_action(client, "old1", AYA, 0, at(D(2026, 1, 13), 7, 30))
    body = r.get_json()
    assert r.status_code == 201 and body["action_prob"] == 0.5 and body["warmup"] is True
    event = [e for e in events("AC-201-FALLBACK") if e.rid == body["rid"]]
    assert len(event) == 1
    assert [a for a in alarms("AC-201-FALLBACK")] == []
    a13 = alarms("A13")
    assert len(a13) == 1 and a13[0].severity == "red" and a13[0].source == "inside_action"
    assert a13[0].rid == body["rid"] and a13[0].fallback_used == "F-A2" and a13[0].value == 0.95


def test_upload_answers_are_logged(mon_client, monkeypatch):
    assert post_register(mon_client, "g1", MON1).status_code == 201
    ts = at(MON1, 7, 25)

    def post(data, gid="g1"):
        body = raw_json({"group_id": gid, "timestamp": ts.isoformat(), "data": data})
        return mon_client.post("/api/v1/upload_data", data=body, content_type="application/json"), body

    cases = [
        ("UD-400-PARAM", "not a dict"),
        ("UD-400-UNKNOWN", {**snapshot(MON1, ts), "aya_mood": 3}),
        ("UD-400-MISSINGFIELD", {k: v for k, v in snapshot(MON1, ts).items() if k != "slot"}),
        ("UD-400-TYPE", snapshot(MON1, ts, slot="noon")),
        ("UD-400-RANGE", snapshot(MON1, ts, aya_diary_mood=7)),
        ("UD-400-INCONSISTENT", snapshot(MON1, ts, week_in_study=3)),
    ]
    for code, data in cases:
        r, body = post(data)
        assert r.status_code == 400, (code, r.get_json())
        event = _assert_logged(r, code, group_id="g1", body=body)
        alarm = [a for a in alarms(code) if a.rid == event.rid][0]
        assert alarm.fallback_used == "F-U1"
        assert "7" not in alarm.message  # the rejected value stays out of the alarm (§2.2)
    r, body = post(snapshot(MON1, ts), gid="nobody")
    assert r.status_code == 404
    _assert_logged(r, "UD-404-GROUP", group_id="nobody", body=body)

    import app.routes.data as data_route

    monkeypatch.setattr(data_route, "check_request", lambda body: 1 / 0)
    r, _ = post(snapshot(MON1, ts))
    assert r.status_code == 500
    _assert_logged(r, "UD-500-INTERNAL", group_id="g1")
    monkeypatch.undo()

    class Broken:
        class query:  # noqa: N801
            @staticmethod
            def filter_by(**_):
                raise OperationalError("SELECT", {}, Exception("db down"))

    monkeypatch.setattr(data_route, "Group", Broken)
    r, _ = post(snapshot(MON1, ts))
    assert r.status_code == 503
    _assert_logged(r, "UD-503-DB", group_id="g1")


def test_update_answers_are_logged(mon_client, monkeypatch):
    r = mon_client.post("/api/v1/update", json={})
    assert r.status_code == 400
    _assert_logged(r, "UP-400-PARAM")

    import app.routes.update as update_route

    def broken(*args, **kwargs):
        raise OperationalError("INSERT", {}, Exception("db down"))

    monkeypatch.setattr(update_route, "ModelUpdateRequests", broken)
    r = post_update(mon_client, at(MON1, 3))
    assert r.status_code == 503
    _assert_logged(r, "UP-503-ENQUEUE")
    monkeypatch.undo()

    monkeypatch.setattr(update_route, "check_fields", lambda data: 1 / 0)
    r = post_update(mon_client, at(MON1, 3))
    assert r.status_code == 500
    _assert_logged(r, "UP-500-INTERNAL")


def test_alarm_endpoint_failures_are_logged(mon_client, monkeypatch):
    import app.monitoring.blueprint as bp

    def unreadable(now):
        raise OperationalError("SELECT", {}, Exception("db down"))

    monkeypatch.setattr(bp, "read_alarm_window", unreadable)
    r = mon_client.get("/api/v1/monitor/alarms")
    assert r.status_code == 503
    _assert_logged(r, "MA-503-STORE")
    monkeypatch.undo()

    monkeypatch.setattr(bp, "alarm_feed_record", lambda row: 1 / 0)
    from app.monitoring import raise_alarm

    raise_alarm("G0", message="status", source="scheduled")
    r = mon_client.get("/api/v1/monitor/alarms")
    assert r.status_code == 500
    _assert_logged(r, "MA-500-INTERNAL")


def test_framework_answers_are_logged(mon_app, mon_client):
    def boom():
        raise RuntimeError("uncaught")

    mon_app.add_url_rule("/api/v1/test_boom", "test_boom", boom, methods=["POST"])
    r = mon_client.get("/api/v1/nope")
    assert r.status_code == 404
    _assert_logged(r, "XX-404-ROUTE")
    r = mon_client.get("/api/v1/action")
    assert r.status_code == 405
    _assert_logged(r, "XX-405-METHOD")
    r = mon_client.post("/api/v1/test_boom", json={"group_id": "g1"})
    assert r.status_code == 500 and r.get_json()["title"] == "Internal Error"
    _assert_logged(r, "XX-500-UNCAUGHT", group_id="g1")


def test_unmapped_answer_gets_unmapped_code_and_alarm(mon_app, mon_client):
    from app.routes.envelope import envelope, new_rid

    mon_app.add_url_rule("/api/v1/teapot", "teapot", lambda: envelope(418, "Teapot", "short and stout", new_rid()))
    r = mon_client.get("/api/v1/teapot")
    assert r.status_code == 418
    event = _assert_logged(r, "XX-418-UNMAPPED")
    assert event.endpoint == "/api/v1/teapot"


def test_paths_outside_the_api_are_not_logged(mon_client):
    assert mon_client.get("/nope").status_code == 404
    assert events() == []


def test_after_response_failure_is_e7_and_never_changes_the_answer(mon_client, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("classifier bug")

    monkeypatch.setattr(codes, "classify", broken)
    r = mon_client.get("/api/v1/nope")
    assert r.status_code == 404 and r.get_json()["message"] == "Unknown route."
    e7 = alarms("E7")
    assert len(e7) == 1 and e7[0].detail["where"] == "after_response"
    assert "classifier bug" not in e7[0].message and "classifier bug" not in str(e7[0].detail)


def test_after_response_survives_a_dead_store(mon_client, monkeypatch):
    """E7 itself cannot be written when the database is gone: only the log remains; the answer is unchanged."""
    import app.monitoring.hooks as hooks

    def dead(*args, **kwargs):
        raise OperationalError("INSERT", {}, Exception("db down"))

    monkeypatch.setattr(hooks, "record_answer", dead)
    monkeypatch.setattr(db.session, "commit", dead)
    r = mon_client.get("/api/v1/nope")
    assert r.status_code == 404
    monkeypatch.undo()
    assert events() == []
