"""
WS-M tests of alarm writing and the host's alarm feed (Monitoring-Spec §4,
§8.1, §9.1, M4, M7; API-Spec §2.5, §5.10, §5.11).
"""

from __future__ import annotations

import datetime

from app.extensions import db
from app.models import MonitorAlarm, MonitorPull
from app.monitoring import raise_alarm
from app.monitoring.alarms import add_note
from tests.test_monitoring_support import (  # noqa: F401 - fixtures
    MON1,
    alarms,
    at,
    mon_app,
    mon_client,
    post_register,
    post_upload,
)


def test_raise_alarm_uses_catalog_defaults_and_check_list_version(mon_app):
    row = raise_alarm("C11", message="no update", source="scheduled", raised_at=at(MON1, 3, 30),
                      detail={"study_week_monday": MON1, "x": float("nan")})
    assert row is not None
    stored = MonitorAlarm.query.filter_by(event_id=row.event_id).one()
    assert (stored.severity, stored.owner, stored.check_list_version) == ("red", "host", "0.2")
    assert stored.detail == {"study_week_monday": "2026-01-05", "x": "nan"}
    assert len(stored.event_id) == 36 and stored.raised_at == at(MON1, 3, 30)


def test_raise_alarm_severity_override_and_dedup(mon_app):
    first = raise_alarm("B11", message="m", source="scheduled", dedup_key="B11:k", severity="red")
    assert first is not None and first.severity == "red" and first.owner == "host"
    assert raise_alarm("B11", message="again", source="scheduled", dedup_key="B11:k") is None
    assert len(alarms("B11")) == 1
    assert raise_alarm("B11", message="other finding", source="scheduled", dedup_key="B11:k2") is not None
    assert len(alarms("B11")) == 2


def test_raise_alarm_never_raises_and_records_e7(mon_app):
    assert raise_alarm("NOT-A-CHECK", message="m", source="scheduled") is None
    assert raise_alarm("A13", message="m", source="not_a_source") is None
    e7 = alarms("E7")
    assert len(e7) == 2 and all(a.severity == "yellow" and a.owner == "rl" for a in e7)
    assert alarms("A13") == []


def test_long_dedup_keys_are_hashed(mon_app):
    key = "A2:" + "x" * 400
    assert raise_alarm("A2", message="m", source="scheduled", dedup_key=key) is not None
    assert raise_alarm("A2", message="m", source="scheduled", dedup_key=key) is None
    assert len(alarms("A2")[0].dedup_key) <= 255


def test_notes_are_appended_never_replaced(mon_app):
    row = raise_alarm("A4", message="m", source="scheduled")
    add_note(row.event_id, "investigated", "alice")
    add_note(row.event_id, "fixed in release 1.2", "bob")
    stored = MonitorAlarm.query.filter_by(event_id=row.event_id).one()
    assert stored.note.startswith("investigated") and "fixed in release 1.2" in stored.note
    assert stored.note_by == "bob" and stored.noted_at is not None
    assert stored.message == "m"


# ------------------------------------------------------------------ the feed


def _set_clock(client, ts):
    """Advance the host clock (CLOCK_MODE = "host") with one upload stamped at ``ts``."""
    from app.models import Group

    if Group.query.filter_by(group_id="clock").first() is None:
        assert post_register(client, "clock", MON1).status_code == 201
    assert post_upload(client, "clock", MON1, ts).status_code == 201


def test_feed_returns_the_rolling_24_hours_with_owner_and_offset(mon_app, mon_client):
    now = at(datetime.date(2026, 1, 13), 5, 50)
    old = raise_alarm("C11", message="old", source="scheduled", raised_at=now - datetime.timedelta(hours=25))
    edge = raise_alarm("C7", message="edge", source="scheduled", raised_at=now - datetime.timedelta(hours=24))
    recent = raise_alarm("A8", message="recent", source="after_response", raised_at=now - datetime.timedelta(hours=1))
    future = raise_alarm("A2", message="future", source="scheduled", raised_at=now + datetime.timedelta(hours=1))
    _set_clock(mon_client, now)

    r = mon_client.get("/api/v1/monitor/alarms")
    body = r.get_json()
    assert r.status_code == 200 and body["title"] == "Success"
    assert body["message"] == "Returning 2 alarms." and body["count"] == 2
    ids = [a["event_id"] for a in body["alarms"]]
    assert ids == [edge.event_id, recent.event_id]
    assert old.event_id not in ids and future.event_id not in ids
    first = body["alarms"][1]
    assert set(first) == {"event_id", "item_id", "severity", "owner", "raised_at", "message", "detail"}
    assert first["owner"] == "host" and first["severity"] == "yellow" and first["item_id"] == "A8"
    assert first["raised_at"] == "2026-01-13T04:50:00-05:00"
    assert "host_action" not in first and "heartbeat" not in body


def test_feed_offset_follows_daylight_saving(mon_app, mon_client):
    summer = datetime.datetime(2026, 7, 6, 5, 45)
    raise_alarm("C7", message="summer", source="scheduled", raised_at=summer)
    _set_clock(mon_client, summer + datetime.timedelta(minutes=5))
    alarm = mon_client.get("/api/v1/monitor/alarms").get_json()["alarms"][0]
    assert alarm["raised_at"] == "2026-07-06T05:45:00-04:00"


def test_overlapping_pulls_repeat_event_ids_for_host_dedup(mon_app, mon_client):
    now = at(datetime.date(2026, 1, 13), 5, 50)
    raise_alarm("C7", message="m", source="scheduled", raised_at=now - datetime.timedelta(hours=2))
    _set_clock(mon_client, now)
    a = mon_client.get("/api/v1/monitor/alarms").get_json()
    b = mon_client.get("/api/v1/monitor/alarms").get_json()
    assert [x["event_id"] for x in a["alarms"]] == [x["event_id"] for x in b["alarms"]]
    assert a["rid"] != b["rid"]


def test_every_pull_is_recorded_including_failures(mon_app, mon_client, monkeypatch):
    now = at(datetime.date(2026, 1, 13), 5, 50)
    _set_clock(mon_client, now)
    ok = mon_client.get("/api/v1/monitor/alarms").get_json()
    alias = mon_client.get("/api/v1/monitor/events").get_json()
    assert alias["status"] == 200 and alias["message"] == "Returning 0 alarms."

    import app.monitoring.blueprint as bp
    from sqlalchemy.exc import OperationalError

    def unreadable(now):
        raise OperationalError("SELECT", {}, Exception("db down"))

    monkeypatch.setattr(bp, "read_alarm_window", unreadable)
    failed = mon_client.get("/api/v1/monitor/alarms")
    body = failed.get_json()
    assert failed.status_code == 503 and body["title"] == "Service Unavailable"
    assert body["message"] == "Monitoring store unavailable; retry with backoff."
    assert body["count"] is None and body["alarms"] is None and body["rid"]

    pulls = {p.rid: p for p in MonitorPull.query.all()}
    assert pulls[ok["rid"]].status == 200 and pulls[ok["rid"]].returned_count == 0
    assert pulls[ok["rid"]].window_start == now - datetime.timedelta(hours=24)
    assert pulls[alias["rid"]].status == 200
    assert pulls[body["rid"]].status == 503 and pulls[body["rid"]].returned_count is None


def test_internal_error_answer_shape(mon_app, mon_client, monkeypatch):
    import app.monitoring.blueprint as bp

    raise_alarm("G0", message="m", source="scheduled")
    monkeypatch.setattr(bp, "alarm_feed_record", lambda row: 1 / 0)
    r = mon_client.get("/api/v1/monitor/alarms")
    body = r.get_json()
    assert r.status_code == 500 and body["title"] == "Internal Error" and body["message"] == "Internal server error."
    assert body["count"] is None and body["alarms"] is None
    assert MonitorPull.query.filter_by(rid=body["rid"]).one().status == 500


def test_feed_window_follows_the_latest_monitoring_run_in_host_mode(mon_app, mon_client):
    """In host mode a run evaluated at 05:45 makes its alarms visible to a pull made right after it."""
    from app.monitoring import run_monitoring

    _set_clock(mon_client, at(datetime.date(2026, 1, 12), 3, 0))
    run_monitoring(now=at(datetime.date(2026, 1, 12), 5, 45), since=at(datetime.date(2026, 1, 12), 5, 40))
    raise_alarm("C7", message="m", source="scheduled", raised_at=at(datetime.date(2026, 1, 12), 5, 45))
    body = mon_client.get("/api/v1/monitor/alarms").get_json()
    assert "C7" in [a["item_id"] for a in body["alarms"]]
    db.session.remove()
