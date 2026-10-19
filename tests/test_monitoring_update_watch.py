"""
WS-M tests of the weekly update watch with a simulated clock (Monitoring-Spec
§11.3 item 2; §5.C C1, C7, C11; B11; §7 F-U2, F-U4; §9.4 ``update_weeks``).

The host's calls go through the real routes (``CLOCK_MODE = "host"``: the
request timestamps are the study clock) and monitoring runs at chosen logical
times. Two dyads consent on Monday 2026-01-05, so with ``n_warm = 5`` every
update is a cohort warm-up update that completes without fitting (F-U3).
The cases: a normal week; no ``/update`` by 03:30; the update cannot start
(503); the update fails; stuck and resent; stuck and not resent; not ready
by 05:45; week-closing uploads missing; a week before learning starts; a
second ``/update`` in the same week (its C5 match is tested with a learning
trial in ``test_monitoring_trial.py``).
"""

from __future__ import annotations

from app.models import ModelUpdateRequests, PolicyVersion, UpdateWeek
from app.monitoring import run_monitoring
from tests.test_monitoring_support import (  # noqa: F401 - fixtures
    MON1,
    D,
    alarms,
    at,
    mon_app,
    mon_client,
    post_register,
    post_update,
    post_upload,
)

W = D(2026, 1, 12)
W_NEXT = D(2026, 1, 19)


def _dyads(client, closing=("d1", "d2"), week=W):
    for gid in ("d1", "d2"):
        if post_register(client, gid, MON1).status_code != 201:
            raise AssertionError(gid)
    _closing(client, closing, week)


def _closing(client, gids, week):
    for gid in gids:
        r = post_upload(client, gid, MON1, at(week, 2, 0), current_game_on="miss")
        assert r.status_code == 201, r.get_json()


def _week(monday=W) -> UpdateWeek | None:
    return UpdateWeek.query.filter_by(study_week_monday=monday).first()


def _items() -> list[str]:
    return [a.item_id for a in alarms()]


def _no_process(monkeypatch):
    """The update thread never runs (killed / hung): the request stays ``processing``."""
    import app.routes.update as update_route

    monkeypatch.setattr(update_route, "process_update_request", lambda app, rid, algo=None: None)


def test_normal_week(mon_client):
    _dyads(mon_client)
    rid = post_update(mon_client, at(W, 3, 0)).get_json()["rid"]
    assert ModelUpdateRequests.query.filter_by(rid=rid).one().status == "completed"
    run_monitoring(now=at(W, 6, 0), since=at(W, 0, 0))
    assert not {"C11", "C7", "B11", "C1"} & set(_items())
    week = _week()
    assert week.status == "ready" and week.rids == [rid] and week.first_call_at == at(W, 3, 0)
    assert week.missing_closing_uploads == [] and week.c7_raised is False and week.ready_at is not None


def test_no_update_by_0330_then_not_ready(mon_client):
    _dyads(mon_client)
    run_monitoring(now=at(W, 3, 31), since=at(W, 0, 0))
    c11 = alarms("C11")
    assert len(c11) == 1 and (c11[0].severity, c11[0].owner) == ("red", "host")
    assert c11[0].raised_at == at(W, 3, 30)
    assert _week().status == "waiting"
    run_monitoring(now=at(W, 5, 46), since=at(W, 3, 31))
    c7 = alarms("C7")
    assert len(c7) == 1 and c7[0].detail["reason"] == "never_arrived" and c7[0].raised_at == at(W, 5, 45)
    assert _week().c7_raised is True
    feed = mon_client.get("/api/v1/monitor/alarms").get_json()  # the 05:50 pull carries C11 and C7
    assert {"C11", "C7"} <= {a["item_id"] for a in feed["alarms"]}


def test_update_cannot_start(mon_client, monkeypatch):
    _dyads(mon_client)
    import app.routes.update as update_route
    from sqlalchemy.exc import OperationalError

    def broken(*args, **kwargs):
        raise OperationalError("INSERT", {}, Exception("db down"))

    monkeypatch.setattr(update_route, "ModelUpdateRequests", broken)
    r = post_update(mon_client, at(W, 3, 0))
    assert r.status_code == 503
    monkeypatch.undo()
    enqueue = alarms("UP-503-ENQUEUE")
    assert len(enqueue) == 1 and (enqueue[0].severity, enqueue[0].owner) == ("red", "rl")
    assert ModelUpdateRequests.query.count() == 0  # nothing was recorded
    run_monitoring(now=at(W, 5, 46), since=at(W, 0, 0))
    assert alarms("C11") and alarms("C7")[0].detail["reason"] == "never_arrived"


def test_update_fails(mon_client, monkeypatch):
    _dyads(mon_client)
    from app import learner

    def crash(*args, **kwargs):
        raise RuntimeError("fit crashed")

    monkeypatch.setattr(learner, "run_weekly_update", crash)
    rid = post_update(mon_client, at(W, 3, 0)).get_json()["rid"]
    assert ModelUpdateRequests.query.filter_by(rid=rid).one().status == "failed"
    c1 = alarms("C1")
    assert len(c1) == 1 and c1[0].detail["kind"] == "update_failed" and c1[0].detail["agent"] == "all"
    assert c1[0].source == "after_response" and c1[0].fallback_used == "F-U2"
    assert _week().status == "failed"
    run_monitoring(now=at(W, 5, 46), since=at(W, 0, 0))
    assert alarms("C7")[0].detail["reason"] == "failed"
    assert len(alarms("C1")) == 1  # the runner found the same C1


def test_stuck_update_resent_by_the_host(mon_client, monkeypatch):
    _dyads(mon_client)
    _no_process(monkeypatch)
    first = post_update(mon_client, at(W, 3, 0)).get_json()["rid"]
    monkeypatch.undo()
    result = run_monitoring(now=at(W, 3, 35), since=at(W, 0, 0))
    assert result["timeouts"] == [first]
    stuck = ModelUpdateRequests.query.filter_by(rid=first).one()
    assert (stuck.status, stuck.error_message) == ("failed", "timeout")
    assert _week().status == "stuck"
    second = post_update(mon_client, at(W, 3, 40)).get_json()["rid"]
    assert _week().status == "ready" and _week().rids == [first, second]
    run_monitoring(now=at(W, 5, 46), since=at(W, 3, 35))
    assert not {"C7", "C11", "C1"} & set(_items())


def test_stuck_update_not_resent(mon_client, monkeypatch):
    _dyads(mon_client)
    _no_process(monkeypatch)
    post_update(mon_client, at(W, 3, 0))
    run_monitoring(now=at(W, 3, 35), since=at(W, 0, 0))
    run_monitoring(now=at(W, 5, 46), since=at(W, 3, 35))
    c7 = alarms("C7")
    assert len(c7) == 1 and c7[0].detail["reason"] == "timeout"
    assert _week().status == "stuck" and _week().c7_raised is True
    assert alarms("C1") == []  # a timeout is reported by C7, not C1


def test_not_ready_by_0545(mon_client, monkeypatch):
    _dyads(mon_client)
    _no_process(monkeypatch)
    post_update(mon_client, at(W, 5, 30))
    run_monitoring(now=at(W, 5, 46), since=at(W, 0, 0))
    assert alarms("C11")  # nothing by 03:30
    c7 = alarms("C7")
    assert len(c7) == 1 and c7[0].detail["reason"] == "running"


def test_update_arriving_after_the_deadline_is_not_activated(mon_client):
    _dyads(mon_client)
    rid = post_update(mon_client, at(W, 5, 50)).get_json()["rid"]
    assert ModelUpdateRequests.query.filter_by(rid=rid).one().status == "completed"
    run_monitoring(now=at(W, 6, 0), since=at(W, 0, 0))
    assert alarms("C7")[0].detail["reason"] == "never_arrived"
    assert _week().status == "failed" and _week().ready_at is None


def test_week_closing_uploads_missing(mon_client):
    _dyads(mon_client, closing=("d1",))
    post_update(mon_client, at(W, 3, 0))
    b11 = alarms("B11")
    assert len(b11) == 1 and b11[0].severity == "yellow" and b11[0].owner == "host"
    assert b11[0].detail == {"study_week_monday": "2026-01-12", "group_ids": ["d2"],
                             "update_rid": b11[0].rid}
    assert b11[0].raised_at == at(W, 3, 0) and b11[0].source == "after_response"
    assert _week().missing_closing_uploads == ["d2"]
    _closing(mon_client, ("d1",), W_NEXT)
    post_update(mon_client, at(W_NEXT, 3, 0))
    red = [a for a in alarms("B11") if a.severity == "red"]
    assert len(red) == 1 and red[0].detail["group_ids"] == ["d2"]
    run_monitoring(now=at(W_NEXT, 6, 0), since=at(W, 0, 0))
    assert len(alarms("B11")) == 2  # the runner finds the same two findings


def test_week_before_learning_starts(mon_client):
    for gid in ("d1", "d2"):
        assert post_register(mon_client, gid, MON1).status_code == 201
    run_monitoring(now=at(MON1, 6, 0), since=at(MON1, 0, 0))  # nobody was active the week before: no update due
    assert not {"C11", "C7"} & set(_items())
    _closing(mon_client, ("d1", "d2"), W)
    rid = post_update(mon_client, at(W, 3, 0)).get_json()["rid"]
    run_monitoring(now=at(W, 6, 0), since=at(MON1, 6, 0))
    assert not {"C11", "C7", "C1", "B5", "C3"} & set(_items())
    assert PolicyVersion.query.count() == 0 and _week().status == "ready" and _week().rids == [rid]


def test_second_update_in_the_same_week(mon_client):
    _dyads(mon_client, closing=("d1",))
    first = post_update(mon_client, at(W, 3, 0)).get_json()["rid"]
    second = post_update(mon_client, at(W, 4, 0)).get_json()["rid"]
    week = _week()
    assert week.rids == [first, second] and week.status == "ready"
    assert len(alarms("B11")) == 1  # B11 is decided at the first call only
    run_monitoring(now=at(D(2026, 1, 13), 2, 30), since=at(W, 0, 0))
    assert alarms("C5") == [] and alarms("C7") == []
