"""
WS-M tests of the records and system checks (Monitoring-Spec §5.D: D1, D3;
§5.E: E1, E5, E6, E7; §5.G: G0), the monitoring program (catch-up at each
check's own scheduled time, ``monitor_runs``, rerun idempotency) and the
``flask monitor`` commands (§8.3).
"""

from __future__ import annotations

import datetime
import json

import pytest

from app.extensions import db
from app.models import ApiErrorEvent, MonitorAlarm, MonitorPull, MonitorRun
from app.monitoring import raise_alarm, run_monitoring
from app.monitoring.checks import REGISTRY, records, system
from tests.test_monitoring_support import (  # noqa: F401 - fixtures
    AYA,
    MON1,
    D,
    add_action,
    add_dyad,
    add_upload,
    alarms,
    app_factory,
    at,
    ctx,
    keys,
    mon_app,
)

MON2 = D(2026, 1, 12)


def _new(result):
    return [a["item_id"] for a in result["alarms"]]


# ---------------------------------------------------------------------- D1


def test_d1_fires_without_manifest_then_quiet_after_export_then_row_change(app_factory, tmp_path):
    from app import export_all_tables

    app = app_factory(MONITOR_EXPORT_DIR=str(tmp_path / "exports"))
    res = run_monitoring(now=at(MON2, 7, 0), item="D1")
    assert _new(res) == ["D1"] and alarms("D1")[0].detail["problems"][0]["kind"] == "manifest_missing"

    for k in range(10):
        add_dyad(f"g{k}", MON1)
    export_all_tables(str(tmp_path / "exports"))
    res = run_monitoring(now=at(D(2026, 1, 13), 7, 0), item="D1")
    assert _new(res) == []

    for k in range(10, 14):  # groups: 10 -> 14 rows (+40%)
        add_dyad(f"g{k}", MON1)
    export_all_tables(str(tmp_path / "exports"))
    res = run_monitoring(now=at(D(2026, 1, 14), 7, 0), item="D1")
    assert _new(res) == ["D1"]
    problem = alarms("D1")[-1].detail["problems"][0]
    assert problem["kind"] == "row_count_change"
    assert {"file": "groups.csv", "rows": 14, "rows_day_before": 10, "change": 0.4} in problem["files"]
    assert app.config["MONITOR_EXPORT_DIR"].endswith("exports")


def test_d1_missing_table_in_manifest(app_factory, tmp_path):
    app_factory(MONITOR_EXPORT_DIR=str(tmp_path))
    reading = {"present": True, "generated_at": "2026-01-12T06:00:00-05:00", "read_at": "2026-01-12T07:00:00",
               "files": {"groups.csv": 3}, "path": str(tmp_path / "MANIFEST.json")}
    problems = records.d1_problems(reading, None)
    assert problems[0]["kind"] == "tables_missing" and "actions.csv" in problems[0]["files"]


# ---------------------------------------------------------------------- D3


def _event(code, status, title, message, when):
    db.session.add(ApiErrorEvent(event_id=f"e-{code}-{when.isoformat()}", code=code, endpoint="/api/v1/action",
                                 http_status=status, title=title, message=message, rid="r", occurred_at=when))
    db.session.commit()


def test_d3_lists_unmapped_and_undocumented_answers_of_last_week(mon_app):
    _event("XX-418-UNMAPPED", 418, "Teapot", "short", at(D(2026, 1, 7), 9))
    _event("AC-400-PARAM", 400, "Invalid Parameter", "timestamp must be an ISO-8601 timestamp; got 'x'.",
           at(D(2026, 1, 8), 9))
    _event("XX-404-ROUTE", 404, "Not Found", "Unknown route.", at(D(2026, 1, 8), 9))
    found = records.check_d3(ctx(at(MON2, 7, 0)))
    assert keys(found) == ["D3:2026-01-05"]
    assert [r["code"] for r in found[0].detail["unmapped"]] == ["XX-418-UNMAPPED"]
    assert [r["code"] for r in found[0].detail["undocumented_wording"]] == ["AC-400-PARAM"]


def test_d3_quiet_when_every_answer_is_documented(mon_app):
    _event("XX-404-ROUTE", 404, "Not Found", "Unknown route.", at(D(2026, 1, 8), 9))
    _event("XX-418-UNMAPPED", 418, "Teapot", "short", at(MON2, 9))  # this week: next Monday's D3
    assert records.check_d3(ctx(at(MON2, 7, 0))) == []


# ---------------------------------------------------------------------- E1


def test_e1_raised_when_a_reading_first_crosses_its_limit(mon_app, monkeypatch):
    readings = iter([{"memory_pct": 50.0, "disk_pct": 10.0}, {"memory_pct": 90.0, "disk_pct": 10.0},
                     {"memory_pct": 91.0, "disk_pct": 10.0}, {"memory_pct": 40.0, "disk_pct": 95.0},
                     {"memory_pct": 88.0, "disk_pct": 95.0}])
    monkeypatch.setattr(system, "read_server_resources", lambda: next(readings))
    seen = []
    for k in range(5):
        res = run_monitoring(now=at(MON2, 10, 5 * k))
        seen.append(sorted(a["item_id"] for a in res["alarms"] if a["item_id"] == "E1"))
    assert seen == [[], ["E1"], [], ["E1"], ["E1"]]
    metrics = [a.detail["metric"] for a in alarms("E1")]
    assert metrics == ["memory_pct", "disk_pct", "memory_pct"]
    assert all(a.severity == "red" for a in alarms("E1"))
    runs = MonitorRun.query.filter_by(kind="regular").order_by(MonitorRun.id).all()
    assert runs[1].detail["e1"]["memory_pct"] == 90.0


def test_e1_reads_real_resources(mon_app):
    reading = system.read_server_resources()
    assert set(reading) >= {"memory_pct", "disk_pct", "db_conn_pct", "db_dialect"}
    assert reading["db_dialect"] == "sqlite" and reading["db_conn_pct"] is None
    assert reading["disk_pct"] is None or 0.0 <= reading["disk_pct"] <= 100.0


# ---------------------------------------------------------------------- E5


def test_e5_fires_after_a_gap_and_the_missed_checks_are_caught_up(mon_app):
    add_dyad("d1", MON1)
    run_monitoring(now=at(MON2, 3, 0))
    res = run_monitoring(now=at(MON2, 6, 0))  # down from 03:00 to 06:00: C11 (03:30) and C7 (05:45) caught up
    new = _new(res)
    assert "E5" in new and "C11" in new and "C7" in new
    e5 = alarms("E5")[0]
    assert e5.detail["gap_minutes"] == 180.0 and e5.detail["alarms_raised_in_catch_up"] >= 2
    c11 = alarms("C11")[0]
    assert c11.raised_at == at(MON2, 3, 30)  # evaluated at its own scheduled time


def test_e5_quiet_on_regular_spacing(mon_app):
    run_monitoring(now=at(MON2, 10, 0))
    res = run_monitoring(now=at(MON2, 10, 5))
    assert "E5" not in _new(res)
    res = run_monitoring(now=at(MON2, 10, 20), since=at(MON2, 10, 5))  # an explicit window is never E5
    assert "E5" not in _new(res)


# ---------------------------------------------------------------------- E6


def _pull(rid, ts, status=200):
    db.session.add(MonitorPull(rid=rid, status=status, window_start=ts - datetime.timedelta(hours=24),
                               returned_count=0 if status == 200 else None))
    db.session.commit()


def test_e6_raised_once_when_the_host_stops_pulling(mon_app):
    run_monitoring(now=at(MON2, 5, 0))
    _pull("p1", at(MON2, 5, 50))
    _pull("p-failed", at(MON2, 17, 50), status=503)
    res = run_monitoring(now=at(D(2026, 1, 13), 6, 0))
    assert "E6" in _new(res)
    e6 = alarms("E6")
    assert len(e6) == 1 and e6[0].raised_at == at(D(2026, 1, 13), 5, 50) and e6[0].owner == "host"
    res = run_monitoring(now=at(D(2026, 1, 13), 6, 5))
    assert "E6" not in _new(res)


def test_e6_quiet_with_regular_pulls(mon_app):
    run_monitoring(now=at(MON2, 5, 0))
    for k, hh in enumerate((5, 17)):
        _pull(f"p{k}", at(MON2, hh, 50))
    _pull("p2", at(D(2026, 1, 13), 5, 50))
    res = run_monitoring(now=at(D(2026, 1, 13), 6, 0), since=at(MON2, 5, 0))
    assert "E6" not in _new(res)


# ---------------------------------------------------------------------- E7


def test_e7_when_a_check_crashes_and_the_run_goes_on(mon_app, monkeypatch):
    import dataclasses

    from app.monitoring import checks

    def broken(ctx):
        raise ZeroDivisionError("check bug with private text")

    monkeypatch.setitem(checks.REGISTRY, "A13", dataclasses.replace(checks.REGISTRY["A13"], fn=broken))
    add_dyad("d1", MON1)
    res = run_monitoring(now=at(MON2, 3, 31), since=at(MON2, 3, 0))
    assert res["status"] == "error" and res["errors"][0]["item"] == "A13"
    assert "C11" in _new(res)  # the other checks still ran
    e7 = alarms("E7")
    assert len(e7) == 1 and "private text" not in json.dumps(e7[0].detail) and "private text" not in e7[0].message
    assert db.session.get(MonitorRun, res["run_id"]).status == "error"


def test_e7_quiet_when_checks_succeed(mon_app):
    res = run_monitoring(now=at(MON2, 10, 0))
    assert res["errors"] == [] and alarms("E7") == []


# ---------------------------------------------------------------------- G0


def test_g0_daily_status_content(mon_app):
    add_dyad("d1", MON1)
    add_action("d1", AYA, 0, at(D(2026, 1, 11), 7, 30), action=1, reason="no_policy")
    raise_alarm("C11", message="m", source="scheduled", raised_at=at(MON2, 3, 30))
    res = run_monitoring(now=at(MON2, 7, 0), since=at(MON2, 6, 0))
    assert "G0" in _new(res)
    g0 = alarms("G0")[0]
    assert g0.severity == "green" and g0.dedup_key == "G0:2026-01-12"
    d = g0.detail
    assert set(d) >= {"monitoring_runs_24h", "decisions", "update", "fallbacks_24h", "error_codes_24h",
                      "eb_fallback_rate", "alarms_24h"}
    assert d["alarms_24h"]["red"] == 1
    assert d["decisions"][AYA]["made"] == 1 and d["decisions"][AYA]["expected"] == 2
    assert d["fallbacks_24h"]["F-A3"] == {AYA: 1}
    res = run_monitoring(now=at(MON2, 7, 1), since=at(MON2, 6, 0))
    assert "G0" not in _new(res)


def test_g0_not_raised_outside_its_time(mon_app):
    res = run_monitoring(now=at(MON2, 6, 59), since=at(MON2, 6, 0))
    assert "G0" not in _new(res)


# ------------------------------------------------------------ the program


def test_runs_are_recorded_with_kind_window_and_logical_now(mon_app):
    a = run_monitoring(now=at(MON2, 10, 0))
    b = run_monitoring(now=at(MON2, 10, 5))
    c = run_monitoring(now=at(MON2, 9, 0), since=at(MON2, 8, 0))
    d = run_monitoring(now=at(MON2, 10, 6), item="A13")
    runs = {r.id: r for r in MonitorRun.query.all()}
    assert runs[a["run_id"]].kind == "regular" and runs[a["run_id"]].window_start == at(MON2, 9, 55)
    assert runs[b["run_id"]].window_start == at(MON2, 10, 0) and runs[b["run_id"]].logical_now == at(MON2, 10, 5)
    assert runs[c["run_id"]].kind == "rerun" and runs[c["run_id"]].window_start == at(MON2, 8, 0)
    assert runs[d["run_id"]].kind == "check" and runs[d["run_id"]].item_id == "A13"
    assert all(r.finished_at is not None and r.status == "ok" for r in runs.values())
    with pytest.raises(ValueError):
        run_monitoring(now=at(MON2, 11, 0), item="NOPE")


def test_rerun_of_the_same_window_raises_nothing_new(mon_app):
    add_dyad("d1", MON1)
    add_action("d1", AYA, 0, at(D(2026, 1, 9), 7, 30), pi=0.4)  # A13 (warm-up pi != 0.5)
    add_upload("d1", at(MON2, 2, 0), aya_messages_delivered_week=14, cp_messages_delivered_week=0)
    first = run_monitoring(now=at(D(2026, 1, 14), 8, 0), since=at(MON1))
    assert first["n_alarms"] > 0
    n = MonitorAlarm.query.count()
    second = run_monitoring(now=at(D(2026, 1, 14), 8, 0), since=at(MON1))
    assert second["n_alarms"] == 0 and MonitorAlarm.query.count() == n
    assert sorted(first["findings"]) == sorted(second["findings"])
    assert {"A13", "C11", "C7", "A2", "G0"} <= {a.item_id for a in alarms()}


def test_loop_runs_and_sleeps(mon_app):
    from app.monitoring.runner import loop

    sleeps = []
    assert loop(max_runs=2, sleep=sleeps.append) == 2
    assert len(sleeps) == 1 and 0.0 <= sleeps[0] <= 300.0
    assert MonitorRun.query.filter_by(kind="regular").count() == 2


# -------------------------------------------------------------------- CLI


def test_cli_run_status_show_note_report(mon_app):
    runner = mon_app.test_cli_runner()
    add_dyad("d1", MON1)
    out = runner.invoke(args=["monitor", "run", "--since", "2026-01-12T00:00", "--until", "2026-01-12T06:00"])
    assert out.exit_code == 0, out.output
    assert "new red C11" in out.output and "new yellow C7" in out.output
    c11 = alarms("C11")[0]

    out = runner.invoke(args=["monitor", "show", c11.event_id])
    assert out.exit_code == 0 and json.loads(out.output)["item_id"] == "C11"
    out = runner.invoke(args=["monitor", "note", c11.event_id, "host cron restarted", "--by", "ziping"])
    assert out.exit_code == 0 and "Note added" in out.output
    db.session.expire_all()
    assert alarms("C11")[0].note == "host cron restarted" and alarms("C11")[0].note_by == "ziping"

    out = runner.invoke(args=["monitor", "status"])
    assert out.exit_code == 0 and "This week's update" in out.output and "Last monitoring run" in out.output

    out = runner.invoke(args=["monitor", "report", "--since", "2026-01-01", "--until", "2026-01-31"])
    assert out.exit_code == 0, out.output
    assert "RED alarms" in out.output and "C11" in out.output and "Fallbacks" in out.output
    out = runner.invoke(args=["monitor", "report", "--since", "2026-01-01", "--format", "json"])
    report = json.loads(out.output)
    assert report["by_severity"]["red"][0]["item_id"] == "C11"
    assert report["red_alarms_without_note"] == []

    out = runner.invoke(args=["monitor", "run", "--item", "C7"])
    assert out.exit_code == 0
    out = runner.invoke(args=["monitor", "loop", "--max-runs", "1"])
    assert out.exit_code == 0 and "after 1 runs" in out.output
    out = runner.invoke(args=["monitor", "run", "--until", "2026-01-12"])
    assert out.exit_code != 0
    out = runner.invoke(args=["monitor", "show", "no-such-event"])
    assert out.exit_code != 0


def test_registry_covers_every_scheduled_check():
    from app.monitoring import catalog

    scheduled = set(catalog.CHECKS) - {"E5", "E7"}
    assert scheduled <= set(REGISTRY)
    assert REGISTRY["G0"].order == max(item.order for item in REGISTRY.values())
