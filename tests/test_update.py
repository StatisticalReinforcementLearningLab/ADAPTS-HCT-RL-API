"""
``POST /api/v1/update`` (API-Spec §2.4, §5.6, §7.1; Monitoring-Spec §6.5, §7
F-U4; contract D13, D14, D21): 202 / 400 / 503 / 500 answers, the
``model_update_requests`` columns, the monitoring hooks, the learner call, the
failure paths, the background thread, backups and the reproducibility
snapshot. ``app.learner.run_weekly_update`` is monkeypatched.
"""

from __future__ import annotations

import datetime
import json
import os
import time
import zipfile

import pytest
from sqlalchemy.exc import OperationalError

from app import learner, monitoring
from app.extensions import db
from app.models import (
    ModelParameters,
    ModelUpdateRequests,
    StudyData,
    UpdateReproducibilitySnapshot,
)
from app.routes import update as update_module
from tests.conftest import ENVELOPE_KEYS, assert_envelope, post_action, register_group, upload

UPDATE_TS = "2026-01-12T03:00:00"


@pytest.fixture
def hooks(monkeypatch):
    """Record the monitoring hooks /update calls (contract §3.1)."""
    events = []
    monkeypatch.setattr(monitoring, "on_update_received", lambda rid: events.append(("received", rid)))
    monkeypatch.setattr(monitoring, "on_update_finished", lambda rid, summary: events.append(("finished", rid, summary)))
    return events


@pytest.fixture
def fit(monkeypatch):
    """A stand-in weekly update recording its arguments."""
    calls = []

    def run_weekly_update(update_rid, request_timestamp, started_wall):
        calls.append((update_rid, request_timestamp, started_wall))
        return {
            "study_week_monday": datetime.date(2026, 1, 12),
            "agents": {"dyad_game": {"status": "published"}},
            "status": "completed",
        }

    monkeypatch.setattr(learner, "run_weekly_update", run_weekly_update)
    return calls


def _row(rid) -> ModelUpdateRequests:
    return ModelUpdateRequests.query.filter_by(rid=rid).one()


# ------------------------------------------------------------------ 202


def test_update_is_accepted_and_completes(client, hooks, fit):
    response = client.post("/api/v1/update", json={"timestamp": UPDATE_TS})
    out = assert_envelope(response, 202, "Update Accepted", "Model update started.")
    assert set(out) == set(ENVELOPE_KEYS), "/update answers with the envelope only"
    rid = out["rid"]

    row = _row(rid)
    assert row.status == "completed"
    assert row.error_message is None
    assert row.request_timestamp == datetime.datetime(2026, 1, 12, 3, 0)
    assert row.study_week_monday == datetime.date(2026, 1, 12)
    assert row.started_at is not None and row.completed_at is not None
    assert row.started_at <= row.completed_at
    # D13: published_at_study = request timestamp + wall-clock duration of the run.
    assert row.request_timestamp <= row.published_at_study <= row.request_timestamp + datetime.timedelta(minutes=1)

    assert fit == [(rid, datetime.datetime(2026, 1, 12, 3, 0), row.started_at)]
    assert [e[0] for e in hooks] == ["received", "finished"]
    assert hooks[0][1] == rid and hooks[1][1] == rid
    summary = hooks[1][2]
    assert summary["status"] == "completed"
    assert summary["agents"] == {"dyad_game": {"status": "published"}}
    assert summary["n_rewards_derived"] == 0


def test_study_week_follows_the_detroit_local_timestamp(client, hooks, fit):
    # 2026-01-12T04:30Z is Sunday 23:30 in Detroit: the update belongs to week 2026-01-05.
    out = assert_envelope(client.post("/api/v1/update", json={"timestamp": "2026-01-12T04:30:00Z"}), 202)
    row = _row(out["rid"])
    assert row.request_timestamp == datetime.datetime(2026, 1, 11, 23, 30)
    assert row.study_week_monday == datetime.date(2026, 1, 5)


def test_learner_summary_published_time_is_used(client, hooks, monkeypatch):
    published = datetime.datetime(2026, 1, 12, 3, 4, 5)
    monkeypatch.setattr(learner, "run_weekly_update",
                        lambda rid, ts, wall: {"status": "completed", "agents": {}, "published_at_study": published})
    out = assert_envelope(client.post("/api/v1/update", json={"timestamp": UPDATE_TS}), 202)
    assert _row(out["rid"]).published_at_study == published


def test_rewards_are_derived_before_the_learner_runs(client, hooks, monkeypatch):
    register_group(client, "dyad_a")
    upload(client, "dyad_a", "2026-01-06T07:30:00")
    assert_envelope(post_action(client, "dyad_a", "cp_message", 0, "2026-01-06T07:35:00"), 201)
    upload(client, "dyad_a", "2026-01-07T07:30:00", daily_diary_completed=True, cp_diary_mood=4)
    seen = []

    def run_weekly_update(rid, ts, wall):
        seen.append([(r.decision_type, r.reward) for r in StudyData.query.all()])
        return {"status": "completed", "agents": {}}

    monkeypatch.setattr(learner, "run_weekly_update", run_weekly_update)
    assert_envelope(client.post("/api/v1/update", json={"timestamp": UPDATE_TS}), 202)
    assert seen == [[("cp_message", pytest.approx(0.8))]]
    assert hooks[-1][2]["n_rewards_derived"] == 1


# ---------------------------------------------------------- fit failures


def test_learner_exception_marks_the_request_failed(client, hooks, monkeypatch):
    def boom(rid, ts, wall):
        raise RuntimeError("EB objective diverged")

    monkeypatch.setattr(learner, "run_weekly_update", boom)
    out = assert_envelope(client.post("/api/v1/update", json={"timestamp": UPDATE_TS}), 202,
                          "Update Accepted", "Model update started.")
    row = _row(out["rid"])
    assert row.status == "failed"
    assert "EB objective diverged" in row.error_message
    assert row.completed_at is not None
    assert row.published_at_study is None
    finished = hooks[-1]
    assert finished[0] == "finished" and finished[2]["status"] == "failed"
    assert "EB objective diverged" in finished[2]["error"]


def test_failed_summary_marks_the_request_failed(client, hooks, monkeypatch):
    monkeypatch.setattr(learner, "run_weekly_update",
                        lambda rid, ts, wall: {"status": "failed", "error": "C3 rejected aya_message", "agents": {}})
    out = assert_envelope(client.post("/api/v1/update", json={"timestamp": UPDATE_TS}), 202)
    row = _row(out["rid"])
    assert row.status == "failed"
    assert row.error_message == "C3 rejected aya_message"


def test_monitoring_hook_failures_do_not_change_the_outcome(client, fit, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("monitoring bug")

    monkeypatch.setattr(monitoring, "on_update_received", broken)
    monkeypatch.setattr(monitoring, "on_update_finished", broken)
    out = assert_envelope(client.post("/api/v1/update", json={"timestamp": UPDATE_TS}), 202)
    assert _row(out["rid"]).status == "completed"


def test_fault_update_hang_sleeps_after_the_row_is_written(client, app, hooks, fit, monkeypatch):
    slept = []

    def fake_sleep(seconds):
        slept.append((seconds, ModelUpdateRequests.query.count()))

    monkeypatch.setattr(update_module.time, "sleep", fake_sleep)
    app.config["FAULT_UPDATE_HANG_SECONDS"] = 0.25
    assert_envelope(client.post("/api/v1/update", json={"timestamp": UPDATE_TS}), 202)
    assert slept == [(0.25, 1)]

    slept.clear()
    app.config["FAULT_INJECTION_ENABLED"] = False
    assert_envelope(client.post("/api/v1/update", json={"timestamp": "2026-01-12T03:05:00"}), 202)
    assert slept == []


# ------------------------------------------------------------------ 400


@pytest.mark.parametrize(
    "body, message",
    [
        ({}, "timestamp is required."),
        ({"time": UPDATE_TS}, "timestamp is required."),
        ({"timestamp": 1736668800}, "timestamp must be a string or datetime object."),
        ({"timestamp": "Monday 3am"}, "timestamp must be an ISO-8601 timestamp; got 'Monday 3am'."),
    ],
)
def test_bad_requests_are_400(client, hooks, body, message):
    out = assert_envelope(client.post("/api/v1/update", json=body), 400, "Invalid Parameter", message)
    assert set(out) == set(ENVELOPE_KEYS)
    assert ModelUpdateRequests.query.count() == 0
    assert hooks == []


def test_empty_body_is_400(client):
    response = client.post("/api/v1/update", data=b"", content_type="application/json")
    assert_envelope(response, 400, "Invalid Parameter", "timestamp is required.")


# ------------------------------------------------------------- 503 / 500


def test_row_write_failure_is_503_and_records_nothing(client, hooks, fit, monkeypatch):
    def fail_commit():
        raise OperationalError("INSERT INTO model_update_requests", {}, Exception("database is locked"))

    monkeypatch.setattr(db.session, "commit", fail_commit)
    out = assert_envelope(client.post("/api/v1/update", json={"timestamp": UPDATE_TS}), 503,
                          "Service Unavailable", "Could not start the update; retry.")
    assert set(out) == set(ENVELOPE_KEYS)
    monkeypatch.undo()
    assert ModelUpdateRequests.query.count() == 0
    assert hooks == [] and fit == []


def test_thread_start_failure_is_503(make_app, monkeypatch, hooks, fit):
    app = make_app(UPDATE_RUN_INLINE=False)
    client = app.test_client()

    class NoThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(update_module, "Thread", NoThread)
    out = assert_envelope(client.post("/api/v1/update", json={"timestamp": UPDATE_TS}), 503,
                          "Service Unavailable", "Could not start the update; retry.")
    row = _row(out["rid"])
    assert row.status == "failed"
    assert row.error_message == "could not start the update thread"


def test_unexpected_failure_is_500(client, monkeypatch):
    def boom(value):
        raise RuntimeError("bug")

    monkeypatch.setattr(update_module, "parse_timestamp", boom)
    out = assert_envelope(client.post("/api/v1/update", json={"timestamp": UPDATE_TS}), 500,
                          "Internal Error", "Internal server error.")
    assert set(out) == set(ENVELOPE_KEYS)
    assert ModelUpdateRequests.query.count() == 0


# ------------------------------------------------- thread, backup, snapshot


def test_background_thread_with_a_file_database(make_app, tmp_path, monkeypatch, hooks):
    app = make_app(
        UPDATE_RUN_INLINE=False,
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{tmp_path / 'thread.db'}",
    )
    app.config["FAULT_UPDATE_HANG_SECONDS"] = 0.3
    monkeypatch.setattr(learner, "run_weekly_update", lambda rid, ts, wall: {"status": "completed", "agents": {}})
    client = app.test_client()
    out = assert_envelope(client.post("/api/v1/update", json={"timestamp": UPDATE_TS}), 202)
    rid = out["rid"]
    db.session.remove()
    assert ModelUpdateRequests.query.filter_by(rid=rid).one().status == "processing"
    deadline = time.time() + 20
    status = "processing"
    while time.time() < deadline:
        db.session.remove()
        status = ModelUpdateRequests.query.filter_by(rid=rid).one().status
        if status != "processing":
            break
        time.sleep(0.05)
    assert status == "completed"
    assert [e[0] for e in hooks] == ["received", "finished"]


def test_backup_zips_every_table_but_the_quarantine(make_app, tmp_path, fit):
    root = tmp_path / "backups"
    app = make_app(BACKUP_DATABASE=True, BACKUP_ROOT=str(root))
    client = app.test_client()
    register_group(client, "dyad_a")
    client.post("/api/v1/upload_data", json={"group_id": "dyad_a"})  # 400 -> quarantined
    assert_envelope(client.post("/api/v1/update", json={"timestamp": UPDATE_TS}), 202)
    zips = list(root.glob("*.zip"))
    assert len(zips) == 1
    names = set(zipfile.ZipFile(zips[0]).namelist())
    assert {"groups.csv", "group_history.csv", "actions.csv", "data_uploads.csv",
            "model_update_requests.csv", "policy_versions.csv"} <= names
    assert "upload_quarantine.csv" not in names
    assert not any(p.is_dir() for p in root.iterdir())


def test_repro_snapshot_includes_group_history(make_app, tmp_path, fit):
    root = tmp_path / "repro"
    app = make_app(SAVE_UPDATE_REPRO_SNAPSHOTS=True, REPRO_SNAPSHOT_ROOT=str(root))
    client = app.test_client()
    register_group(client, "dyad_a")
    register_group(client, "dyad_a", start="2026-01-06")
    upload(client, "dyad_a", "2026-01-07T07:25:00")
    assert_envelope(post_action(client, "dyad_a", "aya_message", 0, "2026-01-07T07:30:00"), 201)
    out = assert_envelope(client.post("/api/v1/update", json={"timestamp": UPDATE_TS}), 202)
    snap = UpdateReproducibilitySnapshot.query.filter_by(rid=out["rid"]).one()
    assert os.path.isdir(snap.snapshot_dir)
    history = json.loads((root / out["rid"] / "group_history.json").read_text())
    assert [h["consent_start_date"] for h in history] == ["2026-01-05", "2026-01-06"]
    actions = json.loads((root / out["rid"] / "actions.json").read_text())
    assert actions[0]["upload_id"] is not None and "hyperparams_hash" in actions[0]
    meta = json.loads((root / out["rid"] / "metadata.json").read_text())
    assert meta["group_history_count"] == 2


def test_legacy_learner_update_path(make_app, hooks):
    app = make_app(RL_ALGORITHM="always_send")
    client = app.test_client()
    before = ModelParameters.query.filter(ModelParameters.snapshot_type.is_(None)).count()
    out = assert_envelope(client.post("/api/v1/update", json={"timestamp": UPDATE_TS}), 202)
    assert _row(out["rid"]).status == "completed"
    assert ModelParameters.query.filter(ModelParameters.snapshot_type.is_(None)).count() == before + 1
