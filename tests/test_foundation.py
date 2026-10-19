"""
Foundation tests (WS-F): schema, migration, study clock, study calendar,
hyperparameters, Monte Carlo bank, framework envelopes and start-up guards.

References: implementation contract §2 (schema), D2-D5 (calendar and clock),
D10 (MC bank), D16 (hyperparameters), D19 (framework answers);
Hyperparameters.md §2-§5; main_new.tex Table 1, eq. (cohort-index);
Monitoring-Spec §6.7, §9, §10.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import textwrap
from zoneinfo import ZoneInfo

import numpy as np
import pytest

import config
from app import create_app, export_all_tables
from app import monitoring as monitoring_pkg
from app.clock import (
    format_detroit,
    server_now,
    study_now,
    to_study_local,
    to_utc,
)
from app.deterministic_sampler import DeterministicSampleStream
from app.extensions import db
from app.hyperparams import (
    DEFAULT_HYPERPARAMS_FILE,
    HyperparameterError,
    amend,
    canonical_json,
    check_startup_against_frozen,
    default_hyperparams,
    freeze_if_needed,
    in_force_hash,
    in_force_version,
    latest_frozen,
    load_hyperparams,
)
from app.models import (
    Action,
    ApiErrorEvent,
    DataUpload,
    Group,
    GroupHistory,
    HyperparameterSet,
    ModelParameters,
    ModelUpdateRequests,
    MonitorAlarm,
    MonitorPull,
    MonitorRun,
    NoiseInit,
    PolicyVersion,
    StudyData,
    UpdateWeek,
    UploadQuarantine,
    export_models,
)
from app.study_calendar import (
    active_dyads,
    cohort_warmup_active,
    day_in_study,
    enrolled_dyads,
    per_dyad_warmup,
    study_week,
    warmup_reason,
    week1_start,
    week_monday,
    weekly_deadline,
)

API_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETROIT = ZoneInfo("America/Detroit")
D = datetime.date
DT = datetime.datetime


@pytest.fixture
def fapp():
    app = create_app("config.TestingConfig")
    with app.app_context():
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def fclient(fapp):
    return fapp.test_client()


def _file_config(tmp_path, **attrs):
    """A TestingConfig subclass backed by a file SQLite DB under tmp_path."""
    base = {"SQLALCHEMY_DATABASE_URI": "sqlite:///" + str(tmp_path / "foundation.db")}
    base.update(attrs)
    return type("FileTestingConfig", (config.TestingConfig,), base)


# ===================================================================== models

NEW_TABLES = {
    "policy_versions",
    "hyperparameter_sets",
    "noise_init",
    "group_history",
    "upload_quarantine",
    "monitor_alarms",
    "api_error_events",
    "monitor_runs",
    "update_weeks",
    "monitor_pulls",
}
LEGACY_TABLES = {
    "groups",
    "data_uploads",
    "actions",
    "model_parameters",
    "thompson_sampling_params",
    "model_update_requests",
    "study_data",
    "standardization_baselines",
    "update_reproducibility_snapshots",
}


def test_create_all_builds_every_contract_table(fapp):
    from sqlalchemy import inspect

    inspector = inspect(db.engine)
    assert set(inspector.get_table_names()) == NEW_TABLES | LEGACY_TABLES

    def cols(table):
        return {c["name"]: c for c in inspector.get_columns(table)}

    actions = cols("actions")
    for name in ("policy_week", "policy_version_id", "hyperparams_hash", "mu", "v", "upload_id"):
        assert name in actions and actions[name]["nullable"]
    assert actions["raw_context"]["nullable"] and actions["model_parameters_id"]["nullable"]
    mp = cols("model_parameters")
    for name in ("study_week_monday", "policy_version_id", "update_rid", "hyperparams_hash"):
        assert name in mp
    assert any(ix["column_names"] == ["policy_version_id"] for ix in inspector.get_indexes("model_parameters"))
    assert cols("study_data")["state"]["nullable"]
    mur = cols("model_update_requests")
    for name in ("study_week_monday", "started_at", "published_at_study"):
        assert name in mur
    assert {"dedup_key", "check_list_version", "raised_at", "fallback_used"} <= set(cols("monitor_alarms"))
    assert "logical_now" in cols("monitor_runs")


def test_nullable_action_and_study_rows_and_new_models_persist(fapp):
    ts = DT(2026, 1, 12, 9, 0)
    action = Action(
        group_id="g1",
        action=1,
        rid="rid-no-state",
        state=None,
        decision_idx=0,
        decision_type="aya_message",
        raw_context=None,
        action_prob=0.5,
        random_state={"mode": "no_state", "u": 0.1},
        model_parameters_id=None,
        request_timestamp=ts,
        is_warmup=True,
        warmup_reason="no_state",
        policy_week=D(2026, 1, 12),
        hyperparams_hash="a" * 64,
        mu=0.1,
        v=0.2,
        upload_id=None,
    )
    sd = StudyData("g1", 0, "aya_message", 1, 0.5, None, {}, {"reward_native": 2}, 1.0, ts)
    pv = PolicyVersion("aya_message", D(2026, 1, 12), "upd-1", PolicyVersion.STATUS_PUBLISHED, n_dyads=5)
    mp = ModelParameters(
        snapshot_type="posterior",
        group_id="g1",
        decision_type="aya_message",
        theta=[0.0],
        covariance=[[1.0]],
        study_week_monday=D(2026, 1, 12),
        policy_version_id=1,
        update_rid="upd-1",
        hyperparams_hash="b" * 64,
    )
    mur = ModelUpdateRequests("upd-1", ts, study_week_monday=D(2026, 1, 12), started_at=ts)
    gh = GroupHistory("g1", ["a", "c"], "2026-01-05", "2026-04-15", rid="reg-1")
    q = UploadQuarantine("up-1", '{"bad": true}', "slot: invalid")
    alarm = MonitorAlarm("ev-1", "A13", "yellow", "rl", "inside_action", "pi invalid", "0.2", dedup_key="k1")
    err = ApiErrorEvent("ee-1", "AC-400-PARAM", "/api/v1/action", 400, title="Invalid Parameter", rid="r")
    run = MonitorRun("regular", "ok", logical_now=ts)
    week = UpdateWeek(D(2026, 1, 12), rids=["upd-1"])
    pull = MonitorPull("pull-1", 200, returned_count=0)
    noise = NoiseInit("aya_message", 0.158, 1.8148, 0.009875, 0.026333)
    hs = HyperparameterSet(1, "c" * 64, {"settings": {}}, in_force_from=ts)
    db.session.add_all([action, sd, pv, mp, mur, gh, q, alarm, err, run, week, pull, noise, hs])
    db.session.commit()

    assert Action.query.one().raw_context is None
    assert StudyData.query.one().state is None
    assert UploadQuarantine.query.one().payload_bytes == len(b'{"bad": true}')
    assert len(UploadQuarantine.query.one().payload_sha256) == 64
    assert MonitorAlarm.query.one().detail == {}
    assert MonitorRun.query.one().run_id == run.id
    assert UpdateWeek.query.one().rids == ["upd-1"] and UpdateWeek.query.one().status == "waiting"
    assert NoiseInit.query.one().source == "design"

    from sqlalchemy.exc import IntegrityError

    db.session.add(MonitorAlarm("ev-2", "A13", "yellow", "rl", "scheduled", "dup", "0.2", dedup_key="k1"))
    with pytest.raises(IntegrityError):
        db.session.commit()
    db.session.rollback()


def test_export_models_and_manifest(fapp, tmp_path):
    names = {m.__tablename__ for m in export_models()}
    assert names == (NEW_TABLES | LEGACY_TABLES) - {"upload_quarantine"}
    assert "upload_quarantine" in {m.__tablename__ for m in export_models(include_quarantine=True)}

    db.session.add(Group("g1", {"consent_start_date": "2026-01-05"}))
    db.session.add(UploadQuarantine("up-1", "{}", "bad"))
    db.session.commit()
    out = tmp_path / "exports"
    manifest = export_all_tables(str(out))
    assert "upload_quarantine.csv" not in manifest["files"]
    assert manifest["files"]["groups.csv"]["rows"] == 1
    on_disk = json.loads((out / "MANIFEST.json").read_text())
    assert on_disk["files"] == manifest["files"] and on_disk["include_quarantine"] is False
    import hashlib

    for filename, entry in manifest["files"].items():
        assert hashlib.sha256((out / filename).read_bytes()).hexdigest() == entry["sha256"]
    # JSON columns are exported as JSON, not Python repr.
    assert '{""consent_start_date"":""2026-01-05""}' in (out / "groups.csv").read_text()

    result = fapp.test_cli_runner().invoke(
        args=["export-csv", "--out", str(tmp_path / "q"), "--include-quarantine"]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "q" / "upload_quarantine.csv").exists()
    assert json.loads((tmp_path / "q" / "MANIFEST.json").read_text())["include_quarantine"] is True


# ================================================================= migration

# Schema at revision 20260723_01 (db.create_all() of app/models.py before this
# revision), captured from SQLite.
LEGACY_DDL_20260723 = """
CREATE TABLE groups (id INTEGER NOT NULL, group_id VARCHAR(255) NOT NULL, rid VARCHAR(255),
  group_info JSON NOT NULL, created_at DATETIME NOT NULL, PRIMARY KEY (id), UNIQUE (group_id));
CREATE TABLE data_uploads (id INTEGER NOT NULL, group_id VARCHAR(255) NOT NULL, rid VARCHAR(255),
  data JSON NOT NULL, request_timestamp DATETIME NOT NULL, created_at DATETIME NOT NULL,
  PRIMARY KEY (id), UNIQUE (rid));
CREATE TABLE model_parameters (id INTEGER NOT NULL, probability_of_action FLOAT, snapshot_type VARCHAR(64),
  group_id VARCHAR(255), decision_type VARCHAR(255), agent_decision_index INTEGER, sample_size INTEGER,
  feature_dim INTEGER, theta JSON, covariance JSON, perturbation JSON, metadata_json JSON,
  timestamp DATETIME NOT NULL, PRIMARY KEY (id));
CREATE TABLE actions (id INTEGER NOT NULL, group_id VARCHAR(255) NOT NULL, rid VARCHAR(255) NOT NULL,
  state JSON, decision_idx INTEGER NOT NULL, decision_type VARCHAR(255) NOT NULL, raw_context JSON NOT NULL,
  action INTEGER NOT NULL, action_prob FLOAT NOT NULL, is_warmup BOOLEAN NOT NULL, warmup_reason VARCHAR(32),
  model_param JSON, random_state JSON NOT NULL, model_parameters_id INTEGER NOT NULL,
  request_timestamp DATETIME NOT NULL, timestamp DATETIME NOT NULL, PRIMARY KEY (id),
  CONSTRAINT uq_action_group_type_idx UNIQUE (group_id, decision_type, decision_idx), UNIQUE (rid),
  FOREIGN KEY(model_parameters_id) REFERENCES model_parameters (id));
CREATE TABLE model_update_requests (id INTEGER NOT NULL, rid VARCHAR(255) NOT NULL, status VARCHAR(50) NOT NULL,
  request_timestamp DATETIME NOT NULL, created_at DATETIME NOT NULL, completed_at DATETIME,
  error_message VARCHAR(1024), PRIMARY KEY (id), UNIQUE (rid));
CREATE TABLE standardization_baselines (id INTEGER NOT NULL, group_id VARCHAR(255) NOT NULL,
  decision_type VARCHAR(255) NOT NULL, variable_name VARCHAR(255) NOT NULL, mu FLOAT NOT NULL,
  sigma FLOAT NOT NULL, sample_size INTEGER NOT NULL, created_at DATETIME NOT NULL, PRIMARY KEY (id),
  CONSTRAINT uq_baseline_group_dt_var UNIQUE (group_id, decision_type, variable_name));
CREATE TABLE study_data (id INTEGER NOT NULL, group_id VARCHAR(255) NOT NULL, decision_idx INTEGER NOT NULL,
  decision_type VARCHAR(255) NOT NULL, action INTEGER NOT NULL, action_prob FLOAT NOT NULL, state JSON NOT NULL,
  raw_context JSON NOT NULL, outcome JSON NOT NULL, reward FLOAT, request_timestamp DATETIME NOT NULL,
  derived_at DATETIME, created_at DATETIME NOT NULL, PRIMARY KEY (id),
  CONSTRAINT uq_study_group_type_idx UNIQUE (group_id, decision_type, decision_idx));
CREATE TABLE thompson_sampling_params (id INTEGER NOT NULL, group_id VARCHAR(255) NOT NULL,
  decision_type VARCHAR(255) NOT NULL, params JSON NOT NULL, updated_at DATETIME NOT NULL, PRIMARY KEY (id),
  CONSTRAINT uq_group_decision UNIQUE (group_id, decision_type));
CREATE TABLE update_reproducibility_snapshots (id INTEGER NOT NULL, rid VARCHAR(255) NOT NULL,
  model_parameters_id INTEGER, snapshot_dir VARCHAR(2048) NOT NULL, data_uploads_count INTEGER NOT NULL,
  actions_count INTEGER NOT NULL, groups_count INTEGER NOT NULL, total_bytes BIGINT NOT NULL,
  created_at DATETIME NOT NULL, PRIMARY KEY (id));
CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL,
  CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num));
INSERT INTO alembic_version VALUES ('20260723_01');
INSERT INTO model_parameters (probability_of_action, timestamp) VALUES (0.5, '2026-01-01 00:00:00');
INSERT INTO actions (group_id, rid, state, decision_idx, decision_type, raw_context, action, action_prob,
  is_warmup, random_state, model_parameters_id, request_timestamp, timestamp)
  VALUES ('g', 'r1', '[1.0]', 0, 'aya_message', '{}', 1, 0.5, 1, '{}', 1, '2026-01-05 09:00:00',
  '2026-01-05 09:00:01');
"""

# Runs in a subprocess: Alembic's fileConfig() would otherwise reset the test
# process's logging configuration.
_MIGRATION_SCRIPT = textwrap.dedent(
    """
    import json, sys
    from flask import Flask
    from flask_migrate import Migrate, upgrade
    from sqlalchemy import inspect
    from app.extensions import db
    import app.models  # noqa: F401

    mode, path, migrations = sys.argv[1], sys.argv[2], sys.argv[3]
    flask_app = Flask("migration_check")
    flask_app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + path
    db.init_app(flask_app)
    Migrate(flask_app, db, directory=migrations)
    with flask_app.app_context():
        if mode == "upgrade":
            upgrade(directory=migrations)
        else:
            db.create_all()
        ins = inspect(db.engine)
        out = {}
        for t in ins.get_table_names():
            if t == "alembic_version":
                continue
            out[t] = {
                "columns": sorted([c["name"], bool(c["nullable"])] for c in ins.get_columns(t)),
                "unique": sorted(
                    [list(u["column_names"]) for u in ins.get_unique_constraints(t)]
                    + [list(i["column_names"]) for i in ins.get_indexes(t) if i["unique"]]
                ),
                "indexes": sorted(list(i["column_names"]) for i in ins.get_indexes(t) if not i["unique"]),
                "fks": sorted([f["constrained_columns"], f["referred_table"]] for f in ins.get_foreign_keys(t)),
            }
        print("SCHEMA=" + json.dumps(out, sort_keys=True))
    """
)


def _schema_via_subprocess(mode: str, db_path: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _MIGRATION_SCRIPT, mode, db_path, os.path.join(API_ROOT, "migrations")],
        cwd=API_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PYTHONPATH": API_ROOT},
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = next(line for line in proc.stdout.splitlines() if line.startswith("SCHEMA="))
    return json.loads(line[len("SCHEMA="):])


@pytest.fixture(scope="module")
def reference_schema(tmp_path_factory):
    return _schema_via_subprocess("create_all", str(tmp_path_factory.mktemp("ref") / "ref.db"))


def test_migration_from_empty_db_matches_create_all(tmp_path, reference_schema):
    migrated = _schema_via_subprocess("upgrade", str(tmp_path / "empty.db"))
    assert set(migrated) == set(reference_schema)
    for table, spec in reference_schema.items():
        assert migrated[table] == spec, table


def test_migration_from_legacy_schema_matches_create_all_and_keeps_rows(tmp_path, reference_schema):
    db_path = str(tmp_path / "legacy.db")
    con = sqlite3.connect(db_path)
    con.executescript(LEGACY_DDL_20260723)
    con.close()
    migrated = _schema_via_subprocess("upgrade", db_path)
    assert set(migrated) == set(reference_schema)
    for table, spec in reference_schema.items():
        assert migrated[table] == spec, table
    con = sqlite3.connect(db_path)
    assert con.execute("SELECT version_num FROM alembic_version").fetchall() == [("20261019_01",)]
    assert con.execute("SELECT rid, model_parameters_id, policy_week FROM actions").fetchall() == [("r1", 1, None)]
    con.close()


def test_migration_converts_deployed_quarantine_text_to_exact_bytes(tmp_path, reference_schema):
    """The previous head stored quarantine bodies as Text on deployed DBs."""
    db_path = str(tmp_path / "previous-head.db")
    _schema_via_subprocess("create_all", db_path)
    raw_text = '{"note":"caf\u00e9"}'
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        DROP TABLE upload_quarantine;
        CREATE TABLE upload_quarantine (
          id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
          rid VARCHAR(255) NOT NULL,
          group_id VARCHAR(255),
          raw_body TEXT NOT NULL,
          validation_error TEXT NOT NULL,
          payload_sha256 VARCHAR(64) NOT NULL,
          payload_bytes INTEGER NOT NULL,
          received_at DATETIME NOT NULL
        );
        CREATE TABLE alembic_version (
          version_num VARCHAR(32) NOT NULL PRIMARY KEY
        );
        INSERT INTO alembic_version VALUES ('20261016_01');
        """
    )
    raw_bytes = raw_text.encode("utf-8")
    con.execute(
        """INSERT INTO upload_quarantine
           (rid, raw_body, validation_error, payload_sha256, payload_bytes, received_at)
           VALUES (?, ?, 'invalid', ?, ?, '2026-10-18 12:00:00')""",
        ("q-old", raw_text, hashlib.sha256(raw_bytes).hexdigest(), len(raw_bytes)),
    )
    con.commit()
    con.close()

    migrated = _schema_via_subprocess("upgrade", db_path)
    assert migrated["upload_quarantine"] == reference_schema["upload_quarantine"]
    con = sqlite3.connect(db_path)
    assert con.execute("SELECT typeof(raw_body), raw_body FROM upload_quarantine").fetchone() == (
        "blob",
        raw_bytes,
    )
    assert con.execute("SELECT version_num FROM alembic_version").fetchone() == ("20261019_01",)
    con.close()


# ===================================================================== clock


def test_to_study_local_naive_is_detroit_local():
    assert to_study_local("2026-11-02T03:00:10") == DT(2026, 11, 2, 3, 0, 10)
    assert to_study_local("2026-11-02 03:00:10") == DT(2026, 11, 2, 3, 0, 10)
    assert to_study_local(DT(2026, 7, 1, 8, 0)) == DT(2026, 7, 1, 8, 0)
    assert to_study_local("2026-11-02") == DT(2026, 11, 2, 0, 0)


def test_to_study_local_converts_aware_values():
    assert to_study_local("2026-11-02T08:00:10Z") == DT(2026, 11, 2, 3, 0, 10)  # EST, UTC-5
    assert to_study_local("2026-07-01T12:00:00+00:00") == DT(2026, 7, 1, 8, 0)  # EDT, UTC-4
    assert to_study_local("2026-07-01T09:00:00-03:00") == DT(2026, 7, 1, 8, 0)
    aware = DT(2026, 1, 5, 14, 0, tzinfo=datetime.timezone.utc)
    result = to_study_local(aware)
    assert result == DT(2026, 1, 5, 9, 0) and result.tzinfo is None


def test_to_study_local_dst_boundaries():
    # Fall back, 2026-11-01 02:00 EDT -> 01:00 EST: both 01:30s map to naive 01:30.
    assert to_study_local("2026-11-01T05:30:00Z") == DT(2026, 11, 1, 1, 30)
    assert to_study_local("2026-11-01T06:30:00Z") == DT(2026, 11, 1, 1, 30)
    # Spring forward, 2026-03-08 02:00 EST -> 03:00 EDT.
    assert to_study_local("2026-03-08T06:59:00Z") == DT(2026, 3, 8, 1, 59)
    assert to_study_local("2026-03-08T07:00:00Z") == DT(2026, 3, 8, 3, 0)


@pytest.mark.parametrize("garbage", ["", "   ", "not a time", "2026-13-01T00:00:00", 123, 1.5, None, True, D(2026, 1, 5)])
def test_to_study_local_rejects_garbage(garbage):
    with pytest.raises(ValueError):
        to_study_local(garbage)


def test_format_detroit_and_utc():
    assert format_detroit(DT(2026, 11, 2, 3, 0, 10)) == "2026-11-02T03:00:10-05:00"
    assert format_detroit(DT(2026, 7, 1, 8, 0)) == "2026-07-01T08:00:00-04:00"
    assert format_detroit(DT(2026, 7, 1, 12, 0, tzinfo=datetime.timezone.utc)) == "2026-07-01T08:00:00-04:00"
    assert to_utc(DT(2026, 1, 5, 9, 0)) == DT(2026, 1, 5, 14, 0, tzinfo=datetime.timezone.utc)


def test_server_now_is_naive_detroit_wall_clock():
    now = server_now()
    assert now.tzinfo is None
    reference = datetime.datetime.now(DETROIT).replace(tzinfo=None)
    assert abs((reference - now).total_seconds()) < 5


def test_study_now_host_mode_tracks_latest_host_timestamp(fapp):
    assert fapp.config["CLOCK_MODE"] == "host"
    assert abs((study_now() - server_now()).total_seconds()) < 5  # nothing received yet
    db.session.add(DataUpload("g1", {}, DT(2031, 3, 2, 8, 0), rid="u1"))
    db.session.add(ModelUpdateRequests("upd", DT(2031, 3, 2, 3, 0)))
    db.session.commit()
    assert study_now() == DT(2031, 3, 2, 8, 0)
    db.session.add(ModelUpdateRequests("upd2", DT(2031, 3, 9, 3, 0)))
    db.session.commit()
    assert study_now() == DT(2031, 3, 9, 3, 0)
    fapp.config["CLOCK_MODE"] = "wall"
    assert abs((study_now() - server_now()).total_seconds()) < 5


# ============================================================ study calendar


def test_week1_start_for_each_weekday():
    # 2026-01-05 is a Monday.
    expected = {
        D(2026, 1, 5): D(2026, 1, 5),
        D(2026, 1, 6): D(2026, 1, 12),
        D(2026, 1, 7): D(2026, 1, 12),
        D(2026, 1, 8): D(2026, 1, 12),
        D(2026, 1, 9): D(2026, 1, 12),
        D(2026, 1, 10): D(2026, 1, 12),
        D(2026, 1, 11): D(2026, 1, 12),
    }
    for consent, start in expected.items():
        assert week1_start(consent) == start
        assert week1_start(consent.isoformat()) == start
        assert week1_start(consent).weekday() == 0


def test_partial_week_study_week_and_day_in_study():
    consent = D(2026, 1, 14)  # Wednesday: partial week Wed-Sun, week 1 from 2026-01-19
    assert study_week(consent, D(2026, 1, 14)) == 0
    assert study_week(consent, D(2026, 1, 18)) == 0
    assert study_week(consent, D(2026, 1, 19)) == 1
    assert study_week(consent, D(2026, 1, 25)) == 1
    assert study_week(consent, D(2026, 1, 26)) == 2
    assert study_week(consent, DT(2026, 1, 26, 23, 59)) == 2
    assert day_in_study(consent, D(2026, 1, 14)) == 1
    assert day_in_study(consent, "2026-01-19") == 6
    assert study_week(D(2026, 1, 5), D(2026, 1, 5)) == 1  # Monday consent: no partial week
    assert week_monday(D(2026, 1, 18)) == D(2026, 1, 12)
    assert week_monday(DT(2026, 1, 19, 0, 0)) == D(2026, 1, 19)
    assert weekly_deadline(D(2026, 11, 4), "MON 05:45") == DT(2026, 11, 2, 5, 45)
    assert weekly_deadline(D(2026, 11, 2), "MON 03:30") == DT(2026, 11, 2, 3, 30)


# Five dyads; week1_start: A, B 2026-01-05; C, D 2026-01-12; E (Wednesday
# consent) 2026-01-19.
COHORT = {
    "A": "2026-01-05",
    "B": "2026-01-04",
    "C": "2026-01-12",
    "D": "2026-01-08",
    "E": "2026-01-14",
}


def test_cohort_rule_counts_dyads_that_completed_week1():
    # Week of 2026-01-19: all five have started week 1 (|E_W| = 5) but only four
    # have completed it (|E_{W-1}| = 4 < 5): still cohort warm-up (D2).
    assert len(enrolled_dyads(D(2026, 1, 19), COHORT)) == 5
    assert enrolled_dyads(D(2026, 1, 12), COHORT) == ["A", "B", "C", "D"]
    assert cohort_warmup_active(D(2026, 1, 19), COHORT, n_warm=5) is True
    # Any day of that week gives the same answer.
    assert cohort_warmup_active(D(2026, 1, 25), COHORT, n_warm=5) is True
    # Week of 2026-01-26: E completed week 1 on Sunday 2026-01-25.
    assert cohort_warmup_active(D(2026, 1, 26), COHORT, n_warm=5) is False
    assert cohort_warmup_active(D(2026, 1, 19), COHORT, n_warm=4) is False
    assert cohort_warmup_active(D(2026, 1, 5), COHORT, n_warm=2) is True


def test_active_and_enrolled_sets():
    assert active_dyads(D(2026, 1, 5), 14, COHORT) == ["A", "B"]
    assert active_dyads(D(2026, 1, 12), 14, COHORT) == ["A", "B", "C", "D"]
    # A/B week 14 is the week of 2026-04-06; from 2026-04-13 they are out of I_W
    # but stay in E_W (pooled forever).
    assert "A" in active_dyads(D(2026, 4, 6), 14, COHORT)
    assert active_dyads(D(2026, 4, 13), 14, COHORT) == ["C", "D", "E"]
    assert enrolled_dyads(D(2026, 4, 13), COHORT) == ["A", "B", "C", "D", "E"]
    assert active_dyads(D(2026, 1, 12), 1, COHORT) == ["C", "D"]
    # Default T comes from the hyperparameters (trial_weeks = 14).
    assert active_dyads(D(2026, 4, 6), groups=COHORT) == ["A", "B", "C", "D", "E"]


def test_per_dyad_warmup_and_reason_order():
    consent = D(2026, 1, 14)
    assert per_dyad_warmup(consent, D(2026, 1, 15), t_warm=1) is True  # partial week
    assert per_dyad_warmup(consent, D(2026, 1, 25), t_warm=1) is True  # week 1
    assert per_dyad_warmup(consent, D(2026, 1, 26), t_warm=1) is False  # week 2
    assert per_dyad_warmup(consent, D(2026, 1, 26), t_warm=2) is True
    assert per_dyad_warmup(consent, D(2026, 1, 26)) is False  # default t_warm = 1
    # Cohort is checked first.
    assert warmup_reason("2026-01-05", D(2026, 1, 21), COHORT) == "cohort"
    # Cohort satisfied from the week of 2026-01-26. A dyad consenting on
    # Wednesday 2026-01-28 has its partial week 01-28..02-01 and week 1 from 02-02.
    assert warmup_reason("2026-01-28", D(2026, 1, 29), COHORT) == "week1"
    assert warmup_reason("2026-01-28", D(2026, 2, 3), COHORT) == "week1"
    assert warmup_reason("2026-01-28", D(2026, 2, 9), COHORT) is None
    # Same dyad during cohort warm-up: cohort wins.
    assert warmup_reason("2026-01-14", D(2026, 1, 24), COHORT) == "cohort"
    assert warmup_reason("2026-01-14", D(2026, 1, 27), COHORT) is None  # its week 2
    assert warmup_reason("2026-01-05", D(2026, 1, 27), COHORT) is None


def test_calendar_reads_groups_table_and_history_rows(fapp):
    for gid, start in COHORT.items():
        db.session.add(Group(gid, {"member_list": [], "consent_start_date": start, "consent_end_date": "2026-06-01"}))
    db.session.commit()
    assert cohort_warmup_active(D(2026, 1, 19)) is True
    assert cohort_warmup_active(D(2026, 1, 26)) is False
    assert enrolled_dyads(D(2026, 1, 12)) == ["A", "B", "C", "D"]
    history = [GroupHistory(gid, [], start, "2026-06-01") for gid, start in COHORT.items()]
    # A later re-registration of A moves its consent date; the last row per dyad wins.
    history.append(GroupHistory("A", [], "2026-02-02", "2026-06-01"))
    assert enrolled_dyads(D(2026, 1, 12), history) == ["B", "C", "D"]


# ============================================================ hyperparameters


def test_defaults_equal_design_table1():
    hp = default_hyperparams()
    assert hp.discount == {"aya": 13 / 14, "cp": 6 / 7, "rel": 0.0}
    assert hp.trial_weeks == 14
    assert hp.respondent_weights == (1 / 3, 2 / 3)
    assert hp.w_adh == 0.0
    assert hp.sigma_sq == {"aya": 0.158, "cp": 0.122, "rel": 0.078}
    assert hp.normalize_rewards is True
    assert (hp.m, hp.nu0, hp.lambda0) == (2.0, 4.0, 0.1)
    assert (hp.l_min, hp.l_max, hp.c_rho, hp.b_rho, hp.kappa_rho) == (0.2, 0.8, 3.5, 20.0, 1.0)
    assert (hp.mc_samples, hp.mc_seed) == (500, 12345)
    assert (hp.n_warm, hp.t_warm) == (5, 1)
    assert (hp.j_backups, hp.k_max, hp.eps_tol, hp.eps_ridge) == (1, 200, 1e-4, 1e-6)
    assert hp.rho_at_zero == pytest.approx(1 / 3)


def test_derived_constants_per_agent():
    hp = default_hyperparams()
    expected = {
        # agent: (p_h, p_f, lambda_main, lambda_mod, tau0_act, tau0_mod, b_rho, r_max, run_in_mod)
        "aya": (7, 4, 16.0, 6.0, 0.009875, 0.026333, 40.0, 2.0, 24.0),
        "cp": (6, 4, 14.0, 6.0, 0.008714, 0.020333, 100.0, 5.0, 24.0),
        "rel": (7, 2, 16.0, 2.0, 0.004875, 0.039, 100.0, 5.0, 8.0),
    }
    for agent, (p_h, p_f, lm, lx, t_act, t_mod, b_rho, r_max, run_mod) in expected.items():
        a = hp.agent(agent)
        assert (a.p_h, a.p_f, a.p_main, a.p_mod) == (p_h, p_f, p_h + 1, p_f - 1)
        assert (a.lambda_main, a.lambda_mod) == (lm, lx)
        assert a.tau0_act == pytest.approx(t_act, abs=5e-7)
        assert a.tau0_mod == pytest.approx(t_mod, abs=5e-7)
        assert a.b_rho == b_rho and a.r_max == r_max
        assert a.run_in_main == lm and a.run_in_mod == run_mod
        assert a.local_penalty_diag() == [lm] * (p_h + 1) + [lx] * (p_f - 1)
        assert a.tau0_diag() == [a.tau0_act] + [a.tau0_mod] * (p_f - 1)
        assert hp.agent(a.decision_type) is a
    assert hp.agent("aya").sigma_bar_sq == pytest.approx(1 / (4 * (1 - (13 / 14) ** 2)))
    assert hp.agent("cp").sigma_bar_sq == pytest.approx(49 / 52)
    assert hp.agent("rel").sigma_bar_sq == pytest.approx(0.25)
    floor_act, floor_mod = hp.eb_floor("aya_message", 5)
    assert floor_act == pytest.approx(4 * 0.009875 / 11) and floor_mod == pytest.approx(4 * 0.158 / 6 / 11)


def test_w_adh_rescales_rel_reward_range_and_slope():
    hp = load_hyperparams(os.devnull, overrides={"rel_reward": {"w_adh": 0.5}})
    assert hp.agent("rel").r_max == 6.0 and hp.agent("rel").b_rho == 120.0


def test_unnormalized_rewards_use_native_slope_and_bound():
    hp = load_hyperparams(os.devnull, overrides={"rewards.normalize": False})
    assert hp.agent("cp").b_rho == 20.0 and hp.agent("cp").reward_divisor == 1.0
    assert hp.agent("cp").sigma_bar_sq == pytest.approx(25 * 49 / 52)


@pytest.mark.parametrize(
    "overrides, key",
    [
        ({"allocation": {"c_rh": 3.0}}, "allocation.c_rh"),
        ({"warmpu": {"n_warm": 5}}, "warmpu"),
        ({"discount": {"aya": 0.9, "parent": 0.5}}, "discount.parent"),
        ({"noise": {"sigma_sq": 0.1}}, "noise.sigma_sq"),
    ],
)
def test_unknown_or_malformed_key_is_named(overrides, key):
    with pytest.raises(HyperparameterError) as excinfo:
        load_hyperparams(os.devnull, overrides=overrides)
    assert excinfo.value.key == key
    assert repr(key) in str(excinfo.value)


@pytest.mark.parametrize(
    "overrides, key",
    [
        ({"discount.aya": 1.0}, "discount.aya"),
        ({"discount.cp": -0.1}, "discount.cp"),
        ({"allocation.l_min": 0.9}, "allocation.l_min"),
        ({"allocation.l_max": 1.0}, "allocation.l_max"),
        ({"allocation.mc_samples": 50}, "allocation.mc_samples"),
        ({"allocation.mc_samples": 500.5}, "allocation.mc_samples"),
        ({"warmup.n_warm": 1}, "warmup.n_warm"),
        ({"warmup.t_warm": 0}, "warmup.t_warm"),
        ({"rel_reward.respondent_weights": [0.5, 0.6]}, "rel_reward.respondent_weights"),
        ({"rel_reward.respondent_weights": [0.5]}, "rel_reward.respondent_weights"),
        ({"rel_reward.w_adh": -1}, "rel_reward.w_adh"),
        ({"noise.sigma_sq.rel": 0.3}, "noise.sigma_sq.rel"),
        ({"numerics.eps_tol": 0}, "numerics.eps_tol"),
        ({"penalty.m": "abc"}, "penalty.m"),
        ({"penalty.m": True}, "penalty.m"),
        ({"rewards.normalize": "yes"}, "rewards.normalize"),
        ({"eb.nu0": float("nan")}, "eb.nu0"),
    ],
)
def test_out_of_range_value_is_named(overrides, key):
    with pytest.raises(HyperparameterError) as excinfo:
        load_hyperparams(os.devnull, overrides=overrides)
    assert excinfo.value.key == key


def test_fraction_strings_and_numeric_strings(tmp_path):
    path = tmp_path / "fractions.yaml"
    path.write_text(
        'discount: {aya: "13/14", cp: "6/7"}\n'
        'rel_reward:\n  respondent_weights: ["1/3", "2/3"]\n'
        "numerics:\n  eps_tol: 1e-4\n"
    )
    hp = load_hyperparams(str(path))
    assert hp.discount["aya"] == 13 / 14 and hp.discount["cp"] == 6 / 7
    assert hp.eps_tol == 1e-4
    assert hp.sources["discount.aya"] == "file" and hp.sources["trial_weeks"] == "default"
    # Same resolved values as the defaults -> same fingerprint.
    assert hp.sha256 == default_hyperparams().sha256
    decimal = tmp_path / "decimal.yaml"
    decimal.write_text("discount: {aya: 0.9285714285714286}\n")
    assert load_hyperparams(str(decimal)).sha256 == default_hyperparams().sha256


def test_hash_is_stable_and_sensitive():
    a, b = default_hyperparams(), load_hyperparams(os.devnull)
    assert a.sha256 == b.sha256 and len(a.sha256) == 64
    assert a.canonical_json() == canonical_json(json.loads(a.canonical_json()))
    changed = load_hyperparams(os.devnull, overrides={"penalty.m": 3})
    assert changed.sha256 != a.sha256
    assert changed.agent("aya").lambda_main == 24.0
    # Provenance (path, sources) is not part of the fingerprint.
    assert load_hyperparams(os.devnull, overrides={"penalty.m": 2}).sha256 == a.sha256


def test_shipped_file_overrides_nothing_and_env_var(tmp_path, monkeypatch):
    with open(DEFAULT_HYPERPARAMS_FILE) as handle:
        lines = [line.strip() for line in handle if line.strip()]
    assert lines and all(line.startswith("#") for line in lines)
    monkeypatch.delenv("ADAPTS_HYPERPARAMS", raising=False)
    assert load_hyperparams().sha256 == default_hyperparams().sha256
    custom = tmp_path / "custom.yaml"
    custom.write_text("warmup:\n  n_warm: 3\n")
    monkeypatch.setenv("ADAPTS_HYPERPARAMS", str(custom))
    hp = load_hyperparams()
    assert hp.n_warm == 3 and hp.source_path == str(custom)
    monkeypatch.setenv("ADAPTS_HYPERPARAMS", str(tmp_path / "missing.yaml"))
    with pytest.raises(HyperparameterError):
        load_hyperparams()


def test_app_exposes_hyperparams_and_mirrors_warmup_keys(fapp):
    assert fapp.hyperparams.sha256 == default_hyperparams().sha256
    assert fapp.config["WARMUP_COHORT_MIN_DYADS"] == 5 and fapp.config["WARMUP_WEEKS"] == 1


def test_invalid_hyperparameter_file_refuses_start(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("allocation:\n  l_min: 0.95\n")
    cfg = type("BadHp", (config.TestingConfig,), {"HYPERPARAMS_PATH": str(bad)})
    with pytest.raises(HyperparameterError, match="allocation.l_min"):
        create_app(cfg)


def test_freeze_amend_and_mismatch(fapp, monkeypatch):
    hp = fapp.hyperparams
    assert latest_frozen() is None and in_force_hash() is None
    assert check_startup_against_frozen(hp) is None
    t0 = DT(2026, 2, 2, 3, 0)
    row = freeze_if_needed("first learning update", in_force_from=t0)
    assert (row.version, row.sha256, row.in_force_from) == (1, hp.sha256, t0)
    assert row.settings == json.loads(hp.canonical_json())
    # Idempotent.
    assert freeze_if_needed("again").id == row.id and HyperparameterSet.query.count() == 1
    assert check_startup_against_frozen(hp).id == row.id

    seen = []
    monkeypatch.setattr(monitoring_pkg, "on_startup_settings_mismatch", lambda detail: seen.append(detail))
    other = load_hyperparams(os.devnull, overrides={"penalty.m": 3})
    with pytest.raises(RuntimeError, match="Refusing to start"):
        check_startup_against_frozen(other)
    assert len(seen) == 1
    assert seen[0]["frozen_sha256"] == hp.sha256 and seen[0]["running_sha256"] == other.sha256
    assert {"key": "settings.penalty.m", "frozen": 2.0, "running": 3.0} in seen[0]["differences"]

    def broken(_detail):
        raise RuntimeError("alarm store down")

    monkeypatch.setattr(monitoring_pkg, "on_startup_settings_mismatch", broken)
    with pytest.raises(RuntimeError, match="Refusing to start"):
        check_startup_against_frozen(other)

    t1 = DT(2026, 3, 2, 3, 0)
    v2 = amend("protocol amendment 1", hp=other, in_force_from=t1)
    assert v2.version == 2 and v2.reason == "protocol amendment 1"
    assert check_startup_against_frozen(other).id == v2.id
    assert in_force_hash(DT(2026, 1, 1)) is None
    assert in_force_hash(DT(2026, 2, 10)) == hp.sha256
    assert in_force_hash(DT(2026, 3, 9)) == other.sha256
    assert in_force_version().version == 2
    with pytest.raises(RuntimeError, match="nothing to amend"):
        amend("same again", hp=other)
    with pytest.raises(ValueError):
        amend("  ", hp=hp)


def test_startup_refuses_when_frozen_set_differs(tmp_path, monkeypatch):
    cfg = _file_config(tmp_path)
    app = create_app(cfg)
    with app.app_context():
        freeze_if_needed("first learning update", in_force_from=DT(2026, 2, 2, 3, 0))
        db.session.remove()
    # Same settings: boots.
    create_app(cfg)
    seen = []
    monkeypatch.setattr(monitoring_pkg, "on_startup_settings_mismatch", lambda detail: seen.append(detail))
    changed = type("Changed", (cfg,), {"HYPERPARAMS_OVERRIDES": {"eb": {"nu0": 5}}})
    with pytest.raises(RuntimeError, match="Refusing to start"):
        create_app(changed)
    assert len(seen) == 1 and seen[0]["frozen_version"] == 1


def test_hyperparams_cli(fapp, tmp_path):
    runner = fapp.test_cli_runner()
    result = runner.invoke(args=["hyperparams", "show"])
    assert result.exit_code == 0, result.output
    assert "derived.agents.aya.lambda_main" in result.output and "frozen: no" in result.output

    good = tmp_path / "good.yaml"
    good.write_text("penalty:\n  m: 4\n")
    result = runner.invoke(args=["hyperparams", "validate", str(good)])
    assert result.exit_code == 0 and "sha256=" in result.output
    bad = tmp_path / "bad.yaml"
    bad.write_text("penalty:\n  mm: 4\n")
    result = runner.invoke(args=["hyperparams", "validate", str(bad)])
    assert result.exit_code != 0 and "penalty.mm" in result.output

    result = runner.invoke(args=["hyperparams", "diff", str(good)])
    assert result.exit_code == 0 and "Nothing is frozen" in result.output
    freeze_if_needed("test", in_force_from=DT(2026, 2, 2, 3, 0))
    result = runner.invoke(args=["hyperparams", "diff", str(good)])
    assert result.exit_code == 1 and "settings.penalty.m" in result.output
    result = runner.invoke(args=["hyperparams", "amend", "--reason", "raise m", "--file", str(good)])
    assert result.exit_code == 0, result.output
    assert latest_frozen().version == 2 and latest_frozen().reason == "raise m"
    result = runner.invoke(args=["hyperparams", "diff", str(good)])
    assert result.exit_code == 0 and "Identical" in result.output


# ============================================================ sampler bank


def test_mc_bank_matches_dedicated_generator_and_never_consumes_streams(tmp_path):
    plain = DeterministicSampleStream.fresh(n_normals=100, n_uniforms=50, seed=3)
    stream = DeterministicSampleStream.fresh(n_normals=100, n_uniforms=50, seed=3, mc_seed=12345, mc_samples=500)
    assert np.array_equal(plain.draw_normal(100), stream.draw_normal(100))
    stream.restore({"normal": 0, "uniform": 0})
    expected = np.random.default_rng(12345).standard_normal(500)
    assert stream.mc_bank_info == {"mc_seed": 12345, "mc_samples": 500}
    before = stream.cursor()
    bank = stream.get_mc_bank(12345, 500)
    assert np.array_equal(bank, expected)
    assert stream.cursor() == before
    with pytest.raises(ValueError):
        bank[0] = 1.0  # read-only
    # Other (seed, M): regenerated, identical to the dedicated generator.
    assert np.array_equal(stream.get_mc_bank(7, 200), np.random.default_rng(7).standard_normal(200))
    assert stream.cursor() == before


def test_mc_bank_save_load_round_trip(tmp_path):
    stream = DeterministicSampleStream.fresh(n_normals=100, n_uniforms=50, seed=3, mc_seed=99, mc_samples=300)
    stream.draw_uniform()
    stream.draw_normal(4)
    path = stream.save(str(tmp_path / "buf"))
    loaded = DeterministicSampleStream.load(path)
    assert loaded.cursor() == {"normal": 4, "uniform": 1}
    assert loaded.mc_bank_info == {"mc_seed": 99, "mc_samples": 300}
    assert np.array_equal(loaded.get_mc_bank(99, 300), np.random.default_rng(99).standard_normal(300))
    # A buffer saved without a bank still serves the right bank.
    bare = DeterministicSampleStream.fresh(n_normals=10, n_uniforms=10, seed=1)
    bare_loaded = DeterministicSampleStream.load(bare.save(str(tmp_path / "bare.npz")))
    assert bare_loaded.mc_bank_info is None
    assert np.array_equal(bare_loaded.get_mc_bank(5, 100), np.random.default_rng(5).standard_normal(100))


def test_app_sampler_carries_configured_bank(fapp):
    hp = fapp.hyperparams
    assert fapp.sampler.mc_bank_info == {"mc_seed": hp.mc_seed, "mc_samples": hp.mc_samples}
    assert fapp.sampler.n_uniforms == 50_000


def test_init_buffer_cli_stores_bank(tmp_path):
    cfg = type("BufCfg", (config.TestingConfig,), {"SAMPLE_BUFFER_PATH": None})
    app = create_app(cfg)
    target = tmp_path / "study_buffer.npz"
    app.config.update(SAMPLE_BUFFER_PATH=str(target), SAMPLE_BUFFER_NORMALS=100, SAMPLE_BUFFER_UNIFORMS=100)
    result = app.test_cli_runner().invoke(args=["init-buffer"])
    assert result.exit_code == 0, result.output
    loaded = DeterministicSampleStream.load(str(target))
    assert loaded.mc_bank_info == {"mc_seed": 12345, "mc_samples": 500}
    assert loaded.n_uniforms == 100


# ============================================== simulation config and restart


def test_simulation_style_file_db_uses_wal_and_restores_cursor(tmp_path):
    cfg = type(
        "SimCfg",
        (config.SimulationConfig,),
        {
            "SQLALCHEMY_DATABASE_URI": "sqlite:///" + str(tmp_path / "run" / "sim.db"),
            "SAMPLE_BUFFER_PATH": str(tmp_path / "run" / "buf.npz"),
            "REPRO_SNAPSHOT_ROOT": str(tmp_path / "run" / "repro"),
            "SAMPLE_BUFFER_NORMALS": 1000,
            "SAMPLE_BUFFER_UNIFORMS": 1000,
        },
    )
    assert cfg.CLOCK_MODE == "host" and cfg.FAULT_INJECTION_ENABLED is True
    assert cfg.UPDATE_RUN_INLINE is False and cfg.BACKUP_DATABASE is False
    app = create_app(cfg)
    assert os.path.exists(tmp_path / "run" / "buf.npz")
    with app.app_context():
        with db.engine.connect() as conn:
            assert conn.exec_driver_sql("PRAGMA journal_mode").scalar() == "wal"
            assert conn.exec_driver_sql("PRAGMA busy_timeout").scalar() == 30_000
        stamps = [(3, 0), (9, 0), (5, 0)]  # committed out of cursor order
        for i, (uniform, normal) in enumerate(stamps):
            db.session.add(
                Action(
                    "g1", 1, f"r{i}", None, i, "aya_message", None, 0.5,
                    {"mode": "warmup", "sampler_cursor_end": {"normal": normal, "uniform": uniform}},
                    None, DT(2026, 1, 5, 9, 0) + datetime.timedelta(days=i),
                )
            )
        db.session.commit()
        db.session.remove()
    restarted = create_app(cfg)
    assert restarted.sampler.cursor() == {"normal": 0, "uniform": 9}
    assert restarted.sampler.mc_bank_info == {"mc_seed": 12345, "mc_samples": 500}


def test_fault_settings_honoured_only_when_enabled(fapp):
    from app import fault_setting

    assert fault_setting("FAULT_ACTION_CRASH") == frozenset()
    assert fault_setting("FAULT_FORCE_PI") is None
    fapp.config.update(
        FAULT_ACTION_CRASH="aya_message, dyad_game",
        FAULT_FORCE_PI="0.95",
        FAULT_SAMPLER_SHIFT="3",
        FAULT_UPDATE_HANG_SECONDS=2,
    )
    assert fault_setting("FAULT_ACTION_CRASH") == {"aya_message", "dyad_game"}
    assert fault_setting("FAULT_FORCE_PI") == 0.95
    assert fault_setting("FAULT_SAMPLER_SHIFT") == 3
    assert fault_setting("FAULT_UPDATE_HANG_SECONDS") == 2.0
    fapp.config["FAULT_INJECTION_ENABLED"] = False
    assert fault_setting("FAULT_ACTION_CRASH") == frozenset()
    assert fault_setting("FAULT_FORCE_PI") is None
    assert fault_setting("FAULT_SAMPLER_SHIFT") == 0
    assert fault_setting("FAULT_UPDATE_HANG_SECONDS") == 0.0
    with pytest.raises(KeyError):
        fault_setting("FAULT_UNKNOWN")


def test_invalid_clock_mode_refuses_start():
    cfg = type("BadClock", (config.TestingConfig,), {"CLOCK_MODE": "logical"})
    with pytest.raises(ValueError, match="CLOCK_MODE"):
        create_app(cfg)


def test_production_never_enables_fault_injection():
    assert config.Config.FAULT_INJECTION_ENABLED is False
    assert config.ProductionConfig.FAULT_INJECTION_ENABLED is False
    assert config.Config.RL_ALGORITHM == os.getenv("RL_ALGORITHM", "inf_lsvi_eb")
    assert config.TestingConfig.RL_ALGORITHM == "inf_lsvi_eb"
    assert config.Config.SAMPLE_BUFFER_UNIFORMS == 200_000
    assert config.Config.MONITOR_WEEK_READY_BY == os.getenv("MONITOR_WEEK_READY_BY", "MON 05:45")
    assert config.Config.UPLOAD_SLOT_NOON_HOUR == 12


# ===================================================== framework answers (D19)


def _assert_envelope(response, status, title, message):
    body = response.get_json()
    assert response.status_code == status
    assert body["status"] == status and body["title"] == title and body["message"] == message
    assert isinstance(body["rid"], str) and body["rid"]
    assert body["server_timestamp"]
    return body


def test_framework_404_and_405_use_envelope_with_rid(fclient):
    a = _assert_envelope(fclient.get("/api/v1/nope"), 404, "Not Found", "Unknown route.")
    b = _assert_envelope(fclient.get("/api/v1/nope"), 404, "Not Found", "Unknown route.")
    assert a["rid"] != b["rid"]
    _assert_envelope(fclient.get("/api/v1/action"), 405, "Method Not Allowed", "Method not allowed.")
    _assert_envelope(fclient.delete("/api/v1/update"), 405, "Method Not Allowed", "Method not allowed.")


def test_framework_500_uses_envelope_with_rid():
    app = create_app("config.TestingConfig")

    @app.route("/__boom")
    def boom():
        raise ZeroDivisionError("boom")

    with app.app_context():
        _assert_envelope(app.test_client().get("/__boom"), 500, "Internal Error", "Internal server error.")
        db.drop_all()


def test_monitoring_load_failure_refuses_start(monkeypatch):
    def broken(_app):
        raise ImportError("monitoring missing")

    monkeypatch.setattr(monitoring_pkg, "init_app", broken)
    with pytest.raises(ImportError):
        create_app("config.TestingConfig")


def test_no_module_imports_the_top_level_monitoring_package():
    import ast

    for root in ("app", "tests", "tools"):
        for dirpath, _, files in os.walk(os.path.join(API_ROOT, root)):
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                with open(path, encoding="utf-8") as handle:
                    tree = ast.parse(handle.read(), filename=path)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        assert all(alias.name.split(".")[0] != "monitoring" for alias in node.names), path
                    elif isinstance(node, ast.ImportFrom) and node.level == 0:
                        assert (node.module or "").split(".")[0] != "monitoring", path
