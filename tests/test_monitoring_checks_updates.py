"""
WS-M tests of the upload checks (Monitoring-Spec §5.B: B11, B5, B8, B3), the
update checks (§5.C: C11, C7, C1, C3, C9, C10; F-U4) and the model
diagnostics (§5.L: L1-L4): one case where each fires and one where it does
not, on a temporary database built through the ORM.
"""

from __future__ import annotations

import datetime

import pytest

from app.extensions import db
from app.models import HyperparameterSet, ModelUpdateRequests, NoiseInit, UpdateWeek
from app.monitoring import on_startup_settings_mismatch
from app.monitoring.checks import diagnostics, update, uploads
from tests.test_monitoring_support import (  # noqa: F401 - fixtures
    AYA,
    CP,
    MON1,
    REL,
    D,
    add_action,
    add_dyad,
    add_model_row,
    add_update,
    add_upload,
    add_version,
    alarms,
    app_factory,
    at,
    ctx,
    keys,
    mon_app,
)

MON2 = D(2026, 1, 12)
MON3 = D(2026, 1, 19)
MON4 = D(2026, 1, 26)


# --------------------------------------------------------------------- B11


def test_b11_yellow_then_red_when_missed_two_weeks_in_a_row(mon_app):
    add_dyad("d1", MON1)
    add_dyad("d2", MON1)
    add_upload("d1", at(MON2, 2, 0))
    week2 = add_update("u2", at(MON2, 3, 0))
    findings, missing = uploads.b11_for_week(MON2, week2)
    assert missing == ["d2"] and keys(findings) == ["B11:2026-01-12:yellow"]
    assert findings[0].detail["group_ids"] == ["d2"] and findings[0].at == at(MON2, 3, 0)
    assert findings[0].message == "Week-closing upload missing at the 03:00 update for 1 dyads."
    add_upload("d1", at(MON3, 2, 0))
    week3 = add_update("u3", at(MON3, 3, 0))
    findings, _ = uploads.b11_for_week(MON3, week3)
    assert keys(findings) == ["B11:2026-01-19:red"] and findings[0].severity == "red"


def test_b11_quiet_when_every_active_dyad_uploaded(mon_app):
    add_dyad("d1", MON1)
    add_dyad("late", MON3)  # not active in the week before 01-12
    add_upload("d1", at(MON2, 1, 0))
    req = add_update("u2", at(MON2, 3, 0))
    assert uploads.b11_for_week(MON2, req) == ([], [])
    assert uploads.check_b11(ctx(at(MON2, 4, 0), at(MON2))) == []


def test_b11_upload_after_the_update_does_not_count(mon_app):
    add_dyad("d1", MON1)
    req = add_update("u2", at(MON2, 3, 0))
    add_upload("d1", at(MON2, 3, 5))
    assert uploads.b11_for_week(MON2, req)[1] == ["d1"]


# ---------------------------------------------------------------------- B5


def _b5_setup(app_factory, with_both=True):
    app = app_factory(HYPERPARAMS_OVERRIDES={"warmup.n_warm": 2})
    add_dyad("d1", MON1)
    add_dyad("d2", MON1)
    req = add_update("u3", at(MON3, 3, 0))
    for dt in (AYA, CP, REL):
        version = add_version(dt, MON3, "u3", published_at_study=at(MON3, 3, 1))
        for gid in (("d1", "d2") if (with_both or dt != AYA) else ("d1",)):
            add_model_row("local_fit", dt, version, group_id=gid, theta=[0.0])
            add_model_row("posterior", dt, version, group_id=gid, theta=[0.0])
    return app, req


def test_b5_fires_for_a_dyad_without_posterior(app_factory):
    _, req = _b5_setup(app_factory, with_both=False)
    found = uploads.b5_for_update(req)
    assert keys(found) == ["B5:u3:aya_message"]
    assert found[0].detail["dyads"] == [{"group_id": "d2", "cause": "no_local_fit"}]
    assert found[0].fallback_used == "F-A3"


def test_b5_quiet_when_every_dyad_has_a_policy(app_factory):
    _, req = _b5_setup(app_factory, with_both=True)
    assert uploads.b5_for_update(req) == []


def test_b5_quiet_during_cohort_warmup_and_week1(mon_app):
    add_dyad("d1", MON1)
    req = add_update("u3", at(MON3, 3, 0))
    assert uploads.b5_for_update(req) == []  # n_warm = 5: cohort warm-up, every decision is a coin flip


# ---------------------------------------------------------------------- B8


def test_b8_fires_without_upload_and_on_duplicate_week_closing(mon_app):
    add_dyad("d1", MON1)
    add_upload("d1", at(MON2, 1, 0))
    add_upload("d1", at(MON2, 2, 0))
    add_update("u2", at(MON2, 3, 0))
    add_action("d1", AYA, 0, at(MON2, 7, 30))  # nothing in 07:00-07:30
    found = uploads.check_b8(ctx(at(D(2026, 1, 13), 7, 0)))
    assert keys(found) == ["B8:2026-01-12"]
    assert found[0].detail["n_without_upload"] == 1
    assert found[0].detail["duplicate_week_closing"] == [{"group_id": "d1", "n_week_closing_uploads": 2}]


def test_b8_quiet_with_an_upload_before_each_decision(mon_app):
    add_dyad("d1", MON1)
    add_upload("d1", at(MON2, 2, 0))
    add_update("u2", at(MON2, 3, 0))
    add_upload("d1", at(MON2, 7, 25))
    add_action("d1", AYA, 0, at(MON2, 7, 30))
    add_upload("d1", at(MON2, 5, 59))
    add_action("d1", REL, 0, at(MON2, 6, 0))
    assert uploads.check_b8(ctx(at(D(2026, 1, 13), 7, 0))) == []


# ---------------------------------------------------------------------- B3


def _b3_days(miss_today: bool):
    add_dyad("d1", MON1)
    for k in range(7):
        add_upload("d1", at(MON1 + datetime.timedelta(days=k), 7, 25), aya_diary_mood=3, cp_diary_mood="miss")
    add_upload("d1", at(MON2, 7, 25), aya_diary_mood="miss" if miss_today else 4, cp_diary_mood="miss")


def test_b3_fires_when_a_missing_rate_jumps(mon_app):
    _b3_days(miss_today=True)
    found = uploads.check_b3(ctx(at(D(2026, 1, 13), 7, 0)))
    assert keys(found) == ["B3:2026-01-12"]
    assert [f["field"] for f in found[0].detail["fields"]] == ["aya_diary_mood"]


def test_b3_quiet_when_missing_rates_are_stable(mon_app):
    _b3_days(miss_today=False)
    assert uploads.check_b3(ctx(at(D(2026, 1, 13), 7, 0))) == []


# --------------------------------------------------------------------- C11


def test_c11_fires_when_no_update_by_0330(mon_app):
    add_dyad("d1", MON1)
    found = update.check_c11(ctx(at(MON2, 3, 30)))
    assert keys(found) == ["C11:2026-01-12"] and found[0].at == at(MON2, 3, 30)


def test_c11_quiet_when_the_update_arrived_or_none_is_due(mon_app):
    add_dyad("d1", MON1)
    assert update.check_c11(ctx(at(MON1, 3, 30))) == []  # nobody was active the week before 01-05
    add_update("u2", at(MON2, 3, 0))
    assert update.check_c11(ctx(at(MON2, 3, 30))) == []


# ---------------------------------------------------------------------- C7


@pytest.mark.parametrize("setup,reason", [
    (lambda: None, "never_arrived"),
    (lambda: add_update("u", at(MON2, 3, 0), status="processing"), "timeout"),
    (lambda: add_update("u", at(MON2, 5, 30), status="processing"), "running"),
    (lambda: add_update("u", at(MON2, 3, 0), status="failed", error_message="boom"), "failed"),
    (lambda: add_update("u", at(MON2, 3, 0), status="failed", error_message="timeout"), "timeout"),
    (lambda: add_update("u", at(MON2, 5, 40), published_at_study=at(MON2, 5, 50)), "late"),
    (lambda: add_update("u", at(MON2, 6, 0)), "never_arrived"),
])
def test_c7_fires_with_the_reason(mon_app, setup, reason):
    add_dyad("d1", MON1)
    setup()
    found = update.check_c7(ctx(at(MON2, 5, 45)))
    assert keys(found) == ["C7:2026-01-12"]
    assert found[0].detail["reason"] == reason and found[0].fallback_used == "F-U2/F-A3"


def test_c7_quiet_when_ready_by_0545(mon_app):
    add_dyad("d1", MON1)
    add_update("u", at(MON2, 3, 0), published_at_study=at(MON2, 3, 2))
    assert update.check_c7(ctx(at(MON2, 5, 45))) == []
    assert update.week_readiness(MON2)["ready_rid"] == "u"


def test_c7_fires_when_an_agent_version_failed(mon_app):
    add_dyad("d1", MON1)
    add_update("u", at(MON2, 3, 0), published_at_study=at(MON2, 3, 2))
    add_version(AYA, MON2, "u", status="failed")
    found = update.check_c7(ctx(at(MON2, 5, 45)))
    assert found and found[0].detail["reason"] == "not_all_published"


# --------------------------------------------------------------------- F-U4


def test_fu4_marks_a_stuck_update_failed_and_the_week_stuck(mon_app):
    add_update("stuck", at(MON2, 3, 0), status="processing")
    add_update("young", at(MON2, 3, 20), status="processing")
    assert update.mark_timeouts(at(MON2, 3, 35)) == ["stuck"]
    stuck = ModelUpdateRequests.query.filter_by(rid="stuck").one()
    assert (stuck.status, stuck.error_message) == ("failed", "timeout")
    assert ModelUpdateRequests.query.filter_by(rid="young").one().status == "processing"
    week = UpdateWeek.query.filter_by(study_week_monday=MON2).one()
    assert week.status == "stuck" and week.rids == ["stuck"]
    assert update.mark_timeouts(at(MON2, 3, 36)) == []


# ---------------------------------------------------------------------- C1


def test_c1_fires_on_failure_and_on_eb_fallback_red_after_two_weeks(mon_app):
    req = add_update("u2", at(MON2, 3, 0), status="failed", error_message="RuntimeError: boom [SQL: SELECT secret]")
    add_version(AYA, MON2, "u2", status="failed", reason="RuntimeError: boom")
    found = update.c1_for_update(req)
    assert keys(found) == ["C1:u2:aya_message:failed"] and found[0].severity is None
    assert "secret" not in str(found[0].detail)
    req3 = add_update("u3", at(MON3, 3, 0))
    version = add_version(AYA, MON3, "u3")
    add_model_row("hyper", AYA, version, meta={"converged": False, "fallback_used": "previous", "iterations": 200,
                                               "grad_inf": 0.003})
    found = update.c1_for_update(req3)
    assert keys(found) == ["C1:u3:aya_message:eb_fallback"]
    assert found[0].severity == "red" and found[0].detail["iterations"] == 200 and found[0].fallback_used == "F-U5"


def test_c1_failure_before_any_agent_is_reported_for_all(mon_app):
    req = add_update("u2", at(MON2, 3, 0), status="failed", error_message="derivation crashed")
    assert keys(update.c1_for_update(req)) == ["C1:u2:all:failed"]
    req3 = add_update("u3", at(MON3, 3, 0))
    version = add_version(CP, MON3, "u3")
    add_model_row("hyper", CP, version, meta={"converged": False, "fallback_used": "prior"})
    found = update.c1_for_update(req3)
    assert found[0].severity == "red"  # last week's whole-update failure counts for every agent
    assert "the prior" in found[0].message


def test_week_left_running_by_an_update_that_never_started_is_synced(mon_app):
    req = add_update("u2", at(MON2, 3, 0), status="processing")
    update.record_received(req)
    assert UpdateWeek.query.filter_by(study_week_monday=MON2).one().status == "running"
    req.status, req.error_message = "failed", "could not start the update thread"
    db.session.commit()
    update.mark_timeouts(at(MON2, 3, 1))
    assert UpdateWeek.query.filter_by(study_week_monday=MON2).one().status == "failed"


def test_c1_quiet_on_a_clean_update_and_on_timeout(mon_app):
    req = add_update("u2", at(MON2, 3, 0))
    version = add_version(AYA, MON2, "u2")
    add_model_row("hyper", AYA, version, meta={"converged": True, "fallback_used": None})
    assert update.c1_for_update(req) == []
    timeout = add_update("u3", at(MON3, 3, 0), status="failed", error_message="timeout")
    assert update.c1_for_update(timeout) == []


# ---------------------------------------------------------------------- C3


def test_c3_fires_on_rejection_and_turns_red_the_second_week(mon_app):
    for rid, week in (("u2", MON2), ("u3", MON3)):
        add_update(rid, at(week, 3, 0))
        version = add_version(CP, week, rid, status="rejected", reason="C3: finite")
        add_model_row("hyper", CP, version, meta={"c3": {"passed": False, "failures": ["tau2 below floor"]}})
    first = update.c3_for_update(ModelUpdateRequests.query.filter_by(rid="u2").one())
    second = update.c3_for_update(ModelUpdateRequests.query.filter_by(rid="u3").one())
    assert keys(first) == ["C3:u2:cp_message"] and first[0].severity is None
    assert first[0].detail["failures"] == ["tau2 below floor"]
    assert second[0].severity == "red"


def test_c3_quiet_when_published(mon_app):
    req = add_update("u2", at(MON2, 3, 0))
    add_version(CP, MON2, "u2")
    assert update.c3_for_update(req) == []


# ---------------------------------------------------------------------- C9


def _freeze(app, at_time, sha=None):
    db.session.add(HyperparameterSet(version=1, sha256=sha or app.hyperparams.sha256, settings={},
                                     in_force_from=at_time, reason="test"))
    db.session.commit()


def test_c9_fires_for_decisions_and_fits_without_the_approved_fingerprint(mon_app):
    _freeze(mon_app, at(MON2, 3, 0))
    add_dyad("d1", MON1)
    missing = add_action("d1", AYA, 0, at(MON2, 7, 30), hyperparams_hash=None)
    wrong = add_action("d1", AYA, 1, at(MON2, 19, 30), hyperparams_hash="0" * 64)
    assert update.c9_for_action(missing)[0].detail["kind"] == "missing"
    assert update.c9_for_action(wrong)[0].detail["kind"] == "mismatch"
    req = add_update("u2", at(MON2, 3, 0))
    version = add_version(AYA, MON2, "u2", hyperparams_hash=mon_app.hyperparams.sha256)
    add_model_row("posterior", AYA, version, group_id="d1", hyperparams_hash="0" * 64)
    found = update.c9_for_update(req)
    assert keys(found) == ["C9:fit:u2"] and found[0].detail["records"] == {"posterior:aya_message:mismatch": 1}


def test_c9_quiet_before_freezing_and_with_the_right_fingerprint(mon_app):
    add_dyad("d1", MON1)
    early = add_action("d1", AYA, 0, at(MON1, 7, 30), hyperparams_hash=None)
    assert update.c9_for_action(early) == []
    _freeze(mon_app, at(MON2, 3, 0))
    assert update.c9_for_action(early) == []  # decided before the set was in force
    good = add_action("d1", AYA, 1, at(MON2, 7, 30), hyperparams_hash=mon_app.hyperparams.sha256)
    assert update.c9_for_action(good) == []


def test_c9_startup_mismatch_alarm(mon_app):
    on_startup_settings_mismatch({"frozen_version": 1, "frozen_sha256": "a" * 64, "running_sha256": "b" * 64,
                                  "n_differences": 1, "differences": [{"key": "settings.penalty.m",
                                                                       "frozen": 2, "running": 3}]})
    rows = alarms("C9")
    assert len(rows) == 1 and rows[0].severity == "red" and rows[0].source == "startup"
    assert rows[0].detail["differences"][0]["key"] == "settings.penalty.m"


def test_c9_startup_refusal_end_to_end(tmp_path):
    """The API refuses to start on a changed settings file, and the C9 alarm is recorded first."""
    import sqlite3

    from tests.test_monitoring_support import make_app

    uri = f"sqlite:///{tmp_path / 'c9.db'}"
    app = make_app(SQLALCHEMY_DATABASE_URI=uri)
    with app.app_context():
        db.session.add(HyperparameterSet(version=1, sha256="f" * 64, settings=app.hyperparams.to_dict(),
                                         in_force_from=at(MON2, 3, 0), reason="first learning update"))
        db.session.commit()
        db.session.remove()
        db.engine.dispose()
    with pytest.raises(RuntimeError, match="Refusing to start"):
        make_app(SQLALCHEMY_DATABASE_URI=uri)
    with sqlite3.connect(tmp_path / "c9.db") as conn:
        rows = conn.execute("SELECT item_id, severity, source FROM monitor_alarms").fetchall()
    assert rows == [("C9", "red", "startup")]


# --------------------------------------------------------------------- C10


def _noise(clip):
    for dt in (AYA, CP, REL):
        db.session.add(NoiseInit(decision_type=dt, sigma_sq=0.1, sigma_bar_sq=0.2, tau0_act=0.01, tau0_mod=0.02,
                                 clip_used=clip and dt == CP, study_week_monday=MON2))
    db.session.commit()


def test_c10_green_record_after_the_first_learning_update(mon_app):
    assert update.c10_finding() == []
    _noise(clip=False)
    found = update.c10_finding()
    assert keys(found) == ["C10:noise_init"] and found[0].severity is None
    assert set(found[0].detail["agents"]) == {AYA, CP, REL}


def test_c10_yellow_when_the_limit_was_used(mon_app):
    _noise(clip=True)
    found = update.c10_finding()
    assert found[0].severity == "yellow" and found[0].detail["limit_used"] == [CP]


# ------------------------------------------------------------------ L1-L4


def test_l1_l2_l3_fire_from_the_update_rows(mon_app):
    req = add_update("u2", at(MON2, 3, 0))
    version = add_version(AYA, MON2, "u2")
    add_model_row("hyper", AYA, version, meta={"tau2": [0.0101, 0.5], "floor": [0.01, 0.01], "N_w": 3})
    add_model_row("posterior", AYA, version, group_id="d1", meta={"own_data_weight": 0.25})
    add_model_row("posterior", AYA, version, group_id="d2", meta={"own_data_weight": 0.75})
    add_model_row("local_fit", AYA, version, group_id="d1",
                  meta={"spectral_norm_GXtXplus": 1.4, "n_new_rows": 6, "theta_change_inf": 0.2})
    add_model_row("local_fit", AYA, version, group_id="d2",
                  meta={"spectral_norm_GXtXplus": 0.4, "n_new_rows": 0, "theta_change_inf": 1e-6})
    l1 = diagnostics.l1_for_update(req)
    assert keys(l1) == ["L1:u2:aya_message"] and l1[0].detail["blocks"] == ["act"]
    l2 = diagnostics.l2_for_update(req)
    assert keys(l2) == ["L2:u2:aya_message"] and l2[0].value == pytest.approx(0.5)
    l3 = diagnostics.l3_for_update(req)
    assert keys(l3) == ["L3:u2:aya_message", "L3:u2:aya_message:theta_change"]
    assert l3[0].severity is None and l3[0].detail["group_ids"] == ["d1"]
    assert l3[1].severity == "green" and l3[1].detail["theta_change_inf"] == {"d2": 1e-6}


def test_l1_l2_l3_quiet_without_findings(mon_app):
    req = add_update("u2", at(MON2, 3, 0))
    version = add_version(REL, MON2, "u2")
    add_model_row("hyper", REL, version, meta={"tau2": [0.5, 0.5], "floor": [0.01, 0.01]})
    add_model_row("local_fit", REL, version, group_id="d1", meta={"spectral_norm_GXtXplus": 3.0})  # REL: no L3
    assert diagnostics.l1_for_update(req) == []
    assert diagnostics.l2_for_update(req) == []
    assert diagnostics.l3_for_update(req) == []


def test_l4_reports_pi_percentiles_of_the_completed_week(mon_app):
    add_dyad("d1", MON1)
    for k, pi in enumerate((0.2, 0.4, 0.6, 0.8, 0.5)):
        add_action("d1", AYA, k, at(MON2 + datetime.timedelta(days=k), 7, 30), pi=pi, warmup=False)
    add_action("d1", AYA, 9, at(MON2, 19, 30), pi=0.5, warmup=True)  # warm-up: not counted
    found = diagnostics.check_l4(ctx(at(MON3, 4, 0)))
    assert keys(found) == ["L4:2026-01-12:aya_message"]
    assert found[0].detail["n_learned"] == 5 and found[0].detail["percentiles"]["p50"] == pytest.approx(0.5)
    assert found[0].detail["share_near_limit"] == pytest.approx(0.4)


def test_l4_quiet_without_learned_decisions(mon_app):
    add_dyad("d1", MON1)
    add_action("d1", AYA, 0, at(MON2, 7, 30), pi=0.5, warmup=True)
    assert diagnostics.check_l4(ctx(at(MON3, 4, 0))) == []
