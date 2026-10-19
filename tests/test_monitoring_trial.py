"""
WS-M tests on a learning trial driven through the real routes (Monitoring-Spec
§11.3 items 2-4; C5, A4, A10; contract D11, D13, D14).

Three dyads consent on Monday 2026-01-05 with ``n_warm = 2``: the 2026-01-12
update is the first learning update and the 2026-01-19 update the second;
from 01-12 on, AYA, CP and REL decisions are learned. Monitoring runs once
per simulated day with an explicit window (the "live" monitoring).

- C5 reruns each week's update on a copy of the database and matches every
  published number exactly, also with a second ``/update`` in the same week;
  a changed stored result is detected.
- A4 reproduces every decision made by ``/action``; a shifted random-number
  source (``FAULT_SAMPLER_SHIFT``) and a posterior changed after the decision
  are detected.
- A rerun over the whole trial finds exactly the problems the live
  monitoring found (§11.3 item 4) and raises nothing new.
"""

from __future__ import annotations

import datetime

import pytest

from app.extensions import db
from app.models import Action, ModelParameters, ModelUpdateRequests, PolicyVersion
from app.monitoring import run_monitoring
from app.monitoring.checks import decisions
from app.monitoring.checks.rerun import c5_for_update
from tests.test_monitoring_support import AYA, MON1, D, HttpTrial, alarms, at, ctx, make_app

CONSENTS = {"d1": MON1, "d2": MON1, "d3": MON1}
OVERRIDES = {"warmup.n_warm": 2}


def _trial(start: D, end: D, *, live: bool = True, seed: int = 3, app=None):
    app = app or make_app(HYPERPARAMS_OVERRIDES=OVERRIDES)
    ctx_ = app.app_context()
    ctx_.push()
    client = app.test_client()
    trial = HttpTrial(client, CONSENTS, seed=seed)
    previous = at(start)
    d = start
    while d <= end:
        trial.run_day(d)
        if live:
            now = at(d + datetime.timedelta(days=1))
            run_monitoring(now=now, since=previous)
            previous = now
        d += datetime.timedelta(days=1)
    return app, ctx_, client, trial


def _teardown(ctx_):
    db.session.remove()
    db.drop_all()
    ctx_.pop()


@pytest.fixture(scope="module")
def trial():
    app, ctx_, client, t = _trial(MON1, D(2026, 1, 20))
    yield app, client, t
    _teardown(ctx_)


def test_trial_learns_and_publishes_every_week(trial):
    app, _, t = trial
    for week in (D(2026, 1, 12), D(2026, 1, 19)):
        versions = PolicyVersion.query.filter_by(study_week_monday=week).all()
        assert {v.decision_type for v in versions} == {"aya_message", "cp_message", "dyad_game"}
        assert all(v.status == "published" for v in versions)
    learned = Action.query.filter(Action.is_warmup.is_(False)).count()
    assert learned > 50
    assert all(a.hyperparams_hash == app.hyperparams.sha256 for a in Action.query.all())


def test_clean_trial_raises_no_red_alarm(trial):
    red = [(a.item_id, a.message) for a in alarms() if a.severity == "red"]
    assert red == []
    assert alarms("A4") == [] and alarms("A10") == [] and alarms("C5") == [] and alarms("C9") == []
    assert {"C10", "L2", "G0"} <= {a.item_id for a in alarms()}


def test_c5_reruns_match_exactly(trial):
    rids = [r.rid for r in ModelUpdateRequests.query.order_by(ModelUpdateRequests.id).all()]
    learning = [rid for rid in rids if ModelParameters.query.filter_by(update_rid=rid).count()]
    assert len(learning) == 2
    for rid in learning:
        assert c5_for_update(ModelUpdateRequests.query.filter_by(rid=rid).one()) == []


def test_a4_reproduces_every_decision_of_the_trial(trial):
    for day in range(6, 21):
        assert decisions.check_a4(ctx(at(D(2026, 1, day), 1, 0))) == []


def test_full_rerun_finds_exactly_what_live_monitoring_found(trial):
    live = {a.dedup_key for a in alarms() if a.item_id not in ("E5", "E7")}
    result = run_monitoring(now=at(D(2026, 1, 21)), since=at(MON1))
    assert result["n_alarms"] == 0 and result["errors"] == []
    assert live <= set(result["findings"])
    assert set(result["findings"]) <= {a.dedup_key for a in alarms()}


# --------------------------------------------------- faults on a short trial


@pytest.fixture
def short_trial():
    """Through the first learning update (2026-01-12) and that Monday's decisions."""
    app, ctx_, client, t = _trial(MON1, D(2026, 1, 12), live=False)
    yield app, client, t
    _teardown(ctx_)


def _learning_rids():
    return [r for r in ModelUpdateRequests.query.order_by(ModelUpdateRequests.id).all()
            if ModelParameters.query.filter_by(update_rid=r.rid).count()]


def test_second_update_in_the_same_week_still_matches(short_trial):
    _, _, t = short_trial
    t.update(at(D(2026, 1, 12), 4, 0))
    rids = _learning_rids()
    assert [r.request_timestamp for r in rids] == [at(D(2026, 1, 12), 3, 0), at(D(2026, 1, 12), 4, 0)]
    for req in rids:
        assert c5_for_update(req) == []
    from app.learner.policy import policy_in_force

    # Contract D13: between the two runs the first is in force; once both are published (before 05:45)
    # the rerun supersedes it by recency for the whole week.
    assert policy_in_force(AYA, at(D(2026, 1, 12), 3, 30)).update_rid == rids[0].rid
    assert policy_in_force(AYA, at(D(2026, 1, 12), 7, 30)).update_rid == rids[1].rid
    assert policy_in_force(AYA, at(D(2026, 1, 15), 7, 30)).update_rid == rids[1].rid


def test_c5_detects_a_changed_published_result(short_trial):
    req = _learning_rids()[0]
    row = ModelParameters.query.filter_by(update_rid=req.rid, snapshot_type="posterior").first()
    theta = list(row.theta)
    theta[0] += 1e-12
    row.theta = theta
    db.session.commit()
    found = c5_for_update(req)
    assert len(found) == 1 and found[0].detail["kind"] == "mismatch"
    diff = found[0].detail["differences"]
    assert {"agent": row.decision_type, "snapshot": "posterior", "group_id": row.group_id, "kind": "value",
            "field": "theta"} == {k: diff[0][k] for k in ("agent", "snapshot", "group_id", "kind", "field")}


def test_a4_detects_a_shifted_random_number_source(short_trial):
    app, _, t = short_trial
    app.config["FAULT_SAMPLER_SHIFT"] = 1
    t.run_day(D(2026, 1, 13))
    app.config["FAULT_SAMPLER_SHIFT"] = 0
    found = decisions.check_a4(ctx(at(D(2026, 1, 14), 1, 0)))
    kinds = {}
    for f in found:
        kinds.update(f.detail["kinds"])
    assert "uniform_mismatch" in kinds


def test_a4_detects_a_posterior_changed_after_the_decision(short_trial):
    learned = Action.query.filter(Action.is_warmup.is_(False), Action.model_parameters_id.isnot(None)).first()
    assert learned is not None
    assert decisions.check_a4(ctx(at(learned.request_timestamp.date() + datetime.timedelta(days=1), 1, 0))) == []
    post = db.session.get(ModelParameters, learned.model_parameters_id)
    post.theta = [x + 0.5 for x in post.theta]
    db.session.commit()
    found = decisions.check_a4(ctx(at(learned.request_timestamp.date() + datetime.timedelta(days=1), 1, 0)))
    assert any("pi_mismatch" in f.detail["kinds"] for f in found)
