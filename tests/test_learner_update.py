"""
WS-L end-to-end tests of the weekly update and the policy in force on the
TestingConfig app (implementation contract D2, D8-D16, D21; main_new.tex
Algorithm 3; Monitoring-Spec C3, C10, F-U2, F-U3, F-U5).

Six dyads with staggered consent dates (three on non-Mondays) are driven
through the API-Spec §3 calling flow at the ORM level
(``tests.test_learner_support.MiniTrial``). Calendar (n_warm = 5):

- updates on 2026-01-05, -12, -19 are cohort warm-up (|E_{W-1}| = 0, 1, 4);
- 2026-01-26 is the first learning update (d1..d5 have completed week 1;
  d6 enrolled on Tuesday 01-20 and is in its week 1);
- 2026-02-02 is the second learning update.
"""

from __future__ import annotations

import datetime
import json

import numpy as np
import pytest

from app import learner
from app.extensions import db
from app.learner import features as F
from app.learner.allocation import agent_probability
from app.models import (
    DataUpload,
    HyperparameterSet,
    ModelParameters,
    NoiseInit,
    PolicyVersion,
)
from tests.test_learner_support import (  # noqa: F401 - fixture
    MiniTrial,
    add_group,
    add_upload,
    learner_app,
    make_app,
    snapshot,
)

D = datetime.date
DT = datetime.datetime
CONSENTS = {
    "d1": D(2026, 1, 5),   # Monday
    "d2": D(2026, 1, 7),   # Wednesday -> partial week, week 1 from 01-12
    "d3": D(2026, 1, 8),   # Thursday
    "d4": D(2026, 1, 12),  # Monday
    "d5": D(2026, 1, 15),  # Thursday -> week 1 from 01-19
    "d6": D(2026, 1, 20),  # Tuesday  -> week 1 from 01-26
}
W1 = D(2026, 1, 26)
W2 = D(2026, 2, 2)


def _run(trial: MiniTrial, start: D, end: D, *, update: bool = True) -> None:
    d = start
    while d <= end:
        trial.run_day(d, update=update)
        d += datetime.timedelta(days=1)


def _versions(week: D, update_rid: str | None = None) -> dict:
    q = PolicyVersion.query.filter_by(study_week_monday=week)
    if update_rid is not None:
        q = q.filter_by(update_rid=update_rid)
    return {v.decision_type: v for v in q.order_by(PolicyVersion.id.asc()).all()}


def _rows(version_id: int, snapshot_type: str) -> dict:
    rows = ModelParameters.query.filter_by(policy_version_id=version_id, snapshot_type=snapshot_type).all()
    return {r.group_id: r for r in rows}


@pytest.fixture(scope="module")
def trial_app():
    """One trial through the 2026-02-02 update and that Monday's decisions (read-only for the tests below)."""
    app = make_app()
    with app.app_context():
        trial = MiniTrial(CONSENTS, seed=1)
        _run(trial, D(2026, 1, 5), D(2026, 2, 2))
        yield app, trial
        db.session.remove()
        db.drop_all()


# ------------------------------------------------------------ warm-up weeks


def test_cohort_warmup_weeks_publish_nothing(trial_app):
    _, trial = trial_app
    by_week = {s["study_week_monday"]: s for s in trial.update_summaries}
    for week in ("2026-01-05", "2026-01-12", "2026-01-19"):
        s = by_week[week]
        assert s["skip_reason"] == "cohort_warmup" and not s["learning"]
        assert s["status"] == "completed"
        assert all(a["status"] == "skipped" for a in s["agents"].values())
        assert PolicyVersion.query.filter_by(study_week_monday=D.fromisoformat(week)).count() == 0
    # Every decision before the first learning update was a coin flip.
    early = [r for r in trial.decisions if r["ts"] < DT(2026, 1, 26, 3, 0)]
    assert early and all(r["warmup"] and r["pi"] == 0.5 for r in early)
    assert {r["reason"] for r in early} <= {"cohort", "week1"}


def test_first_learning_update_freezes_and_records_noise(trial_app):
    app, trial = trial_app
    s = next(x for x in trial.update_summaries if x["study_week_monday"] == W1.isoformat())
    assert s["learning"] and s["first_learning_update"] and s["noise_init_written"]
    assert s["hyperparams_version"] == 1
    sets = HyperparameterSet.query.all()
    assert len(sets) == 1
    assert sets[0].sha256 == app.hyperparams.sha256
    assert sets[0].in_force_from == DT(2026, 1, 26, 3, 0)
    noise = {n.decision_type: n for n in NoiseInit.query.all()}
    assert set(noise) == set(learner.AGENTS)
    for dt, row in noise.items():
        a = app.hyperparams.agent(dt)
        assert row.sigma_sq == a.sigma_sq and row.source == "design" and not row.clip_used
        assert row.tau0_act == pytest.approx(a.tau0_act) and row.tau0_mod == pytest.approx(a.tau0_mod)
        assert row.n_week1_rewards is None and row.s2 is None
    second = next(x for x in trial.update_summaries if x["study_week_monday"] == W2.isoformat())
    assert not second["first_learning_update"]
    assert NoiseInit.query.count() == 3


def test_first_learning_update_publishes_every_agent(trial_app):
    app, trial = trial_app
    hp = app.hyperparams
    s = next(x for x in trial.update_summaries if x["study_week_monday"] == W1.isoformat())
    versions = _versions(W1)
    assert set(versions) == set(learner.AGENTS)
    rid = s["update_rid"]
    expected_fit = {"aya_message": set(CONSENTS), "cp_message": set(CONSENTS), "dyad_game": set(CONSENTS) - {"d6"}}
    for dt, v in versions.items():
        agent = s["agents"][dt]
        assert v.status == "published" and agent["status"] == "published"
        assert v.update_rid == rid and v.hyperparams_hash == hp.sha256
        assert agent["c3"]["passed"] and agent["eb"]["converged"] and agent["eb"]["fallback_used"] is None
        assert DT(2026, 1, 26, 3, 0) <= v.published_at_study <= DT(2026, 1, 26, 3, 1)
        fits = _rows(v.id, "local_fit")
        posts = _rows(v.id, "posterior")
        hyper = _rows(v.id, "hyper")
        # Every enrolled dyad with data is fitted; every fitted dyad in I_w has a posterior.
        assert set(fits) == expected_fit[dt]
        assert set(posts) == expected_fit[dt] & set(s["active"])
        assert v.n_dyads == len(posts) and agent["n_dyads_fit"] == len(fits)
        assert len(hyper) == 1
        p_h, p_f = F.dims(dt)
        for row in list(fits.values()) + list(posts.values()) + list(hyper.values()):
            assert row.update_rid == rid and row.study_week_monday == W1 and row.hyperparams_hash == hp.sha256
            assert row.decision_type == dt
        for row in fits.values():
            assert len(row.theta) == p_h + p_f and np.asarray(row.covariance).shape == (p_h + p_f,) * 2
            assert row.metadata_json["warm_start"] == {"source": "zero"}
        for row in posts.values():
            assert len(row.theta) == p_f and 0 < row.metadata_json["own_data_weight"] < 1
        hmeta = next(iter(hyper.values())).metadata_json
        assert hmeta["N_w"] == len(fits)
        assert all(t >= f for t, f in zip(hmeta["tau2"], hmeta["floor"]))
    # d6 is in week 1 with no REL decision yet: no REL posterior, and not a B5 case.
    assert s["agents"]["dyad_game"]["dyads_without_posterior"] == {}


def test_local_fit_rows_are_the_usable_decisions(trial_app):
    from app.models import Action, StudyData

    _, trial = trial_app
    v = _versions(W1)["aya_message"]
    fit = _rows(v.id, "local_fit")["d1"]
    usable = []
    for a in Action.query.filter_by(group_id="d1", decision_type="aya_message").order_by(Action.request_timestamp, Action.id):
        if a.request_timestamp >= DT(2026, 1, 26):
            continue
        sd = StudyData.query.filter_by(group_id="d1", decision_type="aya_message", decision_idx=a.decision_idx).first()
        if sd is not None and sd.reward is not None and (a.raw_context or {}).get("design_state"):
            usable.append(a.id)
    assert fit.metadata_json["action_ids"] == usable
    assert fit.sample_size == len(usable) == 42  # three full weeks of twice-daily decisions
    succ = fit.metadata_json["successor"]
    assert succ["boundary"] is True and succ["upload_timestamp"] == "2026-01-26T02:00:00"
    # Week w-1 (01-19) was cohort warm-up for everyone: pi_check = 1/2 at both candidates.
    assert succ["pi_check"] == [0.5, 0.5]
    q = succ["q_rel"]
    assert succ["weights"] == [pytest.approx(1 - q), pytest.approx(q)]
    rel_post = _rows(_versions(W1)["dyad_game"].id, "posterior")["d1"]
    assert rel_post.metadata_json["q_rel"] == pytest.approx(q, abs=0)


def test_learned_probability_is_consistent_with_the_posterior(trial_app):
    app, _ = trial_app
    hp = app.hyperparams
    t = DT(2026, 1, 27, 7, 30)
    up = add_upload(
        "d1",
        DT(2026, 1, 27, 7, 26),
        snapshot(aya_app_engagement=3, aya_app_burden=4.0, current_game_on=1, previous_med_adherence=1,
                 aya_diary_mood=3, week_in_study=4),
        rid="probe-consistency",
    )
    try:
        ds = learner.build_design_state("d1", "aya_message", up, t)
        dec = learner.learned_probability("d1", "aya_message", ds, t)
        version = _versions(W1)["aya_message"]
        assert dec.policy_version_id == version.id and dec.policy_week == W1
        assert dec.hyperparams_hash == hp.sha256
        post = db.session.get(ModelParameters, dec.model_parameters_id)
        assert post.snapshot_type == "posterior" and post.group_id == "d1" and post.policy_version_id == version.id
        alpha = np.asarray(post.theta)
        cov = np.asarray(post.covariance)
        _, f = F.hf_from_design_state("aya_message", ds)
        assert dec.mu == float(f @ alpha)
        assert dec.v == float(f @ cov @ f)
        assert dec.model_param == list(alpha) + list(cov.ravel())
        assert dec.action_prob == agent_probability("aya_message", dec.mu, dec.v)
        assert 0.2 <= dec.action_prob <= 0.8
    finally:
        db.session.delete(up)
        db.session.commit()


def test_learned_decisions_in_the_trial_are_valid(trial_app):
    _, trial = trial_app
    learned = [r for r in trial.decisions if not r["warmup"]]
    assert len(learned) > 100
    assert all(0.2 <= r["pi"] <= 0.8 for r in learned)
    assert all(r["ts"] >= DT(2026, 1, 26, 3, 0) for r in learned)
    # d6 is in week 1 during 01-26..02-01: scheduled warm-up, never learned.
    d6_week1 = [r for r in trial.decisions if r["group_id"] == "d6" and DT(2026, 1, 26) <= r["ts"] < DT(2026, 2, 2)]
    assert d6_week1 and all(r["reason"] == "week1" for r in d6_week1)
    # The REL decision of week 01-26 used the 01-26 REL version.
    rel = next(r for r in learned if r["group_id"] == "d1" and r["decision_type"] == "dyad_game")
    assert rel["decision"].policy_week == W1


def test_no_policy_cases(trial_app):
    up = DataUpload.query.filter_by(group_id="d6").order_by(DataUpload.request_timestamp.desc()).first()
    ds6 = learner.build_design_state("d6", "dyad_game", up, DT(2026, 1, 26, 6, 0))
    # d6 has no REL posterior in the 01-26 version.
    with pytest.raises(learner.NoPolicy):
        learner.learned_probability("d6", "dyad_game", ds6, DT(2026, 1, 26, 6, 0))
    up1 = DataUpload.query.filter_by(group_id="d1").order_by(DataUpload.request_timestamp.desc()).first()
    ds1 = learner.build_design_state("d1", "aya_message", up1, DT(2026, 1, 26, 7, 30))
    # Before the first version is published nothing is in force ...
    with pytest.raises(learner.NoPolicy):
        learner.learned_probability("d1", "aya_message", ds1, DT(2026, 1, 21, 7, 30))
    # ... including at the update's own request time, before its run finished.
    with pytest.raises(learner.NoPolicy):
        learner.learned_probability("d1", "aya_message", ds1, DT(2026, 1, 26, 3, 0, 0))
    assert learner.learned_probability("d1", "aya_message", ds1, DT(2026, 1, 26, 6, 0)).policy_week == W1
    # An unknown dyad has no posterior.
    with pytest.raises(learner.NoPolicy):
        learner.learned_probability("nobody", "aya_message", ds1, DT(2026, 1, 27, 7, 30))


def test_second_week_uses_warm_starts_and_previous_policy(trial_app):
    app, trial = trial_app
    s1 = next(x for x in trial.update_summaries if x["study_week_monday"] == W1.isoformat())
    s2 = next(x for x in trial.update_summaries if x["study_week_monday"] == W2.isoformat())
    v1, v2 = _versions(W1), _versions(W2)
    for dt in learner.AGENTS:
        assert v2[dt].status == "published"
        fits1, fits2 = _rows(v1[dt].id, "local_fit"), _rows(v2[dt].id, "local_fit")
        for gid, row in fits2.items():
            ws = row.metadata_json["warm_start"]
            if gid in fits1:
                assert ws == {"source": "previous", "local_fit_id": fits1[gid].id, "study_week_monday": W1.isoformat()}
                assert row.metadata_json["n_new_rows"] == row.sample_size - fits1[gid].sample_size
            else:
                assert ws == {"source": "zero"}
        h1 = next(iter(_rows(v1[dt].id, "hyper").values()))
        h2 = next(iter(_rows(v2[dt].id, "hyper").values()))
        assert h2.metadata_json["previous_hyper_id"] == h1.id
        expected_start = np.maximum(h1.metadata_json["tau2"], h2.metadata_json["floor"])
        np.testing.assert_array_equal(h2.metadata_json["warm_start_tau2"], expected_start)
    assert s2["agents"]["dyad_game"]["n_dyads_fit"] == 6  # d6 has its first REL row now
    # pi_check at the boundary successors uses the AYA policy in force in week 01-26.
    aya_v1 = v1["aya_message"]
    fits2 = _rows(v2["aya_message"].id, "local_fit")
    d6 = fits2["d6"].metadata_json["successor"]
    assert d6["prev_policy_version_id"] == aya_v1.id
    assert d6["pi_check"] == [0.5, 0.5]  # d6 was in week-1 warm-up during 01-26
    d1 = fits2["d1"].metadata_json["successor"]
    post = _rows(aya_v1.id, "posterior")["d1"]
    alpha, cov = np.asarray(post.theta), np.asarray(post.covariance)
    ds = {"names": list(F.NAMES["aya_message"]), "raw": d1["raw"]}
    for a, pi_check in zip((0, 1), d1["pi_check"]):
        _, f = F.hf_from_design_state("aya_message", F.with_game("aya_message", ds, a))
        mu, v = float(f @ alpha), float(f @ cov @ f)
        assert pi_check == agent_probability("aya_message", mu, v)
    assert d1["q_rel"] == pytest.approx(s2["q_rel"]["d1"], abs=0)
    assert s2["q_rel_source"] == "this_run" and s1["q_rel_source"] == "this_run"
    # The REL policy is fitted first; d6 (week 1 at 01-26) had q = 1/2 then.
    assert s1["q_rel"]["d6"] == 0.5


def test_update_summary_is_json_ready(trial_app):
    _, trial = trial_app
    for s in trial.update_summaries:
        assert json.loads(json.dumps(s, allow_nan=False)) == s
    s = trial.update_summaries[-1]
    aya = s["agents"]["aya_message"]
    assert set(aya["l1"]["at_floor"]) == {"act", "mod"}
    assert aya["l2"]["n_dyads"] == aya["n_posteriors"]
    assert set(aya["l3"]["by_dyad"]) == set(CONSENTS)
    assert s["agents"]["dyad_game"]["l3"] is None
    assert set(aya["c3"]["checks"]) == {"1_finite", "2_valid_variance", "3_floor", "4_pooling_no_increase", "5_dimensions"}


def test_fault_injection_in_learned_probability(trial_app):
    app, _ = trial_app
    up = DataUpload.query.filter_by(group_id="d1").order_by(DataUpload.request_timestamp.desc()).first()
    ds = learner.build_design_state("d1", "cp_message", up, DT(2026, 2, 3, 7, 35))
    saved = {k: app.config[k] for k in ("FAULT_INJECTION_ENABLED", "FAULT_ACTION_CRASH", "FAULT_FORCE_PI")}
    try:
        app.config["FAULT_ACTION_CRASH"] = frozenset({"cp_message"})
        with pytest.raises(RuntimeError) as exc:
            learner.learned_probability("d1", "cp_message", ds, DT(2026, 2, 3, 7, 35))
        assert not isinstance(exc.value, learner.NoPolicy)
        app.config["FAULT_ACTION_CRASH"] = frozenset()
        app.config["FAULT_FORCE_PI"] = 0.95
        assert learner.learned_probability("d1", "cp_message", ds, DT(2026, 2, 3, 7, 35)).action_prob == 0.95
        app.config["FAULT_INJECTION_ENABLED"] = False
        app.config["FAULT_ACTION_CRASH"] = frozenset({"cp_message"})
        pi = learner.learned_probability("d1", "cp_message", ds, DT(2026, 2, 3, 7, 35)).action_prob
        assert 0.2 <= pi <= 0.8
    finally:
        app.config.update(saved)


# --------------------------------------------------------- separate trials


def test_rerun_in_the_same_week_supersedes_with_identical_results():
    app = make_app()
    with app.app_context():
        try:
            trial = MiniTrial(CONSENTS, seed=2)
            _run(trial, D(2026, 1, 5), D(2026, 2, 1))
            first = trial.run_update(DT(2026, 2, 2, 3, 0), rid="first")
            cursor = app.sampler.cursor()
            second = learner.run_weekly_update("second", DT(2026, 2, 2, 3, 20), datetime.datetime.now())
            assert app.sampler.cursor() == cursor  # the update draws no random numbers
            assert not second["first_learning_update"] and HyperparameterSet.query.count() == 1
            v1, v2 = _versions(W2, "first"), _versions(W2, "second")
            for dt in learner.AGENTS:
                assert v1[dt].status == v2[dt].status == "published"
                assert v2[dt].id > v1[dt].id
                for kind in ("local_fit", "hyper", "posterior"):
                    r1, r2 = _rows(v1[dt].id, kind), _rows(v2[dt].id, kind)
                    assert set(r1) == set(r2)
                    for gid in r1:
                        assert r1[gid].theta == r2[gid].theta
                        assert r1[gid].covariance == r2[gid].covariance
                h1 = next(iter(_rows(v1[dt].id, "hyper").values())).metadata_json
                h2 = next(iter(_rows(v2[dt].id, "hyper").values())).metadata_json
                for key in ("tau2", "floor", "iterations", "objective", "warm_start_tau2", "previous_hyper_id"):
                    assert h1[key] == h2[key]
                assert first["agents"][dt]["eb"]["tau2"] == second["agents"][dt]["eb"]["tau2"]
            assert first["q_rel"] == second["q_rel"]
            # The rerun supersedes by recency.
            up = DataUpload.query.filter_by(group_id="d2").order_by(DataUpload.request_timestamp.desc()).first()
            ds = learner.build_design_state("d2", "aya_message", up, DT(2026, 2, 2, 7, 30))
            dec = learner.learned_probability("d2", "aya_message", ds, DT(2026, 2, 2, 7, 30))
            assert dec.policy_version_id == v2["aya_message"].id
        finally:
            db.session.remove()
            db.drop_all()


def test_agent_failure_and_c3_rejection_fall_back(monkeypatch):
    from app.learner import update as U

    app = make_app()
    with app.app_context():
        try:
            trial = MiniTrial(CONSENTS, seed=3)
            _run(trial, D(2026, 1, 5), D(2026, 2, 1))
            real_fit, real_check = U._fit_agent, U.check_policy

            def failing_fit(ctx, decision_type, rows):
                if decision_type == "cp_message":
                    raise RuntimeError("boom")
                return real_fit(ctx, decision_type, rows)

            def rejecting_check(decision_type, *args, **kwargs):
                out = real_check(decision_type, *args, **kwargs)
                if decision_type == "aya_message":
                    out = dict(out, passed=False, failures=["3_floor: injected"])
                    out["checks"] = dict(out["checks"], **{"3_floor": False})
                return out

            monkeypatch.setattr(U, "_fit_agent", failing_fit)
            monkeypatch.setattr(U, "check_policy", rejecting_check)
            s = trial.run_update(DT(2026, 2, 2, 3, 0), rid="bad")
            assert s["status"] == "failed"
            assert s["agents"]["dyad_game"]["status"] == "published"
            assert s["agents"]["cp_message"]["status"] == "failed"
            assert "boom" in s["agents"]["cp_message"]["error"]
            assert s["agents"]["aya_message"]["status"] == "rejected"
            assert s["agents"]["aya_message"]["c3"]["passed"] is False
            v = _versions(W2, "bad")
            assert v["cp_message"].status == "failed" and "boom" in v["cp_message"].reason
            assert ModelParameters.query.filter_by(update_rid="bad", decision_type="cp_message").count() == 0
            assert v["aya_message"].status == "rejected" and v["aya_message"].reason.startswith("C3:")
            assert v["aya_message"].published_at_study is None
            # Rejected rows are kept for diagnosis but never used (F-U2).
            assert ModelParameters.query.filter_by(update_rid="bad", decision_type="aya_message").count() > 0
            up = DataUpload.query.filter_by(group_id="d1").order_by(DataUpload.request_timestamp.desc()).first()
            for dt in ("aya_message", "cp_message"):
                ds = learner.build_design_state("d1", dt, up, DT(2026, 2, 2, 7, 30))
                dec = learner.learned_probability("d1", dt, ds, DT(2026, 2, 2, 7, 30))
                assert dec.policy_week == W1  # last valid policy stays in force
        finally:
            db.session.remove()
            db.drop_all()


def test_eb_fallback_end_to_end_with_k_max_one():
    app = make_app({"numerics.k_max": 1})
    with app.app_context():
        try:
            trial = MiniTrial(CONSENTS, seed=4)
            _run(trial, D(2026, 1, 5), D(2026, 1, 25))
            s = trial.run_update(DT(2026, 1, 26, 3, 0))
            for dt in learner.AGENTS:
                agent = s["agents"][dt]
                a = app.hyperparams.agent(dt)
                assert agent["status"] == "published"
                assert agent["eb"]["converged"] is False and agent["eb"]["fallback_used"] == "prior"
                assert agent["eb"]["theta"] == [0.0] * a.p_f
                assert agent["eb"]["tau2"] == [a.tau0_act, a.tau0_mod]
                v = _versions(W1)[dt]
                posts = _rows(v.id, "posterior")
                fits = _rows(v.id, "local_fit")
                assert posts
                # Posteriors are recomputed from this week's fits under the prior (F-U5).
                from app.learner.eb import posterior

                for gid, row in posts.items():
                    fit = fits[gid]
                    th = np.asarray(fit.theta)
                    cov = np.asarray(fit.covariance)
                    mean, pcov, _ = posterior(th[a.p_h:], cov[a.p_h:, a.p_h:], np.zeros(a.p_f), [a.tau0_act, a.tau0_mod], app.hyperparams.eps_ridge)
                    np.testing.assert_allclose(row.theta, mean, rtol=0, atol=1e-15)
                    np.testing.assert_allclose(row.covariance, pcov, rtol=0, atol=1e-15)
        finally:
            db.session.remove()
            db.drop_all()


def test_missing_week_closing_upload_and_agent_without_rows():
    from app.models import Action, StudyData

    app = make_app()
    with app.app_context():
        try:
            trial = MiniTrial(CONSENTS, seed=5)
            _run(trial, D(2026, 1, 5), D(2026, 1, 25))
            # Week-closing uploads for every dyad but d2 (B11 case).
            for g in sorted(CONSENTS):
                if g != "d2":
                    add_upload(g, DT(2026, 1, 26, 2, 0), trial._values(g, D(2026, 1, 26), "am", game_miss=True))
            trial.derive_rewards(DT(2026, 1, 26, 3, 0))
            # No REL decision has a reward: the REL agent has no usable rows.
            StudyData.query.filter_by(decision_type="dyad_game").delete()
            db.session.commit()
            s = learner.run_weekly_update("wk", DT(2026, 1, 26, 3, 0), datetime.datetime.now())
            assert s["agents"]["dyad_game"]["status"] == "skipped"
            assert s["agents"]["dyad_game"]["reason"] == "no_data"
            assert s["agents"]["aya_message"]["status"] == "published"
            assert s["q_rel_source"] == "default" and set(s["q_rel"].values()) == {0.5}
            v = _versions(W1)
            assert "dyad_game" not in v
            fits = _rows(v["aya_message"].id, "local_fit")
            # d2's Sunday PM outcome never arrived: its last usable row is the
            # Sunday AM decision, whose successor is the same-week PM upload.
            d2 = fits["d2"].metadata_json
            last = db.session.get(Action, d2["action_ids"][-1])
            assert last.request_timestamp == DT(2026, 1, 25, 7, 30)
            assert d2["successor"]["boundary"] is False
            assert d2["successor"]["upload_timestamp"] == "2026-01-25T19:25:00"
            assert d2["successor"]["weights"] == [1.0] and d2["successor"]["pi_check"] == [0.5]
            # Everyone else closes the week normally with two weighted candidates.
            d1 = fits["d1"].metadata_json["successor"]
            assert d1["boundary"] is True and d1["weights"] == [0.5, 0.5]
            assert s["agents"]["aya_message"]["no_successor_dyads"] == []
        finally:
            db.session.remove()
            db.drop_all()


def test_no_data_update_skips(learner_app):
    for i in range(5):
        add_group(f"g{i}", D(2026, 1, 5))
    s = learner.run_weekly_update("empty", DT(2026, 1, 19, 3, 0), datetime.datetime.now())
    assert s["skip_reason"] == "no_data" and not s["learning"] and s["status"] == "completed"
    assert PolicyVersion.query.count() == 0 and HyperparameterSet.query.count() == 0


# --------------------------------------------------------- policy in force


def _version(dt, week, status="published", published_at_study=None, rid="r"):
    v = PolicyVersion(
        decision_type=dt, study_week_monday=week, update_rid=rid, status=status,
        published_at_study=published_at_study, n_dyads=1,
    )
    db.session.add(v)
    db.session.flush()
    db.session.add(
        ModelParameters(snapshot_type="posterior", group_id="g", decision_type=dt, theta=[0.0, 0.0],
                        covariance=[[0.01, 0.0], [0.0, 0.01]], policy_version_id=v.id, study_week_monday=week)
    )
    db.session.commit()
    return v


def test_policy_in_force_deadline_rules(learner_app):
    prev = _version("dyad_game", D(2026, 1, 19), published_at_study=DT(2026, 1, 19, 3, 0, 5))
    late = _version("dyad_game", D(2026, 1, 26), published_at_study=DT(2026, 1, 26, 6, 0))
    _version("dyad_game", D(2026, 1, 26), status="rejected", published_at_study=None)
    pif = learner.policy_in_force
    # A version published after Monday 05:45 is never activated midweek (F-U2).
    assert pif("dyad_game", DT(2026, 1, 26, 7, 0)).id == prev.id
    assert pif("dyad_game", DT(2026, 1, 30, 7, 0)).id == prev.id
    # It becomes the fallback of the following week if that week has none.
    assert pif("dyad_game", DT(2026, 2, 2, 7, 0)).id == late.id
    early = _version("dyad_game", D(2026, 1, 26), published_at_study=DT(2026, 1, 26, 4, 0))
    assert pif("dyad_game", DT(2026, 1, 26, 3, 59)).id == prev.id
    assert pif("dyad_game", DT(2026, 1, 26, 4, 0)).id == early.id
    rerun = _version("dyad_game", D(2026, 1, 26), published_at_study=DT(2026, 1, 26, 5, 0))
    assert pif("dyad_game", DT(2026, 1, 26, 4, 30)).id == early.id
    assert pif("dyad_game", DT(2026, 1, 27, 7, 0)).id == rerun.id
    assert pif("dyad_game", DT(2026, 1, 12, 7, 0)) is None
    assert pif("aya_message", DT(2026, 1, 27, 7, 0)) is None


# ------------------------------------------------------------------ adapter


def test_adapter_draws_one_uniform_and_builds_state(learner_app):
    algo = learner_app.rl_algorithm
    start = learner_app.sampler.cursor()
    action, prob, rs = algo.get_action("g", None, {"action_prob": 0.3}, "aya_message", 0)
    end = learner_app.sampler.cursor()
    assert end["uniform"] == start["uniform"] + 1 and end["normal"] == start["normal"]
    assert rs["sampler_cursor_start"] == start and rs["sampler_cursor_end"] == end
    assert action == int(rs["u"] < 0.3) and prob == 0.3
    add_group("g", D(2026, 1, 5))
    add_upload("g", DT(2026, 1, 12, 7, 25), snapshot(aya_app_engagement=4))
    ok, g = algo.make_state({"group_id": "g", "decision_type": "aya_message"})
    assert ok and len(g) == 8 and g[1] == 1.0
    ok, msg = algo.make_state({"group_id": "nobody", "decision_type": "aya_message"})
    assert not ok
    ok, r = algo.make_reward("g", None, 1, {"decision_type": "cp_message", "data": snapshot(daily_diary_completed=True, cp_diary_mood=5)})
    assert ok and r == 1.0
