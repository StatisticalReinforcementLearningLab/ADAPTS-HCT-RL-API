"""
Reward derivation at ``/update`` time (API-Spec §2.4, §5.4, §7.1;
main_new.tex §5.3; contract D6), against the real ``app.learner`` pairing and
reward functions:

- AYA: first later upload with the opposite slot; level 2/1/0; reward = level/2;
- CP: first ``am`` upload on a later date; ``cp_diary_mood`` if the diary was
  completed, else 0; reward = native/5;
- REL: first upload of a later study week (the week-closing upload);
  completion-weighted relationship quality (1/3, 2/3) plus ``w_adh`` times the
  week's mean AYA level; reward = native/(5 + 2 w_adh).

Also: the stored outcome record, the no-state skip, idempotent re-runs, the
upload cutoff and per-action failures.
"""

from __future__ import annotations

import datetime

import pytest

from app import learner
from app.extensions import db
from app.models import DataUpload, ModelUpdateRequests, StudyData
from app.reward_derivation import derive_study_data
from tests.conftest import assert_envelope, post_action, register_group, upload

GID = "dyad_r"


def _up(client, ts, **overrides):
    response = upload(client, GID, ts, **overrides)
    assert response.status_code == 201, response.get_json()
    return response.get_json()["rid"]


def _act(client, decision_type, idx, ts):
    return assert_envelope(post_action(client, GID, decision_type, idx, ts), 201)


def _sd(decision_type, idx) -> StudyData | None:
    return StudyData.query.filter_by(group_id=GID, decision_type=decision_type, decision_idx=idx).one_or_none()


@pytest.fixture
def dyad(client):
    assert register_group(client, GID).status_code == 201  # consent Monday 2026-01-05
    return client


def _week_one(client, *, closing=None, rel_closing=True):
    """
    A study week 1 with one REL, two AYA and one CP decision, closed by the
    Monday 2026-01-12 week-closing upload (``closing`` overrides its fields).
    Returns the rids of the outcome uploads.
    """
    rids = {}
    _up(client, "2026-01-05T05:55:00", current_game_on="miss")
    _act(client, "dyad_game", 1, "2026-01-05T06:00:00")
    _up(client, "2026-01-06T07:25:00", current_game_on=1)
    _act(client, "aya_message", 0, "2026-01-06T07:30:00")
    _up(client, "2026-01-06T07:33:00", current_game_on=1)
    _act(client, "cp_message", 0, "2026-01-06T07:35:00")
    rids["tue_pm"] = _up(client, "2026-01-06T19:25:00", current_game_on=1, previous_med_adherence=1)
    rids["wed_am"] = _up(client, "2026-01-07T07:25:00", current_game_on=1, previous_med_adherence=1,
                         daily_diary_completed=True, cp_diary_mood=4)
    _up(client, "2026-01-11T19:25:00", current_game_on=1, previous_med_adherence=1)
    _act(client, "aya_message", 1, "2026-01-11T19:30:00")
    if rel_closing:
        fields = dict(previous_med_adherence=0, relationship_quality_aya=4, relationship_quality_cp=3,
                      current_game_on="miss", weekly_survey_completed=True, weekly_relationship_score=3.5)
        fields.update(closing or {})
        rids["closing"] = _up(client, "2026-01-12T01:00:00", **fields)
    return rids


# ---------------------------------------------------------- each agent


def test_aya_rewards_pair_with_the_opposite_slot(dyad):
    rids = _week_one(dyad)
    derive_study_data()
    am = _sd("aya_message", 0)
    assert am.reward == pytest.approx(1.0)  # adherence 1 at the same evening's pm upload
    assert am.outcome["reward_native"] == 2.0
    assert am.outcome["outcome_upload_rid"] == rids["tue_pm"]
    assert am.outcome["fields"] == {"slot": "pm", "previous_med_adherence": 1}
    # Sunday PM decision -> the Monday week-closing upload (adherence 0 -> level 1).
    pm = _sd("aya_message", 1)
    assert pm.reward == pytest.approx(0.5)
    assert pm.outcome["reward_native"] == 1.0
    assert pm.outcome["outcome_upload_rid"] == rids["closing"]


def test_cp_reward_is_the_next_mornings_diary(dyad):
    rids = _week_one(dyad)
    derive_study_data()
    cp = _sd("cp_message", 0)
    assert cp.reward == pytest.approx(0.8)
    assert cp.outcome["reward_native"] == 4.0
    assert cp.outcome["outcome_upload_rid"] == rids["wed_am"]  # not the same day's pm upload
    assert cp.outcome["fields"] == {"slot": "am", "daily_diary_completed": True, "cp_diary_mood": 4}


@pytest.mark.parametrize(
    "completed, mood, native",
    [(True, 5, 5.0), (True, 1, 1.0), (False, 4, 0.0), (True, "miss", 0.0), (True, None, 0.0)],
)
def test_cp_reward_values(dyad, completed, mood, native):
    _up(dyad, "2026-01-06T07:33:00")
    _act(dyad, "cp_message", 0, "2026-01-06T07:35:00")
    _up(dyad, "2026-01-07T07:25:00", daily_diary_completed=completed, cp_diary_mood=mood)
    derive_study_data()
    cp = _sd("cp_message", 0)
    assert cp.outcome["reward_native"] == native
    assert cp.reward == pytest.approx(native / 5.0)


def test_rel_reward_is_read_from_the_week_closing_upload(dyad):
    rids = _week_one(dyad)
    derive_study_data()
    rel = _sd("dyad_game", 1)
    native = (4 / 3 + 2 * 3 / 3) / 1.0  # (1/3 * 4 + 2/3 * 3) / (1/3 + 2/3)
    assert rel.outcome["reward_native"] == pytest.approx(native)
    assert rel.reward == pytest.approx(native / 5.0)
    assert rel.outcome["outcome_upload_rid"] == rids["closing"]
    assert rel.outcome["fields"]["week_in_study"] == 2
    assert "aya_week_mean_level" not in rel.outcome  # w_adh = 0


@pytest.mark.parametrize(
    "q_aya, q_cp, native",
    [(4, "miss", 4.0), ("miss", 2.5, 2.5), ("miss", "miss", 0.0), (None, 3, 3.0), (5, 1, (5 / 3 + 2 / 3))],
)
def test_rel_reward_completion_weights(dyad, q_aya, q_cp, native):
    _week_one(dyad, closing={"relationship_quality_aya": q_aya, "relationship_quality_cp": q_cp})
    derive_study_data()
    rel = _sd("dyad_game", 1)
    assert rel.outcome["reward_native"] == pytest.approx(native)
    assert rel.reward == pytest.approx(native / 5.0)


def test_rel_reward_adds_the_weeks_mean_aya_level(make_app):
    app = make_app(HYPERPARAMS_OVERRIDES={"rel_reward.w_adh": 0.5, "noise.sigma_sq.rel": 0.078})
    client = app.test_client()
    register_group(client, GID)
    _week_one(client)
    derive_study_data()
    # Week 1 AYA decisions: Tuesday AM (level 2), Sunday PM (level 1) -> mean 1.5.
    rel = _sd("dyad_game", 1)
    assert rel.outcome["aya_week_mean_level"] == pytest.approx(1.5)
    assert rel.outcome["aya_week_mean_level_n"] == 2
    native = 10 / 3 + 0.5 * 1.5
    assert rel.outcome["reward_native"] == pytest.approx(native)
    assert rel.reward == pytest.approx(native / (5 + 2 * 0.5))


def test_week_closing_upload_feeds_the_update(dyad):
    """End to end through /update with the real learner (cohort warm-up: no fit, F-U3)."""
    rids = _week_one(dyad)
    out = assert_envelope(dyad.post("/api/v1/update", json={"timestamp": "2026-01-12T03:00:00"}), 202)
    row = ModelUpdateRequests.query.filter_by(rid=out["rid"]).one()
    assert row.status == "completed", row.error_message
    assert {(r.decision_type, r.decision_idx) for r in StudyData.query.all()} == {
        ("dyad_game", 1), ("aya_message", 0), ("aya_message", 1), ("cp_message", 0)
    }
    assert _sd("dyad_game", 1).outcome["outcome_upload_rid"] == rids["closing"]
    assert _sd("aya_message", 1).outcome["outcome_upload_rid"] == rids["closing"]


def test_missing_week_closing_upload_defers_the_rel_and_sunday_outcomes(dyad):
    _week_one(dyad, rel_closing=False)
    derive_study_data()
    assert _sd("dyad_game", 1) is None
    assert _sd("aya_message", 1) is None
    assert _sd("aya_message", 0).reward == pytest.approx(1.0)


# ------------------------------------------------------- record keeping


def test_rows_echo_the_action(dyad, app):
    _week_one(dyad)
    derive_study_data()
    from app.models import Action

    for sd in StudyData.query.all():
        action = Action.query.filter_by(group_id=GID, decision_type=sd.decision_type,
                                        decision_idx=sd.decision_idx).one()
        assert sd.action == action.action
        assert sd.action_prob == action.action_prob
        assert sd.state == action.state  # null on warm-up rows
        assert sd.raw_context == action.raw_context
        assert sd.request_timestamp == action.request_timestamp
        assert sd.derived_at is not None
        assert "design_state" in sd.raw_context


def test_no_state_decisions_are_skipped(dyad):
    _act(dyad, "aya_message", 0, "2026-01-06T07:30:00")  # before any upload -> no_state
    _up(dyad, "2026-01-06T19:25:00", previous_med_adherence=1)
    assert derive_study_data() == 0
    assert StudyData.query.count() == 0


def test_rerun_is_idempotent_and_picks_up_late_outcomes(dyad):
    _up(dyad, "2026-01-06T07:33:00")
    _act(dyad, "cp_message", 0, "2026-01-06T07:35:00")
    assert derive_study_data() == 0  # outcome window still open
    assert StudyData.query.count() == 0
    _up(dyad, "2026-01-07T07:25:00", daily_diary_completed=True, cp_diary_mood=3)
    assert derive_study_data() == 1
    first = _sd("cp_message", 0)
    snapshot = (first.reward, dict(first.outcome), first.derived_at)
    _up(dyad, "2026-01-08T07:25:00", daily_diary_completed=True, cp_diary_mood=5)
    assert derive_study_data() == 0
    again = _sd("cp_message", 0)
    assert (again.reward, again.outcome, again.derived_at) == snapshot
    assert StudyData.query.count() == 1


def test_uploads_received_after_the_cutoff_are_not_used(dyad):
    _up(dyad, "2026-01-06T07:33:00")
    _act(dyad, "cp_message", 0, "2026-01-06T07:35:00")
    rid = _up(dyad, "2026-01-07T07:25:00", daily_diary_completed=True, cp_diary_mood=3)
    row = DataUpload.query.filter_by(rid=rid).one()
    cutoff = row.created_at
    row.created_at = cutoff + datetime.timedelta(seconds=5)
    db.session.commit()
    assert derive_study_data(cutoff=cutoff) == 0
    assert derive_study_data(cutoff=cutoff + datetime.timedelta(seconds=10)) == 1


def test_one_failing_action_does_not_block_the_others(dyad, monkeypatch):
    _week_one(dyad)
    real = learner.compute_reward

    def flaky(decision_type, data, **kwargs):
        if decision_type == "cp_message":
            raise ValueError("bad diary")
        return real(decision_type, data, **kwargs)

    monkeypatch.setattr(learner, "compute_reward", flaky)
    assert derive_study_data() == 3
    cp = _sd("cp_message", 0)
    assert cp.reward is None
    assert cp.outcome["error"].startswith("ValueError")
    assert _sd("aya_message", 0).reward == pytest.approx(1.0)

    monkeypatch.setattr(learner, "compute_reward", real)
    assert derive_study_data() == 1
    assert _sd("cp_message", 0).reward == pytest.approx(0.8)
