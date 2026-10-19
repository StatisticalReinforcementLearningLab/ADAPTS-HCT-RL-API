"""
WS-L tests: design state, standardization, rewards and outcome pairing
(implementation contract D6, D7; main_new.tex §5.3 eqs. (reward-aya),
(reward-cp), (reward-rel-quality), (reward-rel); §5.4 Table 3,
eq. (range-std); API-Spec §2.3, §4.1).
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from app import learner
from app.hyperparams import load_hyperparams
from app.learner import features as F
from app.learner.rewards import compute_reward as compute_reward_hp
from tests.test_learner_support import (  # noqa: F401 - fixture
    add_action,
    add_group,
    add_upload,
    learner_app,
    snapshot,
)

T0 = datetime.datetime(2026, 1, 12, 7, 25)  # a Monday


def _std(x, lo, hi):
    return (x - (hi + lo) / 2.0) / ((hi - lo) / 2.0)


# --------------------------------------------------------------- features


def test_snapshot_helper_matches_schema():
    from app.protocol import SNAPSHOT_SCHEMA

    assert set(snapshot()) == set(SNAPSHOT_SCHEMA)


def test_dimensions_and_hf_split_match_design():
    assert F.dims(F.AYA) == (7, 4)
    assert F.dims(F.CP) == (6, 4)
    assert F.dims(F.REL) == (7, 2)
    assert "burden_aya" not in F.H_NAMES[F.AYA] and "burden_aya" in F.F_NAMES[F.AYA]
    assert "burden_cp" not in F.H_NAMES[F.CP] and "burden_cp" in F.F_NAMES[F.CP]
    assert F.H_NAMES[F.REL] == F.NAMES[F.REL]
    assert F.F_NAMES[F.REL] == ("const", "relq_dyad")


def test_standardize_maps_range_endpoints_and_clips():
    g, clipped = F.standardize(F.AYA, [1.0, 4.0, 14.0, 1.0, 0.0, 5.0, 0.0, 2.5])
    assert g == [1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 0.0]
    assert clipped == []
    g, clipped = F.standardize(F.AYA, [1.0, 1.0, 20.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    assert g[2] == 1.0 and g[1] == -1.0 and g[3] == -1.0
    assert clipped == ["burden_aya"]
    g, clipped = F.standardize(F.REL, [1.0, 5.0, 2.0, 4.0, 12.0, 1.0, 1.0])
    assert clipped == ["burden_avg"] and g[4] == 1.0


def test_standardize_rejects_wrong_length_and_nonfinite():
    with pytest.raises(ValueError):
        F.standardize(F.CP, [1.0, 2.0])
    with pytest.raises(ValueError):
        F.standardize(F.CP, [1.0, 2.0, float("nan"), 0, 0, 0, 0])


def test_aya_design_state_hand_built_snapshot(learner_app):
    add_group("d1", datetime.date(2026, 1, 5))
    up = add_upload(
        "d1",
        T0,
        snapshot(
            aya_app_engagement=3,
            aya_app_burden=2.5,
            current_game_on=1,
            previous_med_adherence=1,
            aya_diary_mood=4,
            aya_diary_physical="miss",
            relationship_quality_aya=3.5,
            slot="am",
            week_in_study=2,
        ),
    )
    ds = learner.build_design_state("d1", F.AYA, up, T0)
    assert ds["names"] == list(F.NAMES[F.AYA])
    assert ds["raw"] == [1.0, 3.0, 2.5, 1.0, 2.0, 4.0, 0.0, 3.5]
    assert ds["upload_id"] == up.id and ds["upload_rid"] == up.rid
    assert ds["slot"] == "am" and ds["week_in_study"] == 2
    assert ds["imputed"] == {"aya_phys": "miss:aya_diary_physical->0"}
    assert ds["clipped"] == []
    g = learner.standardized_g(F.AYA, ds)
    expected = [
        1.0,
        _std(3, 1, 4),
        _std(2.5, 0, 14),
        1.0,
        1.0,
        _std(4, 0, 5),
        -1.0,
        _std(3.5, 0, 5),
    ]
    np.testing.assert_allclose(g, expected, rtol=0, atol=1e-15)
    h, f = F.split_hf(F.AYA, g)
    np.testing.assert_array_equal(h, np.array(g)[[0, 1, 3, 4, 5, 6, 7]])
    np.testing.assert_array_equal(f, np.array(g)[[0, 1, 2, 3]])


def test_cp_design_state_missing_encodings(learner_app):
    add_group("d1", datetime.date(2026, 1, 5))
    up = add_upload(
        "d1",
        T0,
        snapshot(
            cp_app_engagement=4,
            cp_app_burden=7.5,
            current_game_on="miss",
            cp_missing_rate_7d=0.25,
            cp_diary_mood="miss",
            relationship_quality_cp=4.5,
        ),
    )
    ds = learner.build_design_state("d1", F.CP, up, T0)
    assert ds["raw"] == [1.0, 4.0, 7.5, 0.0, 0.25, 0.0, 4.5]
    assert ds["imputed"]["game"] == "miss:current_game_on->0"
    assert ds["imputed"]["cp_mood"] == "miss:cp_diary_mood->0"
    assert ds["clipped"] == ["burden_cp"]
    g = learner.standardized_g(F.CP, ds)
    assert g[2] == 1.0  # clipped
    assert g[3] == -1.0  # game off
    assert g[5] == -1.0  # missing mood is level 0, the bottom of [0, 5]


def test_engagement_locf_and_default(learner_app):
    add_group("d1", datetime.date(2026, 1, 5))
    # No earlier upload: the default level 1.
    first = add_upload("d1", T0, snapshot(aya_app_engagement="miss", cp_app_engagement="miss"))
    ds = learner.build_design_state("d1", F.AYA, first, T0)
    assert ds["raw"][1] == 1.0
    assert ds["imputed"]["eng_aya"].startswith("default:")
    # An earlier non-missing value is carried forward, skipping later misses.
    add_upload("d1", T0 + datetime.timedelta(hours=1), snapshot(aya_app_engagement=3, cp_app_engagement=2))
    add_upload("d1", T0 + datetime.timedelta(hours=2), snapshot(aya_app_engagement="miss", cp_app_engagement=4))
    last = add_upload("d1", T0 + datetime.timedelta(hours=3), snapshot(aya_app_engagement=None, cp_app_engagement="miss"))
    ds = learner.build_design_state("d1", F.AYA, last, T0 + datetime.timedelta(hours=3))
    assert ds["raw"][1] == 3.0
    assert ds["imputed"]["eng_aya"].startswith("locf:aya_app_engagement")
    ds_cp = learner.build_design_state("d1", F.CP, last, T0 + datetime.timedelta(hours=3))
    assert ds_cp["raw"][1] == 4.0
    # Another dyad's uploads are never used.
    add_group("d2", datetime.date(2026, 1, 5))
    other = add_upload("d2", T0 + datetime.timedelta(hours=4), snapshot(aya_app_engagement="miss"))
    assert learner.build_design_state("d2", F.AYA, other, T0)["raw"][1] == 1.0


def test_rel_design_state_with_previous_week_mean_level(learner_app):
    add_group("d1", datetime.date(2026, 1, 5))
    prev_mon = datetime.date(2026, 1, 5)
    # Four AYA decisions in the previous week; three have outcome uploads by
    # decision time (levels 2, 1, 0). The last is a Sunday AM decision whose
    # outcome (a pm upload) never arrives; the Monday 06:00 upload is "am".
    times = [
        datetime.datetime(2026, 1, 6, 7, 30),
        datetime.datetime(2026, 1, 7, 7, 30),
        datetime.datetime(2026, 1, 8, 7, 30),
        datetime.datetime(2026, 1, 11, 7, 30),
    ]
    for i, ts in enumerate(times):
        add_action("d1", F.AYA, i, ts, action=0, action_prob=0.5, design_state=None, slot=None)
    add_upload("d1", datetime.datetime(2026, 1, 6, 19, 25), snapshot(slot="pm", previous_med_adherence=1))
    add_upload("d1", datetime.datetime(2026, 1, 7, 19, 25), snapshot(slot="pm", previous_med_adherence=0))
    add_upload("d1", datetime.datetime(2026, 1, 8, 19, 25), snapshot(slot="pm", previous_med_adherence="miss"))
    decision = datetime.datetime(2026, 1, 12, 6, 0)
    up = add_upload(
        "d1",
        decision,
        snapshot(
            week_in_study=2,
            relationship_quality_aya=3.0,
            relationship_quality_cp=4.5,
            aya_app_engagement=2,
            cp_app_engagement=4,
            aya_app_burden=3.0,
            cp_app_burden=1.0,
            prior_game_action="miss",
            aya_diary_summary=0.6,
            cp_diary_summary="miss",
        ),
    )
    level, n = learner.aya_week_mean_level("d1", prev_mon, decision)
    assert (level, n) == (1.0, 3)
    ds = learner.build_design_state("d1", F.REL, up, decision)
    # relq_dyad = (1/3 * 3 + 2/3 * 4.5) / 1 = 4.0
    np.testing.assert_allclose(ds["raw"], [1.0, 4.0, 1.0, 3.0, 2.0, 0.0, 0.3], rtol=0, atol=1e-12)
    assert ds["aya_week_mean_level_n"] == 3
    assert ds["imputed"]["prior_game"] == "miss:prior_game_action->0"
    assert ds["imputed"]["dsum_cp"] == "miss:cp_diary_summary->0"
    g = learner.standardized_g(F.REL, ds)
    np.testing.assert_allclose(
        g, [1.0, _std(4.0, 0, 5), 0.0, _std(3.0, 1, 4), _std(2.0, 0, 10.5), -1.0, _std(0.3, 0, 1)], atol=1e-12
    )
    # Both reports missing -> relq_dyad = 0; no AYA decisions -> mean level 0.
    add_group("d2", datetime.date(2026, 1, 5))
    up2 = add_upload("d2", decision, snapshot())
    ds2 = learner.build_design_state("d2", F.REL, up2, decision)
    assert ds2["raw"][1] == 0.0 and ds2["raw"][2] == 0.0
    assert "relq_dyad" in ds2["imputed"]


def test_with_game_sets_the_game_coordinate():
    ds = {"names": list(F.NAMES[F.AYA]), "raw": [1, 2, 3, 0, 1, 2, 3, 4], "clipped": []}
    on = F.with_game(F.AYA, ds, 1)
    assert on["raw"][3] == 1.0 and ds["raw"][3] == 0
    with pytest.raises(ValueError):
        F.with_game(F.REL, {"names": list(F.NAMES[F.REL]), "raw": [1] * 7}, 1)


def test_design_state_layout_is_validated():
    with pytest.raises(ValueError):
        learner.standardized_g(F.CP, {"names": list(F.NAMES[F.AYA]), "raw": [1.0] * 8})


# ----------------------------------------------------------------- rewards


def test_aya_level_encoding():
    assert learner.aya_level(1) == 2
    assert learner.aya_level(0) == 1
    assert learner.aya_level("miss") == 0
    assert learner.aya_level(None) == 0
    assert learner.aya_level(True) == 2


def test_compute_reward_every_branch(learner_app):
    assert learner.compute_reward(F.AYA, snapshot(previous_med_adherence=1)) == (1.0, 2.0)
    assert learner.compute_reward(F.AYA, snapshot(previous_med_adherence=0)) == (0.5, 1.0)
    assert learner.compute_reward(F.AYA, snapshot(previous_med_adherence="miss")) == (0.0, 0.0)
    assert learner.compute_reward(F.CP, snapshot(daily_diary_completed=True, cp_diary_mood=4)) == (0.8, 4.0)
    assert learner.compute_reward(F.CP, snapshot(daily_diary_completed=False, cp_diary_mood=4)) == (0.0, 0.0)
    assert learner.compute_reward(F.CP, snapshot(daily_diary_completed=True, cp_diary_mood="miss")) == (0.0, 0.0)
    both = learner.compute_reward(F.REL, snapshot(relationship_quality_aya=3.0, relationship_quality_cp=4.5))
    assert both == pytest.approx((0.8, 4.0), abs=1e-15)
    only_cp = learner.compute_reward(F.REL, snapshot(relationship_quality_aya="miss", relationship_quality_cp=4.5))
    assert only_cp == pytest.approx((0.9, 4.5), abs=1e-15)
    only_aya = learner.compute_reward(F.REL, snapshot(relationship_quality_aya=3.0, relationship_quality_cp=None))
    assert only_aya == pytest.approx((0.6, 3.0), abs=1e-15)
    assert learner.compute_reward(F.REL, snapshot()) == (0.0, 0.0)
    # weekly_survey_completed / weekly_relationship_score are not read.
    assert learner.compute_reward(
        F.REL, snapshot(weekly_survey_completed=True, weekly_relationship_score=5.0)
    ) == (0.0, 0.0)


def test_rel_reward_with_adherence_weight_and_unnormalized():
    hp = load_hyperparams(overrides={"rel_reward.w_adh": 0.5, "noise.sigma_sq.rel": 0.05})
    data = snapshot(relationship_quality_aya=3.0, relationship_quality_cp=4.5)
    norm, native = compute_reward_hp(F.REL, data, aya_week_mean_level=1.5, hp=hp)
    assert native == pytest.approx(4.0 + 0.5 * 1.5)
    assert norm == pytest.approx(native / (5 + 2 * 0.5))
    with pytest.raises(ValueError):
        compute_reward_hp(F.REL, data, hp=hp)
    raw_hp = load_hyperparams(
        overrides={"rewards.normalize": False, "noise.sigma_sq": {"aya": 0.5, "cp": 2.0, "rel": 1.5}}
    )
    assert compute_reward_hp(F.CP, snapshot(daily_diary_completed=True, cp_diary_mood=4), hp=raw_hp) == (4.0, 4.0)


# ----------------------------------------------------------------- pairing


def _up(ts, slot, week=1):
    return {"request_timestamp": ts, "data": {"slot": slot, "week_in_study": week}}


def test_pairing_aya_am_to_pm_and_pm_to_next_am_including_week_closing():
    sun = datetime.date(2026, 1, 18)
    mon = datetime.date(2026, 1, 19)
    at = lambda d, h, m: datetime.datetime.combine(d, datetime.time(h, m))  # noqa: E731
    timeline = [
        _up(at(sun, 7, 25), "am"),
        _up(at(sun, 7, 33), "am"),
        _up(at(sun, 19, 25), "pm"),
        _up(at(mon, 2, 0), "am", week=2),  # week-closing upload
        _up(at(mon, 6, 0), "am", week=2),
        _up(at(mon, 7, 25), "am", week=2),
    ]
    am = learner.find_outcome_upload(F.AYA, "am", at(sun, 7, 30), None, timeline)
    assert am is timeline[2]  # skips the pre-CP am upload
    pm = learner.find_outcome_upload(F.AYA, "pm", at(sun, 19, 30), None, timeline)
    assert pm is timeline[3]  # the Monday week-closing upload
    # Slot derived from the time when not given.
    assert learner.find_outcome_upload(F.AYA, None, at(sun, 19, 30), None, timeline) is timeline[3]
    # Outcome not arrived yet.
    assert learner.find_outcome_upload(F.AYA, "pm", at(sun, 19, 30), None, timeline[:3]) is None


def test_pairing_cp_next_morning_and_rel_next_week():
    d = datetime.date(2026, 1, 14)
    at = lambda day, h, m: datetime.datetime.combine(day, datetime.time(h, m))  # noqa: E731
    nxt = d + datetime.timedelta(days=1)
    timeline = [
        _up(at(d, 7, 33), "am"),  # before the decision: skipped
        _up(at(d, 19, 25), "pm"),
        _up(at(nxt, 7, 25), "am"),
        _up(at(nxt, 7, 33), "am"),
    ]
    assert learner.find_outcome_upload(F.CP, "am", at(d, 7, 35), None, timeline) is timeline[2]
    mon = datetime.date(2026, 1, 12)
    rel_timeline = [
        _up(at(mon, 7, 25), "am", week=2),
        _up(at(mon + datetime.timedelta(days=6), 19, 25), "pm", week=2),
        _up(at(mon + datetime.timedelta(days=7), 2, 0), "am", week=3),
    ]
    assert learner.find_outcome_upload(F.REL, None, at(mon, 6, 0), 2, rel_timeline) is rel_timeline[2]
    assert learner.find_outcome_upload(F.REL, None, at(mon, 6, 0), 2, rel_timeline[:2]) is None
    with pytest.raises(ValueError):
        learner.find_outcome_upload(F.REL, None, at(mon, 6, 0), None, rel_timeline)


def test_pairing_accepts_orm_rows_and_action_slot(learner_app):
    add_group("d1", datetime.date(2026, 1, 5))
    up_am = add_upload("d1", datetime.datetime(2026, 1, 13, 7, 25), snapshot(slot="am"))
    up_pm = add_upload("d1", datetime.datetime(2026, 1, 13, 19, 25), snapshot(slot="pm"))
    act = add_action(
        "d1", F.AYA, 0, datetime.datetime(2026, 1, 13, 7, 30), action=1, action_prob=0.5,
        design_state={"slot": "am", "names": [], "raw": []},
    )
    assert learner.action_slot(act) == "am"
    assert learner.find_outcome_upload(F.AYA, learner.action_slot(act), act.request_timestamp, None, [up_am, up_pm]) is up_pm
    # A no-state fallback has no snapshot: the slot comes from the time of day.
    bare = add_action("d1", F.AYA, 1, datetime.datetime(2026, 1, 13, 19, 30), action=0, action_prob=0.5, design_state=None)
    assert learner.action_slot(bare) == "pm"
