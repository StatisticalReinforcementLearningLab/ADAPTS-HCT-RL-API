"""
WS-M tests of the decision checks (Monitoring-Spec §5.A): for every check one
case where it fires and one where it does not, on a temporary database built
through the ORM (Monitoring-Spec §11.3 item 1).
"""

from __future__ import annotations

import datetime
import itertools

import numpy as np
import pytest

from app.extensions import db
from app.monitoring import check_action_probability
from app.monitoring.checks import decisions
from tests.test_monitoring_support import (  # noqa: F401 - fixtures
    AYA,
    CP,
    MON1,
    REL,
    D,
    add_action,
    add_dyad,
    add_upload,
    add_version,
    add_model_row,
    alarms,
    app_factory,
    at,
    ctx,
    design_state,
    draw,
    keys,
    mon_app,
)

TUE1 = D(2026, 1, 6)
MON2 = D(2026, 1, 12)
MON3 = D(2026, 1, 19)


# --------------------------------------------------------------------- A13


def test_a13_inline_rejects_impossible_learned_pi_and_raises_red(mon_app):
    context = {"group_id": "d1", "decision_type": AYA, "decision_idx": 4, "rid": "r-a13",
               "request_timestamp": "2026-01-13T07:30:00"}
    assert check_action_probability(0.95, learned=True, context=context) is False
    assert check_action_probability(float("nan"), learned=True, context={**context, "rid": "r-nan"}) is False
    rows = alarms("A13")
    assert [a.rid for a in rows] == ["r-a13", "r-nan"]
    assert rows[0].severity == "red" and rows[0].source == "inside_action" and rows[0].value == 0.95
    assert rows[0].raised_at == at(D(2026, 1, 13), 7, 30)


def test_a13_inline_accepts_valid_pi(mon_app):
    hp = mon_app.hyperparams
    for pi in (hp.l_min, 0.5, hp.l_max, hp.l_max + 1e-13):
        assert check_action_probability(pi, learned=True, context={"rid": f"ok-{pi}"}) is True
    assert check_action_probability(0.5, learned=False, context={"rid": "w"}) is True
    assert alarms("A13") == []


def test_a13_scheduled_finds_stored_impossible_pi_once(mon_app):
    add_dyad("d1", MON1)
    add_action("d1", AYA, 0, at(TUE1, 7, 30), pi=0.4, warmup=True, reason="week1", rid="warm-bad")
    add_action("d1", AYA, 1, at(TUE1, 19, 30), pi=0.5, warmup=True, reason="invalid_probability",
               raw_context={"rejected_policy": {"pi": 0.95}}, rid="rejected")
    add_action("d1", AYA, 2, at(D(2026, 1, 7), 7, 30), pi=0.5, warmup=True, reason="week1", rid="fine")
    add_action("d1", CP, 0, at(TUE1, 7, 35), pi=0.85, warmup=False, rid="learned-bad")
    found = decisions.check_a13(ctx(at(D(2026, 1, 8)), at(MON1)))
    assert sorted(keys(found)) == ["A13:learned-bad", "A13:rejected", "A13:warm-bad"]
    assert check_action_probability(0.95, learned=True, context={"rid": "rejected"}) is False
    from app.monitoring.alarms import emit

    for f in found:
        emit(f, source="scheduled")
    assert len(alarms("A13")) == 3  # the inline alarm and the stored decision share one dedup key


# ---------------------------------------------------------------------- A4


def _learned_decision(app, gid, idx, ts, *, perturb=0.0, shift=0, flip=False):
    """A learned AYA decision whose pi is computed by the learner's own allocation from a stored posterior."""
    from app.learner import features as F
    from app.learner.policy import probability_at

    version = add_version(AYA, MON2, "upd-a4")
    alpha = [0.05, -0.02, 0.03, 0.01]
    cov = (np.eye(4) * 0.02).tolist()
    post = add_model_row("posterior", AYA, version, group_id=gid, theta=alpha, cov=cov,
                         hyperparams_hash=app.hyperparams.sha256)
    ds = design_state(AYA, [1.0, 3.0, 4.0, 1.0, 2.0, 4.0, 3.0, 2.0], slot="am")
    _, f = F.hf_from_design_state(AYA, ds)
    pi, mu, v = probability_at(AYA, f, np.asarray(alpha), np.asarray(cov), hp=app.hyperparams)
    action, rs = draw(app, pi, shift=shift)
    rs["mode"] = "learned"
    if flip:
        action = 1 - action
    row = add_action(gid, AYA, idx, ts, action=action, pi=pi + perturb, warmup=False, ds=ds, random_state=rs,
                     model_parameters_id=post.id, policy_version_id=version.id, mu=mu, v=v,
                     hyperparams_hash=app.hyperparams.sha256)
    return row, post


def test_a4_reproduces_exact_decisions(mon_app):
    add_dyad("d1", MON1)
    _learned_decision(mon_app, "d1", 0, at(D(2026, 1, 13), 7, 30))
    _, rs = draw(mon_app, 0.5)
    add_action("d1", CP, 0, at(D(2026, 1, 13), 7, 35), pi=0.5, warmup=True, random_state=rs,
               action=int(rs["u"] < 0.5))
    assert decisions.check_a4(ctx(at(D(2026, 1, 14), 1, 0))) == []


@pytest.mark.parametrize("fault,kind", [
    ("perturb", "pi_mismatch"),
    ("shift", "uniform_mismatch"),
    ("flip", "action_mismatch"),
    ("posterior_changed", "pi_mismatch"),
    ("two_uniforms", "uniform_count"),
])
def test_a4_fires_on_each_kind_of_mismatch(mon_app, fault, kind):
    add_dyad("d1", MON1)
    row, post = _learned_decision(
        mon_app, "d1", 0, at(D(2026, 1, 13), 7, 30),
        perturb=1e-9 if fault == "perturb" else 0.0, shift=1 if fault == "shift" else 0, flip=fault == "flip",
    )
    if fault == "posterior_changed":
        post.theta = [0.9, -0.02, 0.03, 0.01]
        db.session.commit()
    if fault == "two_uniforms":
        rs = dict(row.random_state)
        rs["sampler_cursor_end"] = {"normal": rs["sampler_cursor_end"]["normal"],
                                    "uniform": rs["sampler_cursor_end"]["uniform"] + 1}
        row.random_state = rs
        db.session.commit()
    found = decisions.check_a4(ctx(at(D(2026, 1, 14), 1, 0)))
    assert len(found) == 1 and found[0].dedup_key == "A4:2026-01-13:aya_message"
    assert kind in found[0].detail["kinds"]
    assert found[0].detail["mismatches"][0]["rid"] == row.rid


def test_a4_detects_a_reused_uniform(mon_app):
    add_dyad("d1", MON1)
    _, rs = draw(mon_app, 0.5)
    add_action("d1", AYA, 0, at(D(2026, 1, 13), 7, 30), pi=0.5, action=int(rs["u"] < 0.5), random_state=rs)
    add_action("d1", AYA, 1, at(D(2026, 1, 13), 19, 30), pi=0.5, action=int(rs["u"] < 0.5), random_state=dict(rs))
    found = decisions.check_a4(ctx(at(D(2026, 1, 14), 1, 0)))
    assert found and found[0].detail["kinds"].get("uniform_reused") == 2


# --------------------------------------------------------------------- A10


def test_a10_fires_when_a_learned_pi_is_used_during_cohort_warmup(mon_app):
    add_dyad("d1", MON1)
    add_action("d1", AYA, 0, at(D(2026, 1, 20), 7, 30), pi=0.6, warmup=False)  # week 3, cohort of 1 < n_warm
    add_action("d1", CP, 0, at(D(2026, 1, 20), 7, 35), pi=0.5, warmup=True, reason="no_policy")  # operational
    found = decisions.check_a10(ctx(at(D(2026, 1, 21), 1, 0)))
    assert keys(found) == ["A10:2026-01-20:aya_message"]
    assert found[0].detail["mismatches"][0]["kind"] == "learned_in_warmup"
    assert found[0].detail["mismatches"][0]["expected"] == "cohort"


def test_a10_quiet_when_rules_followed(app_factory):
    app = app_factory(HYPERPARAMS_OVERRIDES={"warmup.n_warm": 2})
    add_dyad("d1", MON1)
    add_dyad("d2", MON1)
    add_action("d1", AYA, 0, at(D(2026, 1, 20), 7, 30), pi=0.6, warmup=False)  # week 3: learned is right
    add_action("d2", AYA, 0, at(D(2026, 1, 20), 7, 30), pi=0.5, warmup=True, reason="no_policy")
    add_dyad("d3", D(2026, 1, 14))  # partial week then week 1 on 01-19
    add_action("d3", AYA, 0, at(D(2026, 1, 20), 7, 30), pi=0.5, warmup=True, reason="week1")
    assert app.hyperparams.n_warm == 2
    assert decisions.check_a10(ctx(at(D(2026, 1, 21), 1, 0))) == []


def test_a10_uses_consent_dates_as_of_the_decision(mon_app):
    """A later re-registration does not change what the rule said at decision time (Monitoring-Spec §9.6)."""
    add_dyad("d1", D(2026, 1, 14))
    add_action("d1", AYA, 0, at(D(2026, 1, 20), 7, 30), pi=0.5, warmup=True, reason="cohort")
    add_dyad("d1", MON1)  # consent window moved after the decision
    assert decisions.check_a10(ctx(at(D(2026, 1, 21), 1, 0))) == []


# ----------------------------------------------------------------- A6 / A7


def _closing_week(aya, cp):
    add_dyad("d1", MON1)
    add_upload("d1", at(MON3, 2, 0), aya_messages_delivered_week=aya, cp_messages_delivered_week=cp)


def test_a6_a7_fire_above_80_and_below_20_percent(mon_app):
    _closing_week(12, 1)
    a6 = decisions.check_a6(ctx(at(MON3, 4, 0)))
    a7 = decisions.check_a7(ctx(at(MON3, 4, 0)))
    assert keys(a6) == ["A6:2026-01-12:d1:aya_message"] and a6[0].value == pytest.approx(12 / 14)
    assert keys(a7) == ["A7:2026-01-12:d1:cp_message"] and a7[0].threshold == 0.20


def test_a6_a7_quiet_inside_the_limits_and_in_the_partial_week(mon_app):
    _closing_week(11, 2)
    assert decisions.check_a6(ctx(at(MON3, 4, 0))) == []
    assert decisions.check_a7(ctx(at(MON3, 4, 0))) == []
    add_dyad("p1", D(2026, 1, 14))  # partial week 01-14..01-18: excluded
    add_upload("p1", at(MON3, 2, 0), aya_messages_delivered_week=14, cp_messages_delivered_week=0)
    assert decisions.check_a6(ctx(at(MON3, 4, 0))) == []


# --------------------------------------------------------------------- A16


def test_a16_fires_on_delivery_mismatches(mon_app):
    add_dyad("d1", MON1)
    add_action("d1", AYA, 0, at(MON2, 7, 30), action=1, ds=design_state(AYA, slot="am"))
    add_action("d1", CP, 0, at(MON2, 7, 35), action=0)
    add_upload("d1", at(MON2, 19, 25), {"slot": "pm", "prompted_by_message": False, "cp_prompted_by_message": True})
    found = decisions.check_a16(ctx(at(D(2026, 1, 14), 7, 0)))
    assert keys(found) == ["A16:2026-01-14"]
    assert found[0].detail["kinds"] == {"action_1_not_delivered": 1, "action_0_delivered": 1}


def test_a16_week_closing_counts_against_sends(mon_app):
    add_dyad("d1", MON1)
    for k in range(3):
        add_action("d1", AYA, k, at(D(2026, 1, 6 + k), 7, 30), action=1)
    add_upload("d1", at(MON2, 2, 0), aya_messages_delivered_week=5, cp_messages_delivered_week=0)
    found = decisions.check_a16(ctx(at(D(2026, 1, 13), 7, 0)))
    assert found and found[0].detail["kinds"] == {"deliveries_exceed_sends": 1}


def test_a16_quiet_when_deliveries_match(mon_app):
    add_dyad("d1", MON1)
    add_action("d1", AYA, 0, at(MON2, 7, 30), action=1, ds=design_state(AYA, slot="am"))
    add_action("d1", CP, 0, at(MON2, 7, 35), action=0)
    add_upload("d1", at(MON2, 19, 25), {"slot": "pm", "prompted_by_message": True, "cp_prompted_by_message": False})
    assert decisions.check_a16(ctx(at(D(2026, 1, 14), 7, 0))) == []


# --------------------------------------------------------------------- A17


def test_a17_fires_on_three_equal_weekly_game_actions(mon_app):
    add_dyad("d1", MON1)
    rows = [add_action("d1", REL, w, at(MON1 + datetime.timedelta(days=7 * w), 6, 0), action=1) for w in range(3)]
    assert decisions.a17_for_action(rows[1]) == []
    found = decisions.a17_for_action(rows[2])
    assert keys(found) == [f"A17:{rows[2].rid}"] and found[0].detail["game"] == 1


def test_a17_quiet_when_the_game_varies(mon_app):
    add_dyad("d1", MON1)
    for w, a in enumerate((1, 0, 1)):
        add_action("d1", REL, w, at(MON1 + datetime.timedelta(days=7 * w), 6, 0), action=a)
    assert decisions.check_a17(ctx(at(MON3, 7, 0), at(MON1))) == []


# ---------------------------------------------------------------------- A2


_IDX = itertools.count(100)


def _full_day(gid, day, *, aya=2, cp=1, rel=True):
    for k in range(aya):
        add_action(gid, AYA, next(_IDX), at(day, 7 + 12 * k, 30))
    for _ in range(cp):
        add_action(gid, CP, next(_IDX), at(day, 7, 35))
    if rel and day.weekday() == 0:
        add_action(gid, REL, next(_IDX), at(day, 6, 0))


def test_a2_fires_on_missing_and_extra_calls(mon_app):
    add_dyad("d1", MON1)
    add_action("d1", AYA, 0, at(MON2, 7, 30))  # one AYA call of two; no CP; no Monday REL
    add_action("d1", AYA, 1, at(MON2, 19, 30))
    add_action("d1", AYA, 2, at(MON2, 20, 30))
    found = decisions.check_a2(ctx(at(D(2026, 1, 13), 7, 0)))
    by_type = {f.decision_type: f for f in found}
    assert set(by_type) == {AYA, CP, REL}
    assert by_type[AYA].detail["extra_calls"] == 1 and by_type[CP].detail["missing_calls"] == 1
    assert by_type[REL].fallback_used == "F-A1"


def test_a2_quiet_on_the_calendar_including_partial_week(mon_app):
    add_dyad("d1", MON1)
    add_dyad("p1", D(2026, 1, 8))  # Thursday: partial week 01-08..01-11 (AYA, CP), week 1 from Monday 01-12
    friday = D(2026, 1, 9)
    _full_day("d1", friday)
    _full_day("p1", friday)        # partial-week day: AYA 2, CP 1, no REL
    assert decisions.check_a2(ctx(at(D(2026, 1, 10), 7, 0))) == []
    _full_day("d1", MON2)
    _full_day("p1", MON2)          # Monday of week 1: REL expected for both
    assert decisions.check_a2(ctx(at(D(2026, 1, 13), 7, 0))) == []


def test_a2_expects_no_rel_decision_in_the_partial_week(mon_app):
    add_dyad("p1", D(2026, 1, 6))  # Tuesday: no partial-week Monday at all
    add_dyad("p2", D(2026, 1, 5) - datetime.timedelta(days=3))  # Friday 01-02: Monday 01-05 is week 1
    add_action("p2", REL, 0, at(MON1, 6, 0))
    _full_day("p2", MON1, rel=False)
    assert decisions.check_a2(ctx(at(D(2026, 1, 6), 7, 0))) == []
    add_action("p1", REL, 0, at(D(2026, 1, 7), 6, 0))  # a REL call on a Wednesday of the partial week
    _full_day("p1", D(2026, 1, 7), rel=False)
    _full_day("p2", D(2026, 1, 7), rel=False)
    found = decisions.check_a2(ctx(at(D(2026, 1, 8), 7, 0)))
    assert [f.decision_type for f in found] == [REL]
    assert found[0].detail["dyads"] == [{"group_id": "p1", "expected": 0, "actual": 1, "study_week": 0}]


# ---------------------------------------------------------------------- A8


def test_a8_fires_outside_the_consent_window(mon_app):
    add_dyad("d1", MON2, D(2026, 3, 1))
    row = add_action("d1", AYA, 0, at(D(2026, 1, 9), 7, 30))
    found = decisions.a8_for_action(row)
    assert keys(found) == [f"A8:{row.rid}"] and found[0].detail["side"] == "before consent_start_date"
    late = add_action("d1", AYA, 1, at(D(2026, 3, 2), 7, 30))
    assert decisions.a8_for_action(late)[0].detail["side"] == "after consent_end_date"


def test_a8_quiet_inside_the_consent_window(mon_app):
    add_dyad("d1", MON2, D(2026, 3, 1))
    add_action("d1", AYA, 0, at(MON2, 7, 30))
    add_action("d1", AYA, 1, at(D(2026, 3, 1), 19, 30))
    assert decisions.check_a8(ctx(at(D(2026, 3, 2)), at(MON1))) == []


# --------------------------------------------------------------------- A15


def test_a15_fires_when_the_game_status_is_stale(mon_app):
    add_dyad("d1", MON1)
    add_action("d1", REL, 1, at(MON2, 6, 0), action=0)
    ds = design_state(AYA, [1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0])  # game = 1
    row = add_action("d1", AYA, 9, at(MON2, 7, 30), pi=0.6, warmup=False, ds=ds)
    found = decisions.a15_for_action(row)
    assert keys(found) == [f"A15:{row.rid}"]
    assert found[0].detail["game_used"] == 1 and found[0].detail["rel_action"] == 0


def test_a15_quiet_when_matching_imputed_or_warmup(mon_app):
    add_dyad("d1", MON1)
    add_action("d1", REL, 1, at(MON2, 6, 0), action=1)
    good = design_state(AYA, [1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0])
    imputed = design_state(AYA, [1.0, 2.0, 1.0, 0.0, 2.0, 3.0, 3.0, 3.0], imputed={"game": "miss->0"})
    add_action("d1", AYA, 9, at(MON2, 7, 30), pi=0.6, warmup=False, ds=good)
    add_action("d1", CP, 9, at(MON2, 7, 35), pi=0.6, warmup=False, ds=design_state(CP, imputed={"game": "miss->0"}))
    add_action("d1", AYA, 10, at(MON2, 19, 30), pi=0.6, warmup=False, ds=imputed)
    add_action("d1", AYA, 11, at(D(2026, 1, 13), 7, 30), pi=0.5, warmup=True,
               ds=design_state(AYA, [1.0, 2.0, 1.0, 0.0, 2.0, 3.0, 3.0, 3.0]))
    assert decisions.check_a15(ctx(at(D(2026, 1, 14)), at(MON1))) == []


# ---------------------------------------------------------------------- A5


def _a5_days(pi):
    for g in range(5):
        add_dyad(f"d{g}", MON1)
    for day in range(7):
        for g in range(5):
            add_action(f"d{g}", AYA, day, at(D(2026, 1, 12 + day), 7, 30), pi=pi, warmup=False)


def test_a5_fires_when_the_cohort_sits_at_a_limit_for_7_days(mon_app):
    _a5_days(0.795)
    found = decisions.check_a5(ctx(at(D(2026, 1, 19), 7, 0)))
    assert keys(found) == ["A5:aya_message:2026-01-18"]
    assert all(d["qualifies"] for d in found[0].detail["days"])


def test_a5_quiet_when_pi_varies(mon_app):
    _a5_days(0.6)
    assert decisions.check_a5(ctx(at(D(2026, 1, 19), 7, 0))) == []
