"""
Server-side scheduled warm-up in /action (API-Spec §2.2; main_new.tex
Algorithm 3 line 3, §5.2; contract D2, D3):

- cohort rule (D2, checked first): a decision in calendar week W is a coin flip
  while fewer than ``n_warm`` dyads have *completed* week 1, i.e. have
  ``week1_start <= Monday(W) - 7 days``;
- per-dyad rule (D3): the decision falls in the dyad's partial week (study
  week 0) or in study weeks ``1..t_warm``.

Warm-up rows have ``action_prob = 0.5``, ``warmup = true``, ``state = null``,
``model_param = null`` and the reason on the actions row only.
"""

from __future__ import annotations

from app.models import Action
from tests.conftest import assert_envelope, install_learned_policy, post_action, register_group, upload


def _reason(rid):
    return Action.query.filter_by(rid=rid).one().warmup_reason


def _decide(client, gid, idx, ts, decision_type="aya_message"):
    upload(client, gid, ts.replace(":30:00", ":25:00"))
    return assert_envelope(post_action(client, gid, decision_type, idx, ts), 201)


def test_cohort_rule_warms_up_the_first_dyads(client):
    register_group(client, "dyad_001")
    out = _decide(client, "dyad_001", 0, "2026-01-06T07:30:00")
    assert out["warmup"] is True
    assert out["action_prob"] == 0.5
    assert out["state"] is None and out["model_param"] is None
    assert "warmup_reason" not in out
    assert _reason(out["rid"]) == "cohort"


def test_cohort_rule_counts_dyads_that_completed_week1_not_started(client, monkeypatch):
    # Five dyads start week 1 on Monday 2026-01-12. Under D2 the cohort rule
    # still holds during that week (|E_{W-1}| = 0 < 5) even though five dyads
    # have *started* week 1 (the older API-Spec wording); it lifts on 2026-01-19.
    install_learned_policy(monkeypatch, pi=0.6)
    for i in range(5):
        register_group(client, f"dyad_{i:03d}", start="2026-01-12")
    in_week1 = _decide(client, "dyad_000", 0, "2026-01-13T07:30:00")
    assert _reason(in_week1["rid"]) == "cohort"
    in_week2 = _decide(client, "dyad_000", 1, "2026-01-20T07:30:00")
    assert in_week2["warmup"] is False
    assert _reason(in_week2["rid"]) is None


def test_week1_rule_applies_once_the_cohort_is_filled(client):
    # Five dyads completed week 1 by 2026-01-12; a dyad consenting Thursday
    # 2026-01-08 is in study week 1 from Monday 2026-01-12.
    for i in range(5):
        register_group(client, f"dyad_{i:03d}")
    register_group(client, "dyad_thu", start="2026-01-08")
    out = _decide(client, "dyad_thu", 0, "2026-01-13T07:30:00")
    assert out["warmup"] is True
    assert _reason(out["rid"]) == "week1"


def test_week1_rule_covers_the_partial_week(client):
    # Cohort filled by Monday 2026-01-12 (five 2025-12-29 starters completed
    # week 1 on 2026-01-04); a Thursday 2026-01-15 consent has its partial week
    # 2026-01-15 .. 2026-01-18 (study week 0) -> week1 reason.
    for i in range(5):
        register_group(client, f"dyad_{i:03d}", start="2025-12-29")
    register_group(client, "dyad_thu", start="2026-01-15")
    out = _decide(client, "dyad_thu", 0, "2026-01-16T07:30:00")
    assert _reason(out["rid"]) == "week1"
    out = _decide(client, "dyad_thu", 1, "2026-01-21T07:30:00")  # study week 1
    assert _reason(out["rid"]) == "week1"


def test_partial_week_dyad_and_cohort_with_partial_weeks(client, monkeypatch):
    # Four Monday starters plus one Thursday starter (week 1 from 2026-01-12).
    install_learned_policy(monkeypatch, pi=0.6)
    for i in range(4):
        register_group(client, f"dyad_{i:03d}")
    register_group(client, "dyad_thu", start="2026-01-08")

    # Partial week of the Thursday starter: nobody has completed week 1.
    out = _decide(client, "dyad_thu", 0, "2026-01-08T07:30:00")
    assert _reason(out["rid"]) == "cohort"
    # Week of 2026-01-12: only the four Monday starters completed week 1 -> cohort.
    out = _decide(client, "dyad_000", 0, "2026-01-13T07:30:00")
    assert _reason(out["rid"]) == "cohort"
    # Week of 2026-01-19: all five completed week 1. The Thursday starter is
    # in study week 2 -> learned; so is a Monday starter (study week 3).
    out = _decide(client, "dyad_thu", 1, "2026-01-19T07:30:00")
    assert out["warmup"] is False and _reason(out["rid"]) is None
    out = _decide(client, "dyad_000", 1, "2026-01-20T07:30:00")
    assert out["warmup"] is False


def test_warmup_sizes_come_from_the_hyperparameters(make_app, monkeypatch):
    app = make_app(HYPERPARAMS_OVERRIDES={"warmup.n_warm": 2, "warmup.t_warm": 2})
    client = app.test_client()
    install_learned_policy(monkeypatch, pi=0.6)
    for i in range(2):
        register_group(client, f"dyad_{i:03d}")
    out = _decide(client, "dyad_000", 0, "2026-01-13T07:30:00")  # cohort filled with 2; study week 2 <= t_warm
    assert _reason(out["rid"]) == "week1"
    out = _decide(client, "dyad_000", 1, "2026-01-20T07:30:00")  # study week 3
    assert out["warmup"] is False


def test_warmup_decision_skips_the_learner(client, monkeypatch):
    calls = []
    install_learned_policy(monkeypatch, pi=0.6, calls=calls)
    register_group(client, "dyad_001")
    out = _decide(client, "dyad_001", 0, "2026-01-06T07:30:00")
    assert out["warmup"] is True
    assert calls == []


def test_decisions_on_every_agent_follow_the_same_gate(client):
    register_group(client, "dyad_001")
    upload(client, "dyad_001", "2026-01-05T05:55:00")
    for idx, (dt, ts) in enumerate(
        [("dyad_game", "2026-01-05T06:00:00"), ("aya_message", "2026-01-05T07:30:00"),
         ("cp_message", "2026-01-05T07:35:00")]
    ):
        out = assert_envelope(post_action(client, "dyad_001", dt, idx, ts), 201)
        assert _reason(out["rid"]) == "cohort"
