"""
``/action`` learned path, fallbacks and failure answers for the
``inf_lsvi_eb`` learner (API-Spec §2.2; Monitoring-Spec §6.3, §7 F-A2/F-A3,
A13; contract D11-D13, D21). The learner functions of contract §3.1 are
monkeypatched so the route plumbing is tested independently of the fit.
"""

from __future__ import annotations

import datetime
import math

import pytest
from sqlalchemy.exc import OperationalError

from app import learner
from app.extensions import db
from app.models import Action, ModelParameters
from app.routes import action as action_module
from tests.conftest import (
    assert_envelope,
    install_learned_policy,
    post_action,
    register_group,
    upload,
)

GID = "dyad_000"
DECISION_TS = "2026-01-13T07:30:00"  # Tuesday of study week 2 for a 2026-01-05 consent
STATE = [1.0, 0.25, -0.5, 1.0]


@pytest.fixture
def cohort(client):
    """Five dyads that completed week 1 by Monday 2026-01-12, so decisions in week 2 are learned (D2, D3)."""
    for i in range(5):
        assert register_group(client, f"dyad_{i:03d}").status_code == 201
    assert upload(client, GID, "2026-01-13T07:25:00").status_code == 201
    return client


@pytest.fixture
def posterior_row(app):
    row = ModelParameters(
        snapshot_type="posterior",
        group_id=GID,
        decision_type="aya_message",
        theta=[0.1, 0.2, 0.3],
        covariance=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        study_week_monday=datetime.date(2026, 1, 12),
        policy_version_id=7,
    )
    db.session.add(row)
    db.session.commit()
    return row.id


def _buffer_uniform(app, index: int) -> float:
    """The buffer's uniform at ``index`` (re-read through the public cursor API, cursor left unchanged)."""
    sampler = app.sampler
    saved = sampler.cursor()
    sampler.restore({"normal": saved["normal"], "uniform": index})
    value = sampler.draw_uniform()
    sampler.restore(saved)
    return value


def _row(out) -> Action:
    return Action.query.filter_by(rid=out["rid"]).one()


def _assert_one_uniform(row: Action, cursor_before: dict):
    rs = row.random_state
    assert rs["sampler_cursor_start"] == cursor_before
    assert rs["sampler_cursor_end"] == {"normal": cursor_before["normal"], "uniform": cursor_before["uniform"] + 1}


# ------------------------------------------------------------------ learned


def test_learned_decision_persists_every_column(cohort, app, monkeypatch, posterior_row):
    calls = []
    install_learned_policy(monkeypatch, pi=0.63, mu=0.12, v=0.04, model_parameters_id=posterior_row,
                           state=STATE, calls=calls)
    cursor = app.sampler.cursor()
    out = assert_envelope(post_action(cohort, GID, "aya_message", 16, DECISION_TS), 201, "Success",
                          "Action requested successfully.")
    assert out["warmup"] is False
    assert out["action_prob"] == pytest.approx(0.63)
    assert out["state"] == STATE
    assert out["model_param"] == [0.1, 0.2, 0.3]

    row = _row(out)
    assert row.is_warmup is False
    assert row.warmup_reason is None
    assert row.action_prob == pytest.approx(0.63)
    assert row.state == STATE
    assert row.model_param == [0.1, 0.2, 0.3]
    assert row.policy_week == datetime.date(2026, 1, 12)
    assert row.policy_version_id == 7
    assert row.model_parameters_id == posterior_row
    assert row.hyperparams_hash == "f" * 64
    assert row.mu == pytest.approx(0.12)
    assert row.v == pytest.approx(0.04)
    assert row.request_timestamp == datetime.datetime(2026, 1, 13, 7, 30)
    assert row.upload_id == row.raw_context["upload_id"] == row.raw_context["design_state"]["upload_id"]
    assert row.raw_context["slot"] == "am"  # legacy projection kept alongside

    # The learner saw the stored design state and the Detroit-local decision time.
    assert calls == [(GID, "aya_message", row.raw_context["design_state"], datetime.datetime(2026, 1, 13, 7, 30))]

    # Exactly one uniform from the buffer, action = int(u < pi) (D11).
    _assert_one_uniform(row, cursor)
    assert row.random_state["mode"] == "learned"
    u = row.random_state["u"]
    assert u == _buffer_uniform(app, cursor["uniform"])
    assert row.action == int(u < 0.63) == out["action"]


def test_each_agent_consumes_exactly_one_uniform(cohort, app, monkeypatch):
    install_learned_policy(monkeypatch, pi=0.5)
    upload(cohort, GID, "2026-01-12T05:55:00")
    for idx, (decision_type, ts) in enumerate(
        [("aya_message", DECISION_TS), ("cp_message", "2026-01-13T07:35:00"), ("dyad_game", "2026-01-13T07:40:00")]
    ):
        before = app.sampler.cursor()
        out = assert_envelope(post_action(cohort, GID, decision_type, idx, ts), 201)
        _assert_one_uniform(_row(out), before)


def test_decision_runs_under_the_process_lock(cohort, monkeypatch):
    seen = []

    def learned_probability(group_id, decision_type, design_state, decision_time):
        seen.append(action_module.ACTION_LOCK.locked())
        raise learner.NoPolicy("none")

    install_learned_policy(monkeypatch)
    monkeypatch.setattr(learner, "learned_probability", learned_probability)
    assert_envelope(post_action(cohort, GID, "aya_message", 1, DECISION_TS), 201)
    assert seen == [True]
    assert not action_module.ACTION_LOCK.locked()


def test_replay_of_a_learned_decision(cohort, app, monkeypatch):
    install_learned_policy(monkeypatch, pi=0.7, state=STATE)
    first = assert_envelope(post_action(cohort, GID, "aya_message", 3, DECISION_TS), 201)
    cursor = app.sampler.cursor()
    second = assert_envelope(post_action(cohort, GID, "aya_message", 3, "2026-01-13T07:31:00"), 200,
                             "Duplicate Decision", f"Returning the existing action for ({GID}, aya_message, 3).")
    for key in ("rid", "action", "action_prob", "warmup", "state", "model_param", "group_id", "decision_type",
                "decision_idx"):
        assert second[key] == first[key], key
    assert app.sampler.cursor() == cursor


# ---------------------------------------------------------------- fallbacks


def _assert_fallback(out, row, reason, *, state):
    assert out["message"] == "Action requested successfully using a fallback."
    assert out["warmup"] is True
    assert out["action_prob"] == 0.5
    assert out["model_param"] is None
    assert out["state"] == state
    assert row.warmup_reason == reason
    assert row.is_warmup is True
    assert row.action_prob == 0.5
    assert row.model_param is None
    assert row.model_parameters_id is None
    assert row.policy_version_id is None and row.policy_week is None
    assert row.mu is None and row.v is None
    assert row.random_state["mode"] == "fallback"
    assert row.action == int(row.random_state["u"] < 0.5)


def test_no_policy_fallback_keeps_the_state(cohort, app, monkeypatch):
    install_learned_policy(monkeypatch, state=STATE, raises=learner.NoPolicy("no posterior for dyad_000"))
    cursor = app.sampler.cursor()
    out = assert_envelope(post_action(cohort, GID, "aya_message", 1, DECISION_TS), 201)
    row = _row(out)
    _assert_fallback(out, row, "no_policy", state=STATE)
    assert row.state == STATE
    assert "design_state" in row.raw_context
    assert row.hyperparams_hash == app.hyperparams.sha256
    _assert_one_uniform(row, cursor)


def test_learner_exception_is_learner_error(cohort, app, monkeypatch):
    install_learned_policy(monkeypatch, state=STATE, raises=ValueError("singular matrix"))
    cursor = app.sampler.cursor()
    out = assert_envelope(post_action(cohort, GID, "aya_message", 1, DECISION_TS), 201)
    row = _row(out)
    _assert_fallback(out, row, "learner_error", state=STATE)
    assert row.raw_context["learner_error"].startswith("ValueError")
    _assert_one_uniform(row, cursor)


def test_state_construction_failure_is_learner_error_with_null_state(cohort, monkeypatch):
    install_learned_policy(monkeypatch)

    def broken(*args, **kwargs):
        raise KeyError("aya_app_engagement")

    monkeypatch.setattr(learner, "build_design_state", broken)
    out = assert_envelope(post_action(cohort, GID, "aya_message", 1, DECISION_TS), 201)
    row = _row(out)
    _assert_fallback(out, row, "learner_error", state=None)
    assert row.state is None
    assert "design_state_error" in row.raw_context


def test_standardization_failure_is_learner_error(cohort, monkeypatch):
    install_learned_policy(monkeypatch)

    def broken(decision_type, design_state):
        raise ValueError("bad coordinate")

    monkeypatch.setattr(learner, "standardized_g", broken)
    out = assert_envelope(post_action(cohort, GID, "aya_message", 1, DECISION_TS), 201)
    _assert_fallback(out, _row(out), "learner_error", state=None)


def test_fault_action_crash_only_hits_the_listed_agent(cohort, app, monkeypatch):
    install_learned_policy(monkeypatch, pi=0.6, state=STATE)
    app.config["FAULT_ACTION_CRASH"] = frozenset({"aya_message"})
    out = assert_envelope(post_action(cohort, GID, "aya_message", 1, DECISION_TS), 201)
    row = _row(out)
    _assert_fallback(out, row, "learner_error", state=STATE)
    assert "FAULT_ACTION_CRASH" in row.raw_context["learner_error"]
    out = assert_envelope(post_action(cohort, GID, "cp_message", 1, "2026-01-13T07:35:00"), 201)
    assert out["warmup"] is False


def test_forced_out_of_range_pi_is_invalid_probability(cohort, app, monkeypatch):
    install_learned_policy(monkeypatch, pi=0.6, state=STATE)
    app.config["FAULT_FORCE_PI"] = 0.95
    cursor = app.sampler.cursor()
    out = assert_envelope(post_action(cohort, GID, "aya_message", 1, DECISION_TS), 201)
    row = _row(out)
    _assert_fallback(out, row, "invalid_probability", state=STATE)
    assert row.raw_context["rejected_policy"]["pi"] == pytest.approx(0.95)
    assert row.raw_context["rejected_policy"]["policy_version_id"] == 7
    _assert_one_uniform(row, cursor)


@pytest.mark.parametrize("bad_pi", [float("nan"), float("inf"), 0.2 - 1e-9, 0.8 + 1e-9, -0.1])
def test_impossible_learned_pi_is_invalid_probability(cohort, monkeypatch, bad_pi):
    install_learned_policy(monkeypatch, pi=bad_pi, state=STATE)
    out = assert_envelope(post_action(cohort, GID, "aya_message", 1, DECISION_TS), 201)
    row = _row(out)
    _assert_fallback(out, row, "invalid_probability", state=STATE)
    rejected = row.raw_context["rejected_policy"]["pi"]
    if math.isfinite(bad_pi):
        assert rejected == pytest.approx(bad_pi)
    else:
        assert isinstance(rejected, str)


@pytest.mark.parametrize("edge_pi", [0.2, 0.8, 0.2 - 5e-13, 0.8 + 5e-13])
def test_pi_at_the_allocation_bounds_is_learned(cohort, monkeypatch, edge_pi):
    install_learned_policy(monkeypatch, pi=edge_pi, state=STATE)
    out = assert_envelope(post_action(cohort, GID, "aya_message", 1, DECISION_TS), 201)
    assert out["warmup"] is False
    assert out["action_prob"] == pytest.approx(edge_pi)


def test_a13_hook_receives_the_learned_pi_and_can_reject_it(cohort, monkeypatch):
    from app import monitoring

    install_learned_policy(monkeypatch, pi=0.5, state=STATE)
    hook_calls = []

    def check(pi, *, learned, context):
        hook_calls.append((pi, learned, context))
        return False

    monkeypatch.setattr(monitoring, "check_action_probability", check)
    out = assert_envelope(post_action(cohort, GID, "aya_message", 4, DECISION_TS), 201)
    _assert_fallback(out, _row(out), "invalid_probability", state=STATE)
    assert len(hook_calls) == 1
    pi, learned, context = hook_calls[0]
    assert pi == 0.5 and learned is True
    assert context["group_id"] == GID
    assert context["decision_type"] == "aya_message"
    assert context["decision_idx"] == 4
    assert context["rid"] == out["rid"]
    assert context["policy_version_id"] == 7


def test_a13_hook_failure_does_not_block_a_valid_decision(cohort, monkeypatch):
    from app import monitoring

    install_learned_policy(monkeypatch, pi=0.55, state=STATE)

    def broken(*args, **kwargs):
        raise RuntimeError("monitoring bug")

    monkeypatch.setattr(monitoring, "check_action_probability", broken)
    out = assert_envelope(post_action(cohort, GID, "aya_message", 1, DECISION_TS), 201)
    assert out["warmup"] is False


def test_after_action_committed_gets_the_row_id_and_cannot_break_the_answer(cohort, monkeypatch):
    from app import monitoring

    install_learned_policy(monkeypatch, pi=0.55)
    ids = []

    def hook(action_id):
        ids.append(action_id)
        raise RuntimeError("monitoring bug")

    monkeypatch.setattr(monitoring, "after_action_committed", hook)
    out = assert_envelope(post_action(cohort, GID, "aya_message", 1, DECISION_TS), 201)
    assert ids == [_row(out).id]


# --------------------------------------------------------- fault injection


def test_fault_sampler_shift_reads_a_shifted_uniform_but_records_the_cursor(cohort, app, monkeypatch):
    install_learned_policy(monkeypatch, pi=0.6)
    app.config["FAULT_SAMPLER_SHIFT"] = 3
    cursor = app.sampler.cursor()
    out = assert_envelope(post_action(cohort, GID, "aya_message", 1, DECISION_TS), 201)
    row = _row(out)
    _assert_one_uniform(row, cursor)
    assert row.random_state["u"] == _buffer_uniform(app, cursor["uniform"] + 3)
    assert row.random_state["u"] != _buffer_uniform(app, cursor["uniform"])
    assert row.action == int(row.random_state["u"] < 0.6)
    assert app.sampler.cursor()["uniform"] == cursor["uniform"] + 1


def test_fault_settings_are_ignored_when_fault_injection_is_off(cohort, app, monkeypatch):
    install_learned_policy(monkeypatch, pi=0.6)
    app.config.update(
        FAULT_INJECTION_ENABLED=False,
        FAULT_FORCE_PI=0.95,
        FAULT_ACTION_CRASH=frozenset({"aya_message"}),
        FAULT_SAMPLER_SHIFT=5,
    )
    cursor = app.sampler.cursor()
    out = assert_envelope(post_action(cohort, GID, "aya_message", 1, DECISION_TS), 201)
    row = _row(out)
    assert out["warmup"] is False and row.warmup_reason is None
    assert row.action_prob == pytest.approx(0.6)
    assert row.random_state["u"] == _buffer_uniform(app, cursor["uniform"])


# --------------------------------------------------------- 503 / 500


def _store_error(*args, **kwargs):
    raise OperationalError("SELECT data_uploads", {}, Exception("database is locked"))


def test_store_unavailable_before_minting_is_503(cohort, app, monkeypatch):
    monkeypatch.setattr(action_module, "_latest_upload", _store_error)
    cursor = app.sampler.cursor()
    out = assert_envelope(post_action(cohort, GID, "aya_message", 1, DECISION_TS), 503, "Service Unavailable",
                          "The decision store is unavailable; use the host fallback.")
    assert out["group_id"] == GID and out["decision_type"] == "aya_message" and out["decision_idx"] == 1
    for key in ("action", "action_prob", "warmup", "state", "model_param"):
        assert out[key] is None
    assert app.sampler.cursor() == cursor
    monkeypatch.undo()
    assert Action.query.count() == 0


def test_group_lookup_failure_is_503(cohort, monkeypatch):
    class BrokenGroup:
        class query:  # noqa: N801
            filter_by = staticmethod(_store_error)

    monkeypatch.setattr(action_module, "Group", BrokenGroup)
    assert_envelope(post_action(cohort, GID, "aya_message", 1, DECISION_TS), 503, "Service Unavailable",
                    "The decision store is unavailable; use the host fallback.")


def test_store_error_inside_the_learner_is_503_not_a_fallback(cohort, app, monkeypatch):
    install_learned_policy(monkeypatch, raises=OperationalError("SELECT policy_versions", {}, Exception("gone")))
    cursor = app.sampler.cursor()
    assert_envelope(post_action(cohort, GID, "aya_message", 1, DECISION_TS), 503, "Service Unavailable")
    assert app.sampler.cursor() == cursor


def test_persist_failure_after_the_draw_is_500(cohort, app, monkeypatch):
    install_learned_policy(monkeypatch, pi=0.6)

    def fail_commit():
        raise OperationalError("INSERT INTO actions", {}, Exception("disk I/O error"))

    cursor = app.sampler.cursor()
    monkeypatch.setattr(db.session, "commit", fail_commit)
    out = assert_envelope(post_action(cohort, GID, "aya_message", 1, DECISION_TS), 500, "Internal Error",
                          "Failed to persist the action.")
    for key in ("action", "action_prob", "warmup", "state", "model_param"):
        assert out[key] is None
    assert out["group_id"] == GID
    # The random number was used (AC-500-PERSIST): the cursor does not rewind.
    assert app.sampler.cursor()["uniform"] == cursor["uniform"] + 1
    monkeypatch.undo()
    assert Action.query.count() == 0


def test_exhausted_buffer_is_500(cohort, app, monkeypatch):
    install_learned_policy(monkeypatch, pi=0.6)
    app.sampler.restore({"normal": 0, "uniform": app.sampler.n_uniforms})
    assert_envelope(post_action(cohort, GID, "aya_message", 1, DECISION_TS), 500, "Internal Error",
                    "Failed to persist the action.")
    assert Action.query.count() == 0


# ---------------------------------------------------------- legacy learners


@pytest.mark.parametrize("algorithm, expected_pr1", [("always_send", 1.0), ("always_none", 0.0)])
def test_legacy_learner_reports_pr_action_1(make_app, algorithm, expected_pr1):
    app = make_app(RL_ALGORITHM=algorithm)
    client = app.test_client()
    for i in range(5):
        register_group(client, f"dyad_{i:03d}")
    assert upload(client, GID, "2026-01-13T07:25:00").status_code == 201
    out = assert_envelope(post_action(client, GID, "aya_message", 1, DECISION_TS), 201, "Success",
                          "Action requested successfully.")
    assert out["warmup"] is False
    assert out["action_prob"] == expected_pr1
    assert out["action"] == int(expected_pr1)
    # Scheduled warm-up and the no-state fallback work the same way for legacy learners.
    register_group(client, "dyad_new", start="2026-01-12")
    assert_envelope(post_action(client, "dyad_new", "aya_message", 1, DECISION_TS), 201, "Success",
                    "Action requested successfully using the no-state fallback.")
    upload(client, "dyad_new", "2026-01-13T07:26:00")
    out = assert_envelope(post_action(client, "dyad_new", "aya_message", 2, DECISION_TS), 201)
    row = Action.query.filter_by(rid=out["rid"]).one()
    assert row.warmup_reason == "week1" and out["action_prob"] == 0.5


def test_legacy_flat_prob_learned_path(make_app):
    app = make_app(RL_ALGORITHM="flat_prob")
    client = app.test_client()
    for i in range(5):
        register_group(client, f"dyad_{i:03d}")
    upload(client, GID, "2026-01-13T07:25:00")
    out = assert_envelope(post_action(client, GID, "cp_message", 1, DECISION_TS), 201)
    assert out["action"] in (0, 1)
    assert out["action_prob"] in (0.5,)  # flat_prob returns Pr(A=1) = the policy row's 0.5
