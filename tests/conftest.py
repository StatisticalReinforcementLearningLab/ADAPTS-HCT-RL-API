"""
Shared fixtures and helpers for the route tests (API-Spec §2, §4.1;
Monitoring-Spec §6; implementation contract §4).

- ``app`` / ``client``: a fresh ``config.TestingConfig`` app per test
  (in-memory SQLite, host clock, fault injection honoured, /update inline).
- ``make_app``: the same with extra config keys, e.g.
  ``make_app(HYPERPARAMS_OVERRIDES={"rel_reward.w_adh": 0.5})``.
- Snapshot helpers for the 26-field flat snapshot: :func:`full_snapshot`
  (field-valid, fixed bookkeeping) and :func:`snapshot_for` /
  :func:`upload` (bookkeeping derived from the dyad's consent date and the
  upload timestamp, so the upload passes the cross-field checks).
- :func:`install_learned_policy` monkeypatches the learned path of /action
  (contract §3.1) so the route plumbing is tested independently of the fit.
  Everything else (design state, pairing, rewards) runs the real
  ``app.learner`` code.
"""

from __future__ import annotations

import datetime
import json

import pytest

from app import create_app, db
from app.protocol import SNAPSHOT_SCHEMA
from app.study_calendar import day_in_study, study_week

DEFAULT_CONSENT_START = "2026-01-05"  # a Monday: week 1 = 2026-01-05 .. 2026-01-11
DEFAULT_CONSENT_END = "2026-04-15"
ENVELOPE_KEYS = ("status", "title", "message", "server_timestamp", "rid")


def _make(config_overrides: dict | None = None):
    import config as config_module

    attrs = dict(config_overrides or {})
    cfg = type("RouteTestConfig", (config_module.TestingConfig,), attrs)
    return create_app(cfg)


@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    app_instance = _make()
    with app_instance.app_context():
        yield app_instance
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


@pytest.fixture
def make_app():
    """Factory for an app with extra config keys; each app is torn down after the test."""
    created = []

    def factory(**config_overrides):
        instance = _make(config_overrides)
        ctx = instance.app_context()
        ctx.push()
        created.append((instance, ctx))
        return instance

    yield factory
    for _instance, ctx in reversed(created):
        db.session.remove()
        db.drop_all()
        ctx.pop()


# --------------------------------------------------------------- snapshots


def full_snapshot(**overrides) -> dict:
    """A complete, field-valid /upload_data ``data`` snapshot (every API-Spec §4.1 field present)."""
    snap = {
        "day_in_study": 1,
        "week_in_study": 1,
        "slot": "am",
        "aya_diary_mood": "miss",
        "aya_diary_physical": "miss",
        "aya_app_engagement": 2,
        "aya_app_burden": 0.0,
        "aya_missing_rate_7d": 0.0,
        "previous_med_adherence": "miss",
        "prompted_by_message": False,
        "aya_messages_delivered_week": 0,
        "cp_diary_mood": "miss",
        "cp_app_engagement": 2,
        "cp_app_burden": 0.0,
        "cp_missing_rate_7d": 0.0,
        "daily_diary_completed": False,
        "cp_prompted_by_message": False,
        "cp_messages_delivered_week": 0,
        "relationship_quality_aya": "miss",
        "relationship_quality_cp": "miss",
        "current_game_on": 1,
        "prior_game_action": "miss",
        "aya_diary_summary": "miss",
        "cp_diary_summary": "miss",
        "weekly_survey_completed": False,
        "weekly_relationship_score": 0.0,
    }
    assert list(snap) == list(SNAPSHOT_SCHEMA), "full_snapshot drifted from SNAPSHOT_SCHEMA"
    snap.update(overrides)
    return snap


def _local(timestamp) -> datetime.datetime:
    from app.clock import to_study_local

    return to_study_local(timestamp)


def snapshot_for(consent_start, timestamp, noon_hour: int = 12, **overrides) -> dict:
    """
    A valid snapshot for an upload at ``timestamp`` by a dyad that consented on
    ``consent_start``: ``day_in_study``, ``week_in_study`` and ``slot`` follow
    API-Spec §4.1; in the partial week ``current_game_on`` is ``"miss"``.
    ``overrides`` win.
    """
    local = _local(timestamp)
    week = study_week(consent_start, local.date())
    base = {
        "day_in_study": day_in_study(consent_start, local.date()),
        "week_in_study": week,
        "slot": "am" if local.hour < noon_hour else "pm",
    }
    if week == 0:
        base["current_game_on"] = "miss"
    return full_snapshot(**{**base, **overrides})


def consent_start_of(client, group_id, default=None):
    """The registered ``consent_start_date`` of ``group_id`` (``default`` when unregistered)."""
    from app.models import Group

    with client.application.app_context():
        group = Group.query.filter_by(group_id=group_id).first()
        return group.group_info["consent_start_date"] if group is not None else default


def register_group(client, group_id, start=DEFAULT_CONSENT_START, end=DEFAULT_CONSENT_END, members=None):
    return client.post(
        "/api/v1/register_group",
        json={
            "group_id": group_id,
            "member_list": members if members is not None else [f"aya_{group_id}", f"cp_{group_id}"],
            "consent_start_date": start,
            "consent_end_date": end,
        },
    )


def upload(client, group_id, timestamp, **overrides):
    """POST a consistent full snapshot for ``group_id`` at ``timestamp`` (``overrides`` win)."""
    consent = consent_start_of(client, group_id, default=_local(timestamp).date().isoformat())
    noon = int(client.application.config.get("UPLOAD_SLOT_NOON_HOUR", 12))
    return client.post(
        "/api/v1/upload_data",
        json={
            "group_id": group_id,
            "timestamp": timestamp,
            "data": snapshot_for(consent, timestamp, noon_hour=noon, **overrides),
        },
    )


def post_action(client, group_id, decision_type, decision_idx, timestamp):
    return client.post(
        "/api/v1/action",
        json={
            "group_id": group_id,
            "timestamp": timestamp,
            "decision_idx": decision_idx,
            "decision_type": decision_type,
        },
    )


def assert_envelope(response, status, title=None, message=None) -> dict:
    """The common envelope (API-Spec §2 preamble): five always-present fields, nonempty ``rid``."""
    body = response.get_json()
    assert response.status_code == status, body
    assert body is not None
    for key in ENVELOPE_KEYS:
        assert key in body, f"missing envelope field {key}: {body}"
    assert body["status"] == status
    assert isinstance(body["title"], str) and body["title"]
    assert isinstance(body["message"], str) and body["message"]
    assert isinstance(body["rid"], str) and body["rid"]
    datetime.datetime.fromisoformat(body["server_timestamp"])
    if title is not None:
        assert body["title"] == title, body
    if message is not None:
        assert body["message"] == message, body
    return body


# -------------------------------------------------- learned-path stand-ins


def stub_design_state(group_id, decision_type, upload, decision_time):
    """A design-state dict of the contract §3.1 shape (route tests only)."""
    names = ["const", "eng", "burden", "game"]
    return {
        "names": names,
        "raw": [1.0, 2.0, 0.0, 1.0],
        "upload_id": upload.id,
        "upload_rid": upload.rid,
        "imputed": {},
        "clipped": [],
    }


def install_learned_policy(monkeypatch, *, pi=0.63, mu=0.12, v=0.04, policy_week=datetime.date(2026, 1, 12),
                           policy_version_id=7, model_parameters_id=None, hyperparams_hash="f" * 64,
                           state=(1.0, 0.0, -1.0, 1.0), raises=None, calls=None):
    """
    Monkeypatch the learned path of /action (contract §3.1):
    ``build_design_state`` -> :func:`stub_design_state`, ``standardized_g`` ->
    ``state``, ``learned_probability`` -> a fixed :class:`PolicyDecision` (or
    ``raises``). ``calls`` (a list) collects the ``learned_probability`` arguments.
    """
    from app import learner

    monkeypatch.setattr(learner, "build_design_state", stub_design_state)
    monkeypatch.setattr(learner, "standardized_g", lambda decision_type, design_state: list(state))

    def learned_probability(group_id, decision_type, design_state, decision_time):
        if calls is not None:
            calls.append((group_id, decision_type, design_state, decision_time))
        if raises is not None:
            raise raises
        return learner.PolicyDecision(
            action_prob=pi,
            mu=mu,
            v=v,
            model_param=[0.1, 0.2, 0.3],
            policy_version_id=policy_version_id,
            policy_week=policy_week,
            model_parameters_id=model_parameters_id,
            hyperparams_hash=hyperparams_hash,
        )

    monkeypatch.setattr(learner, "learned_probability", learned_probability)


def raw_json(payload) -> bytes:
    """Compact JSON bytes of ``payload`` (to post a body verbatim with ``data=``)."""
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")
