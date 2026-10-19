"""
Shared fixtures and an ORM-level trial driver for the WS-L learner tests
(``tests/test_learner_*.py``). No tests live here.

The driver writes ``groups``, ``data_uploads``, ``actions`` and ``study_data``
rows directly through the ORM (not through the routes), following the
API-Spec §3 calling flow: per day a pre-AYA-AM, pre-CP and pre-AYA-PM upload
each followed by its decision; on Mondays a week-closing upload at 02:00, the
weekly update at 03:00 and the pre-``dyad_game`` upload and REL decision at
06:00 (from study week 1). Decisions use the server-side warm-up rules
(contract D2/D3), :func:`app.learner.build_design_state` and, outside warm-up,
:func:`app.learner.learned_probability` (``no_policy`` fallback at 1/2). The
rewards are paired and scored with :func:`app.learner.find_outcome_upload`
and :func:`app.learner.compute_reward`.
"""

from __future__ import annotations

import datetime
import os
import sys

import numpy as np
import pytest

from app import create_app
from app.extensions import db

TESTBED_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Simulation_Testbed")
)


def make_app(overrides: dict | None = None, **config):
    """A TestingConfig app (in-memory SQLite) with optional hyperparameter overrides and config keys."""
    import config as config_module

    attrs = dict(config)
    if overrides is not None:
        attrs["HYPERPARAMS_OVERRIDES"] = overrides
    cls = type("LearnerTestConfig", (config_module.TestingConfig,), attrs)
    return create_app(cls)


@pytest.fixture
def learner_app():
    """A fresh TestingConfig app inside an app context; tables dropped afterwards."""
    app = make_app()
    with app.app_context():
        yield app
        db.session.remove()
        db.drop_all()


def load_testbed():
    """Import the Simulation_Testbed ``algos.deploy`` module, or skip the test if it is absent."""
    if not os.path.isdir(os.path.join(TESTBED_ROOT, "algos")):
        pytest.skip(f"Simulation testbed not found at {TESTBED_ROOT}")
    if TESTBED_ROOT not in sys.path:
        sys.path.append(TESTBED_ROOT)
    try:
        from algos import deploy  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"testbed import failed: {exc}")
    return deploy


# --------------------------------------------------------------- snapshots


def snapshot(**overrides) -> dict:
    """
    A complete API-Spec §4.1 snapshot with neutral values; ``overrides``
    replace fields. Kept equal to ``app.protocol.SNAPSHOT_SCHEMA`` (checked by
    ``test_snapshot_helper_matches_schema``).
    """
    data = {
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
        "cp_prompted_by_message": False,
        "cp_messages_delivered_week": 0,
        "daily_diary_completed": False,
        "relationship_quality_aya": "miss",
        "relationship_quality_cp": "miss",
        "current_game_on": "miss",
        "prior_game_action": "miss",
        "aya_diary_summary": "miss",
        "cp_diary_summary": "miss",
        "weekly_survey_completed": False,
        "weekly_relationship_score": 0.0,
    }
    data.update(overrides)
    return data


def add_group(group_id: str, consent_start: datetime.date, weeks: int = 16):
    from app.models import Group

    end = consent_start + datetime.timedelta(days=7 * weeks)
    row = Group(
        group_id=group_id,
        group_info={
            "member_list": [f"aya_{group_id}", f"cp_{group_id}"],
            "consent_start_date": consent_start.isoformat(),
            "consent_end_date": end.isoformat(),
        },
        rid=f"rg-{group_id}",
    )
    db.session.add(row)
    db.session.commit()
    return row


def add_upload(group_id: str, ts: datetime.datetime, data: dict, rid: str | None = None):
    from app.models import DataUpload

    row = DataUpload(group_id=group_id, data=data, request_timestamp=ts, rid=rid or f"u-{group_id}-{ts.isoformat()}")
    db.session.add(row)
    db.session.commit()
    return row


def add_action(
    group_id: str,
    decision_type: str,
    decision_idx: int,
    ts: datetime.datetime,
    *,
    action: int,
    action_prob: float,
    design_state: dict | None,
    slot: str | None = None,
    is_warmup: bool = True,
    warmup_reason: str | None = "week1",
):
    from app.models import Action

    raw_context = None
    if design_state is not None:
        raw_context = {"slot": slot if slot is not None else design_state.get("slot"), "design_state": design_state}
    row = Action(
        group_id=group_id,
        action=int(action),
        rid=f"a-{group_id}-{decision_type}-{decision_idx}",
        state=None,
        decision_idx=decision_idx,
        decision_type=decision_type,
        raw_context=raw_context,
        action_prob=float(action_prob),
        random_state={"mode": "test"},
        model_parameters_id=None,
        request_timestamp=ts,
        is_warmup=is_warmup,
        warmup_reason=warmup_reason,
    )
    db.session.add(row)
    db.session.commit()
    return row


# ------------------------------------------------------------ trial driver


class MiniTrial:
    """A deterministic, ORM-level miniature of the API-Spec §3 calling flow (see module docstring)."""

    def __init__(self, consents: dict[str, datetime.date], seed: int = 0, effect: float = 0.25):
        from app.study_calendar import week1_start

        self.rng = np.random.default_rng(seed)
        self.effect = float(effect)
        self.consents = dict(consents)
        self.week1 = {g: week1_start(c) for g, c in consents.items()}
        self.counters: dict[tuple[str, str], int] = {}
        self.burden = {g: {"aya": 0.0, "cp": 0.0} for g in consents}
        self.game: dict[str, int | None] = {g: None for g in consents}
        self.prior_game: dict[str, int | None] = {g: None for g in consents}
        self.last_aya_action: dict[str, int] = {g: 0 for g in consents}
        self.last_cp_action: dict[str, int] = {g: 0 for g in consents}
        self.decisions: list[dict] = []
        self.update_summaries: list[dict] = []
        for g, c in consents.items():
            add_group(g, c)

    # -- helpers

    def _study_week(self, g: str, d: datetime.date) -> int:
        from app.study_calendar import study_week

        return study_week(self.consents[g], d)

    def _active(self, g: str, d: datetime.date) -> bool:
        return self.consents[g] <= d and self._study_week(g, d) <= 14

    def _values(self, g: str, d: datetime.date, slot: str, game_miss: bool) -> dict:
        r = self.rng
        week = self._study_week(g, d)
        adh_p = 0.55 + self.effect * self.last_aya_action[g]
        adherence = 1 if r.random() < adh_p else (0 if r.random() < 0.6 else "miss")
        def item():
            return int(r.integers(1, 6)) if r.random() < 0.8 else "miss"
        cp_completed = bool(r.random() < 0.7 + 0.1 * self.last_cp_action[g])
        relq = lambda: float(r.integers(2, 11)) / 2.0 if r.random() < 0.8 else "miss"  # noqa: E731
        return snapshot(
            day_in_study=(d - self.consents[g]).days + 1,
            week_in_study=week,
            slot=slot,
            aya_diary_mood=item(),
            aya_diary_physical=item(),
            aya_app_engagement=int(r.integers(1, 5)) if r.random() < 0.9 else "miss",
            aya_app_burden=round(self.burden[g]["aya"], 6),
            previous_med_adherence=adherence,
            prompted_by_message=bool(self.last_aya_action[g]),
            cp_diary_mood=int(r.integers(1, 6)) if cp_completed else "miss",
            cp_app_engagement=int(r.integers(1, 5)) if r.random() < 0.9 else "miss",
            cp_app_burden=round(self.burden[g]["cp"], 6),
            cp_missing_rate_7d=float(r.integers(0, 8)) / 7.0,
            daily_diary_completed=cp_completed,
            relationship_quality_aya=relq(),
            relationship_quality_cp=relq(),
            current_game_on="miss" if (game_miss or self.game[g] is None) else self.game[g],
            prior_game_action="miss" if self.prior_game[g] is None else self.prior_game[g],
            aya_diary_summary=round(float(r.random()), 4) if r.random() < 0.8 else "miss",
            cp_diary_summary=round(float(r.random()), 4) if r.random() < 0.8 else "miss",
        )

    def decide(self, g: str, decision_type: str, ts: datetime.datetime, upload) -> dict:
        from app import learner
        from app.study_calendar import warmup_reason

        idx = self.counters.get((g, decision_type), 0)
        self.counters[(g, decision_type)] = idx + 1
        ds = learner.build_design_state(g, decision_type, upload, ts)
        reason = warmup_reason(self.consents[g], ts.date())
        record = {"group_id": g, "decision_type": decision_type, "ts": ts, "decision_idx": idx}
        if reason is not None:
            pi, warm = 0.5, True
        else:
            try:
                decision = learner.learned_probability(g, decision_type, ds, ts)
                pi, warm, reason = decision.action_prob, False, None
                record["decision"] = decision
            except learner.NoPolicy:
                pi, warm, reason = 0.5, True, "no_policy"
        action = int(self.rng.random() < pi)
        add_action(
            g, decision_type, idx, ts, action=action, action_prob=pi, design_state=ds,
            is_warmup=warm, warmup_reason=reason,
        )
        record.update(action=action, pi=pi, warmup=warm, reason=reason, design_state=ds)
        self.decisions.append(record)
        if decision_type == "aya_message":
            self.burden[g]["aya"] = (13 / 14) * self.burden[g]["aya"] + action
            self.last_aya_action[g] = action
        elif decision_type == "cp_message":
            self.burden[g]["cp"] = (6 / 7) * self.burden[g]["cp"] + action
            self.last_cp_action[g] = action
        else:
            self.prior_game[g] = self.game[g]
            self.game[g] = action
        return record

    def derive_rewards(self, as_of: datetime.datetime) -> int:
        """Pair every unscored decision with its outcome upload (stamped by ``as_of``) and write ``study_data``."""
        from app import learner
        from app.models import Action, DataUpload, StudyData
        from app.study_calendar import week_monday

        n = 0
        for action in Action.query.order_by(Action.request_timestamp, Action.id).all():
            if StudyData.query.filter_by(
                group_id=action.group_id, decision_type=action.decision_type, decision_idx=action.decision_idx
            ).first():
                continue
            uploads = (
                DataUpload.query.filter(
                    DataUpload.group_id == action.group_id,
                    DataUpload.request_timestamp > action.request_timestamp,
                    DataUpload.request_timestamp <= as_of,
                )
                .order_by(DataUpload.request_timestamp, DataUpload.id)
                .all()
            )
            ds = (action.raw_context or {}).get("design_state") or {}
            week = ds.get("week_in_study")
            out = learner.find_outcome_upload(
                action.decision_type, learner.action_slot(action), action.request_timestamp, week, uploads
            )
            if out is None:
                continue
            level = None
            if action.decision_type == "dyad_game":
                level, _ = learner.aya_week_mean_level(
                    action.group_id, week_monday(action.request_timestamp), as_of
                )
            reward, native = learner.compute_reward(action.decision_type, out.data, aya_week_mean_level=level)
            db.session.add(
                StudyData(
                    group_id=action.group_id,
                    decision_idx=action.decision_idx,
                    decision_type=action.decision_type,
                    action=action.action,
                    action_prob=action.action_prob,
                    state=action.state,
                    raw_context=action.raw_context or {},
                    outcome={"upload_id": out.id, "reward_native": native},
                    reward=reward,
                    request_timestamp=action.request_timestamp,
                )
            )
            n += 1
        db.session.commit()
        return n

    def run_update(self, ts: datetime.datetime, rid: str | None = None) -> dict:
        from app import learner
        from app.clock import server_now

        self.derive_rewards(ts)
        summary = learner.run_weekly_update(rid or f"upd-{ts.isoformat()}", ts, server_now())
        self.update_summaries.append(summary)
        return summary

    def run_day(self, d: datetime.date, *, update: bool = True) -> dict | None:
        """One calendar day of uploads and decisions; on Mondays also the week-closing upload and the update."""
        summary = None
        at = lambda hh, mm: datetime.datetime.combine(d, datetime.time(hh, mm))  # noqa: E731
        if d.weekday() == 0:
            for g in sorted(self.consents):
                prev = d - datetime.timedelta(days=1)
                if self.consents[g] <= prev and self._study_week(g, prev) <= 14:
                    add_upload(g, at(2, 0), self._values(g, d, "am", game_miss=True))
            if update:
                summary = self.run_update(at(3, 0))
            for g in sorted(self.consents):
                if self._active(g, d) and self._study_week(g, d) >= 1:
                    up = add_upload(g, at(6, 0), self._values(g, d, "am", game_miss=True))
                    self.decide(g, "dyad_game", at(6, 0), up)
        for g in sorted(self.consents):
            if not self._active(g, d):
                continue
            up = add_upload(g, at(7, 25), self._values(g, d, "am", game_miss=False))
            self.decide(g, "aya_message", at(7, 30), up)
            up = add_upload(g, at(7, 33), self._values(g, d, "am", game_miss=False))
            self.decide(g, "cp_message", at(7, 35), up)
            up = add_upload(g, at(19, 25), self._values(g, d, "pm", game_miss=False))
            self.decide(g, "aya_message", at(19, 30), up)
        return summary
