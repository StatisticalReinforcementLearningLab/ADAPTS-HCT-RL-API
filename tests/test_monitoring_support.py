"""
WS-M test support for ``tests/test_monitoring_*.py`` (Monitoring-Spec §11.3).

- :func:`make_app` / the ``mon_app`` fixture: a fresh ``config.TestingConfig``
  app (in-memory SQLite, host clock, /update inline) per test.
- ORM builders that write exactly the rows a check reads (the learner may be
  bypassed: ``policy_versions`` / ``model_parameters`` rows are built directly).
- :class:`HttpTrial`: a miniature of the API-Spec §3 calling flow driven
  through the real HTTP routes, so the inline hooks, the after-response step
  and the learner all run as in production.

A couple of self-tests below keep the helpers honest.
"""

from __future__ import annotations

import datetime
import json

import numpy as np
import pytest

from app import create_app
from app.extensions import db

D = datetime.date
DT = datetime.datetime
MON1 = D(2026, 1, 5)  # a Monday
AYA, CP, REL = "aya_message", "cp_message", "dyad_game"


def make_app(**config):
    """A TestingConfig app with extra config keys (e.g. ``HYPERPARAMS_OVERRIDES``)."""
    import config as config_module

    cfg = type("MonitoringTestConfig", (config_module.TestingConfig,), dict(config))
    return create_app(cfg)


@pytest.fixture
def mon_app():
    app = make_app()
    with app.app_context():
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def mon_client(mon_app):
    return mon_app.test_client()


@pytest.fixture
def app_factory():
    """``app_factory(**config)`` -> an app with its context pushed; torn down after the test."""
    created = []

    def factory(**config):
        app = make_app(**config)
        ctx = app.app_context()
        ctx.push()
        created.append(ctx)
        return app

    yield factory
    for ctx in reversed(created):
        db.session.remove()
        db.drop_all()
        ctx.pop()


def at(day: D, hh: int = 0, mm: int = 0) -> DT:
    return DT.combine(day, datetime.time(hh, mm))


# ------------------------------------------------------------------ builders


def add_dyad(gid: str, start: D, end: D | None = None, *, recorded_at: DT | None = None, members=None):
    """A ``groups`` row and its ``group_history`` row (as /register_group writes them)."""
    from app.models import Group, GroupHistory

    end = end or start + datetime.timedelta(days=7 * 16)
    members = members or [f"aya_{gid}", f"cp_{gid}"]
    info = {"member_list": members, "consent_start_date": start.isoformat(), "consent_end_date": end.isoformat()}
    group = Group.query.filter_by(group_id=gid).first()
    if group is None:
        db.session.add(Group(group_id=gid, group_info=info, rid=f"rg-{gid}"))
    else:
        group.group_info = info
    db.session.add(GroupHistory(group_id=gid, member_list=members, consent_start_date=start.isoformat(),
                                consent_end_date=end.isoformat(), rid=f"rg-{gid}", recorded_at=recorded_at))
    db.session.commit()


_SEQ = {"n": 0}


def _next(prefix: str) -> str:
    _SEQ["n"] += 1
    return f"{prefix}-{_SEQ['n']}"


def add_upload(gid: str, ts: DT, data: dict | None = None, *, created_at: DT | None = None, rid: str | None = None,
               **fields):
    from app.models import DataUpload

    payload = dict(data or {"slot": "am" if ts.hour < 12 else "pm"})
    payload.update(fields)
    row = DataUpload(group_id=gid, data=payload, request_timestamp=ts, rid=rid or _next("u"), created_at=created_at)
    db.session.add(row)
    db.session.commit()
    return row


def design_state(decision_type: str, raw=None, *, imputed=None, slot=None) -> dict:
    """A stored design state of contract D7 with the agent's layout (``raw`` defaults to mid-range values)."""
    from app.learner import features as F

    names = list(F.NAMES[decision_type])
    if raw is None:
        raw = []
        for name, rng in zip(names, F.RANGES[decision_type]):
            raw.append(1.0 if rng is None else (rng[0] + rng[1]) / 2.0)
    return {"decision_type": decision_type, "names": names, "raw": [float(x) for x in raw], "upload_id": None,
            "upload_rid": None, "imputed": dict(imputed or {}), "clipped": [], "slot": slot}


def draw(app, pi: float, *, shift: int = 0) -> tuple[int, dict]:
    """One buffer uniform as /action draws it (contract D11); ``shift`` mimics FAULT_SAMPLER_SHIFT."""
    sampler = app.sampler
    start = sampler.cursor()
    if shift:
        sampler.restore({"normal": start["normal"], "uniform": start["uniform"] + shift})
        u = sampler.draw_uniform()
        sampler.restore({"normal": start["normal"], "uniform": start["uniform"] + 1})
    else:
        u = sampler.draw_uniform()
    end = sampler.cursor()
    return int(u < pi), {"mode": "warmup", "u": u, "sampler_cursor_start": start, "sampler_cursor_end": end}


def add_action(gid: str, decision_type: str, idx: int, ts: DT, *, action: int = 0, pi: float = 0.5,
               warmup: bool = True, reason: str | None = "week1", ds: dict | None = None, random_state=None,
               hyperparams_hash=None, model_parameters_id=None, policy_version_id=None, mu=None, v=None,
               timestamp: DT | None = None, raw_context=None, rid: str | None = None):
    from app.models import Action

    if raw_context is None and ds is not None:
        raw_context = {"design_state": ds, "slot": ds.get("slot")}
    row = Action(
        group_id=gid,
        action=int(action),
        rid=rid or _next("a"),
        state=None,
        decision_idx=idx,
        decision_type=decision_type,
        raw_context=raw_context,
        action_prob=float(pi),
        random_state=random_state if random_state is not None else {"mode": "test"},
        model_parameters_id=model_parameters_id,
        request_timestamp=ts,
        is_warmup=warmup,
        warmup_reason=reason if warmup else None,
        timestamp=timestamp,
        policy_version_id=policy_version_id,
        hyperparams_hash=hyperparams_hash,
        mu=mu,
        v=v,
    )
    db.session.add(row)
    db.session.commit()
    return row


def add_update(rid: str, ts: DT, *, status: str = "completed", published_at_study: DT | None = None,
               started_at: DT | None = None, created_at: DT | None = None, error_message: str | None = None):
    from app.models import ModelUpdateRequests
    from app.study_calendar import week_monday

    row = ModelUpdateRequests(rid, ts, status=status, created_at=created_at, study_week_monday=week_monday(ts),
                              started_at=started_at, published_at_study=published_at_study)
    if status == "completed" and published_at_study is None:
        row.published_at_study = ts + datetime.timedelta(seconds=5)
    row.error_message = error_message
    db.session.add(row)
    db.session.commit()
    return row


def add_version(decision_type: str, week: D, rid: str, *, status: str = "published",
                published_at_study: DT | None = None, hyperparams_hash: str | None = None, reason=None):
    from app.models import PolicyVersion

    if status == "published" and published_at_study is None:
        published_at_study = at(week, 3, 1)
    row = PolicyVersion(decision_type=decision_type, study_week_monday=week, update_rid=rid, status=status,
                        reason=reason, published_at_study=published_at_study, hyperparams_hash=hyperparams_hash,
                        n_dyads=0)
    db.session.add(row)
    db.session.commit()
    return row


def add_model_row(snapshot_type: str, decision_type: str, version, *, group_id=None, theta=None, cov=None,
                  meta=None, hyperparams_hash=None):
    from app.models import ModelParameters

    row = ModelParameters(snapshot_type=snapshot_type, decision_type=decision_type, group_id=group_id,
                          theta=theta, covariance=cov, metadata_json=meta or {},
                          study_week_monday=version.study_week_monday, policy_version_id=version.id,
                          update_rid=version.update_rid, hyperparams_hash=hyperparams_hash)
    db.session.add(row)
    db.session.commit()
    return row


def ctx(now: DT, window_start: DT | None = None, window_end: DT | None = None):
    from app.monitoring.checks.common import CheckContext

    return CheckContext(now=now, window_start=window_start or now - datetime.timedelta(days=1),
                        window_end=window_end or now)


def alarms(item_id: str | None = None) -> list:
    from app.models import MonitorAlarm

    q = MonitorAlarm.query
    if item_id is not None:
        q = q.filter_by(item_id=item_id)
    return q.order_by(MonitorAlarm.id.asc()).all()


def events(code: str | None = None) -> list:
    from app.models import ApiErrorEvent

    q = ApiErrorEvent.query
    if code is not None:
        q = q.filter_by(code=code)
    return q.order_by(ApiErrorEvent.id.asc()).all()


def keys(findings) -> list[str]:
    return [f.dedup_key for f in findings]


# --------------------------------------------------------------- snapshots


def snapshot(consent_start: D, ts: DT, **overrides) -> dict:
    """A field-valid, cross-field-consistent API-Spec §4.1 snapshot for an upload at ``ts``."""
    from app.protocol import SNAPSHOT_SCHEMA
    from app.study_calendar import day_in_study, study_week

    week = study_week(consent_start, ts.date())
    data = {
        "day_in_study": day_in_study(consent_start, ts.date()),
        "week_in_study": week,
        "slot": "am" if ts.hour < 12 else "pm",
        "aya_diary_mood": 3,
        "aya_diary_physical": 3,
        "aya_app_engagement": 2,
        "aya_app_burden": 0.0,
        "aya_missing_rate_7d": 0.0,
        "previous_med_adherence": 1,
        "prompted_by_message": False,
        "aya_messages_delivered_week": 0,
        "cp_diary_mood": 3,
        "cp_app_engagement": 2,
        "cp_app_burden": 0.0,
        "cp_missing_rate_7d": 0.0,
        "daily_diary_completed": True,
        "cp_prompted_by_message": False,
        "cp_messages_delivered_week": 0,
        "relationship_quality_aya": 3.0,
        "relationship_quality_cp": 3.0,
        "current_game_on": "miss" if week == 0 else 0,
        "prior_game_action": "miss",
        "aya_diary_summary": 0.5,
        "cp_diary_summary": 0.5,
        "weekly_survey_completed": True,
        "weekly_relationship_score": 3.0,
    }
    assert list(data) == list(SNAPSHOT_SCHEMA), "test snapshot drifted from SNAPSHOT_SCHEMA"
    data.update(overrides)
    return data


def post_register(client, gid: str, start: D, end: D | None = None, members=None):
    end = end or start + datetime.timedelta(days=7 * 16)
    return client.post("/api/v1/register_group", json={
        "group_id": gid, "member_list": members or [f"aya_{gid}", f"cp_{gid}"],
        "consent_start_date": start.isoformat(), "consent_end_date": end.isoformat()})


def post_upload(client, gid: str, consent_start: D, ts: DT, **overrides):
    return client.post("/api/v1/upload_data", json={"group_id": gid, "timestamp": ts.isoformat(),
                                                     "data": snapshot(consent_start, ts, **overrides)})


def post_action(client, gid: str, decision_type: str, idx: int, ts: DT):
    return client.post("/api/v1/action", json={"group_id": gid, "decision_type": decision_type,
                                                "decision_idx": idx, "timestamp": ts.isoformat()})


def post_update(client, ts: DT):
    return client.post("/api/v1/update", json={"timestamp": ts.isoformat()})


class HttpTrial:
    """
    The API-Spec §3 calling flow through the real routes: per day a pre-AYA-AM,
    pre-CP and pre-AYA-PM upload each followed by its ``/action``; on Mondays a
    week-closing upload at 02:00, ``/update`` at 03:00 and the pre-``dyad_game``
    upload and REL ``/action`` at 06:00 from study week 1. Values are random
    but valid; game status, burdens and delivery counts follow the actions.
    """

    def __init__(self, client, consents: dict[str, D], seed: int = 0, *, skip_closing: set | None = None):
        from app.study_calendar import study_week

        self.client = client
        self.consents = dict(consents)
        self.rng = np.random.default_rng(seed)
        self.idx: dict[tuple[str, str], int] = {}
        self.game: dict[str, int | None] = {g: None for g in consents}
        self.prior_game: dict[str, int | None] = {g: None for g in consents}
        self.burden = {g: {"aya": 0.0, "cp": 0.0} for g in consents}
        self.sent = {g: {"aya": 0, "cp": 0} for g in consents}
        self.last = {g: {"aya": 0, "cp": 0} for g in consents}
        self.skip_closing = set(skip_closing or ())
        self.study_week = study_week
        self.answers: list[dict] = []
        self.update_rids: list[str] = []
        for g, start in consents.items():
            r = post_register(client, g, start)
            assert r.status_code == 201, r.get_json()

    def _active(self, g: str, d: D) -> bool:
        return self.consents[g] <= d and self.study_week(self.consents[g], d) <= 14

    def _values(self, g: str, ts: DT, *, game_miss: bool) -> dict:
        r = self.rng
        week = self.study_week(self.consents[g], ts.date())

        def item():
            return int(r.integers(1, 6)) if r.random() < 0.85 else "miss"

        def score():
            return float(r.integers(2, 11)) / 2.0 if r.random() < 0.85 else "miss"

        game = "miss" if (game_miss or week == 0 or self.game[g] is None) else self.game[g]
        prior = "miss" if (week <= 1 or self.prior_game[g] is None) else self.prior_game[g]
        return snapshot(
            self.consents[g], ts,
            aya_diary_mood=item(), aya_diary_physical=item(), cp_diary_mood=item(),
            aya_app_engagement=int(r.integers(1, 5)) if r.random() < 0.9 else "miss",
            cp_app_engagement=int(r.integers(1, 5)) if r.random() < 0.9 else "miss",
            aya_app_burden=round(self.burden[g]["aya"], 6), cp_app_burden=round(self.burden[g]["cp"], 6),
            previous_med_adherence=(1 if r.random() < 0.5 + 0.3 * self.last[g]["aya"] else (0 if r.random() < 0.6
                                                                                           else "miss")),
            prompted_by_message=bool(self.last[g]["aya"]), cp_prompted_by_message=bool(self.last[g]["cp"]),
            aya_messages_delivered_week=self.sent[g]["aya"], cp_messages_delivered_week=self.sent[g]["cp"],
            daily_diary_completed=bool(r.random() < 0.8),
            relationship_quality_aya=score(), relationship_quality_cp=score(),
            current_game_on=game, prior_game_action=prior,
            aya_diary_summary=round(float(r.random()), 4), cp_diary_summary=round(float(r.random()), 4),
            cp_missing_rate_7d=round(float(r.integers(0, 8)) / 7.0, 6),
        )

    def upload(self, g: str, ts: DT, *, game_miss: bool = False):
        r = self.client.post("/api/v1/upload_data", json={"group_id": g, "timestamp": ts.isoformat(),
                                                           "data": self._values(g, ts, game_miss=game_miss)})
        assert r.status_code == 201, r.get_json()
        return r

    def decide(self, g: str, decision_type: str, ts: DT) -> dict:
        idx = self.idx.get((g, decision_type), 0)
        self.idx[(g, decision_type)] = idx + 1
        r = post_action(self.client, g, decision_type, idx, ts)
        body = r.get_json()
        assert r.status_code == 201, body
        self.answers.append(body)
        a = int(body["action"])
        if decision_type == AYA:
            self.burden[g]["aya"] = (13 / 14) * self.burden[g]["aya"] + a
            self.sent[g]["aya"] += a
            self.last[g]["aya"] = a
        elif decision_type == CP:
            self.burden[g]["cp"] = (6 / 7) * self.burden[g]["cp"] + a
            self.sent[g]["cp"] += a
            self.last[g]["cp"] = a
        else:
            self.prior_game[g] = self.game[g]
            self.game[g] = a
        return body

    def update(self, ts: DT) -> dict:
        r = post_update(self.client, ts)
        body = r.get_json()
        assert r.status_code == 202, body
        self.update_rids.append(body["rid"])
        return body

    def run_day(self, d: D, *, update: bool = True) -> None:
        if d.weekday() == 0:
            prev = d - datetime.timedelta(days=1)
            for g in sorted(self.consents):
                if self._active(g, prev) and g not in self.skip_closing:
                    self.upload(g, at(d, 2, 0), game_miss=True)
                # The delivery counters reset after the week-closing upload (API-Spec §4.1).
                self.sent[g] = {"aya": 0, "cp": 0}
            if update:
                self.update(at(d, 3, 0))
            for g in sorted(self.consents):
                if self._active(g, d) and self.study_week(self.consents[g], d) >= 1:
                    self.upload(g, at(d, 6, 0), game_miss=True)
                    self.decide(g, REL, at(d, 6, 0))
        for g in sorted(self.consents):
            if not self._active(g, d):
                continue
            self.upload(g, at(d, 7, 25))
            self.decide(g, AYA, at(d, 7, 30))
            self.upload(g, at(d, 7, 33))
            self.decide(g, CP, at(d, 7, 35))
            self.upload(g, at(d, 19, 25))
            self.decide(g, AYA, at(d, 19, 30))

    def run(self, start: D, end: D, *, update: bool = True) -> None:
        d = start
        while d <= end:
            self.run_day(d, update=update)
            d += datetime.timedelta(days=1)


def raw_json(payload) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------- self-tests


def test_snapshot_helper_passes_upload_validation(mon_client):
    assert post_register(mon_client, "s1", D(2026, 1, 7)).status_code == 201
    r = post_upload(mon_client, "s1", D(2026, 1, 7), at(D(2026, 1, 8), 7, 25))
    assert r.status_code == 201, r.get_json()


def test_draw_consumes_exactly_one_uniform(mon_app):
    before = mon_app.sampler.cursor()
    _, rs = draw(mon_app, 0.5)
    assert rs["sampler_cursor_start"] == before
    assert rs["sampler_cursor_end"]["uniform"] == before["uniform"] + 1
    assert rs["sampler_cursor_end"]["normal"] == before["normal"]
