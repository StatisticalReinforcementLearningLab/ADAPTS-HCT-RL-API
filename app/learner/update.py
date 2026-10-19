"""
The weekly batch update of the ``inf_lsvi_eb`` learner (implementation
contract D8, D9, D13-D16; main_new.tex Algorithm 3 lines 2-10, Algorithms 1-2,
§5.2; Monitoring-Spec §5.C C1/C3/C10, §5.L, §7 F-U2/F-U3/F-U5, §9.5).

For the calendar week ``W = Monday(request_timestamp)``:

1. Cohort rule (contract D2): if ``|E_{W-1}| < n_warm`` nothing is fitted and
   no version is written (F-U3). Likewise when no agent has a single usable
   row.
2. At the first learning update the hyperparameter set is frozen
   (``in_force_from = request_timestamp``) and one ``noise_init`` row per agent
   records the design sigma^2 (Monitoring-Spec C10).
3. Agents run in the order REL, AYA, CP. For each, every dyad of ``E_W`` with
   at least one usable row is fitted locally (Algorithm 1), the
   hyper-parameters are pooled over those ``N_W`` fits (Algorithm 2) and the
   plug-in posterior is formed for every fitted dyad of ``I_W``.
   A row is a decision of the agent stamped before Monday(W) 00:00 with a
   stored ``design_state`` and a non-null ``study_data.reward`` (fallback
   decisions included when they have both).
4. The C3 gate decides ``published`` or ``rejected``; the agent's
   ``policy_versions`` row and all its ``model_parameters`` rows
   (``local_fit``, ``hyper``, ``posterior``) are written in one transaction.
   ``published_at_study = request_timestamp + wall-clock time elapsed`` at that
   commit. An exception inside an agent rolls that agent back and records a
   ``failed`` version (F-U2); the other agents still run.
5. After REL, ``q^REL_{i,W}`` (the coming week's game-on probability) is
   computed per active dyad from the REL policy in force for week ``W`` at the
   dyad's latest upload; it weights the two boundary successors of the AYA and
   CP targets (eq. (boundary-game-average)).

Warm starts come only from earlier weeks, so a rerun in the same week
reproduces the first run on unchanged data and supersedes it by recency
(contract D14): the local warm start is the dyad's local fit in the most
recent published version of an earlier week; the hyper warm start and the
F-U5 fallback are that version's ``hyper`` row.

The last row's successor is the dyad's latest upload stamped after it, at or
before ``request_timestamp`` and received (``created_at``) by
``started_wall``. When it is stamped in a later calendar week than the last
row (normally the week-closing upload) it is a boundary state with two
candidates, game off/on, weighted by ``(1 - q^REL, q^REL)`` (``q^REL = 1/2``
for dyads no longer in trial); otherwise a single candidate. ``pi_check`` at each
candidate is the probability under the agent's policy in force for week
``W - 1`` (1/2 if the dyad was in warm-up that week or had no posterior).
Without a successor upload the target of the last row is its reward alone.

There is no randomness anywhere in the update.
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
import math
from collections import defaultdict
from typing import Any

import numpy as np

from app.learner import features as F
from app.learner.eb import EBResult, hyper_update, posterior
from app.learner.local_fit import LocalFitResult, SuccessorCandidate, fit_local
from app.learner.policy import (
    SNAPSHOT_HYPER,
    SNAPSHOT_LOCAL_FIT,
    SNAPSHOT_POSTERIOR,
    policy_in_force,
    posterior_arrays,
    posterior_row,
    probability_at,
    week_policy,
)
from app.learner.state import build_design_state
from app.learner.validity import check_policy

logger = logging.getLogger(__name__)

AGENT_ORDER = (F.REL, F.AYA, F.CP)
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
AGENT_SKIPPED = "skipped"
SKIP_COHORT = "cohort_warmup"
SKIP_NO_DATA = "no_data"
L3_THRESHOLD = 1.0
L1_REL = 0.01


# ----------------------------------------------------------------- helpers


def _jsonable(value: Any) -> Any:
    """Arrays / numbers -> JSON values; non-finite floats become ``None``."""
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    return value


@dataclasses.dataclass
class _Row:
    action_id: int
    request_timestamp: datetime.datetime
    h: np.ndarray
    f: np.ndarray
    a: float
    pi: float
    r: float


@dataclasses.dataclass
class _Context:
    hp: Any
    update_rid: str
    request_timestamp: datetime.datetime
    started_wall: datetime.datetime
    W: datetime.date
    week_start: datetime.datetime
    consents: dict
    enrolled: list
    active: list
    q_rel: dict = dataclasses.field(default_factory=dict)
    q_source: str = "default"


def _elapsed_study_time(ctx: _Context) -> datetime.datetime:
    """``request_timestamp + (server_now() - started_wall)``, never before ``request_timestamp`` (contract D13)."""
    from app.clock import server_now

    elapsed = server_now() - ctx.started_wall
    if elapsed < datetime.timedelta(0):
        elapsed = datetime.timedelta(0)
    return ctx.request_timestamp + elapsed


def _load_rows(decision_type: str, dyads: set, week_start: datetime.datetime) -> tuple[dict, int]:
    """Usable rows per dyad (contract D8) ordered by ``(request_timestamp, id)``, and the count of skipped rows."""
    from sqlalchemy import and_

    from app.extensions import db
    from app.models import Action, StudyData

    query = (
        db.session.query(Action, StudyData.reward)
        .join(
            StudyData,
            and_(
                StudyData.group_id == Action.group_id,
                StudyData.decision_type == Action.decision_type,
                StudyData.decision_idx == Action.decision_idx,
            ),
        )
        .filter(
            Action.decision_type == decision_type,
            Action.request_timestamp < week_start,
            StudyData.reward.isnot(None),
        )
        .order_by(Action.group_id.asc(), Action.request_timestamp.asc(), Action.id.asc())
    )
    rows: dict[str, list[_Row]] = defaultdict(list)
    skipped = 0
    for action, reward in query:
        if action.group_id not in dyads:
            continue
        ctx = action.raw_context if isinstance(action.raw_context, dict) else None
        ds = ctx.get("design_state") if ctx else None
        if not isinstance(ds, dict):
            continue
        try:
            h, f = F.hf_from_design_state(decision_type, ds)
            r = float(reward)
            pi = float(action.action_prob)
            if not (math.isfinite(r) and 0.0 < pi < 1.0):
                raise ValueError("reward not finite or action_prob outside (0, 1)")
        except (ValueError, TypeError) as exc:
            skipped += 1
            logger.warning("Skipping %s action %s in the fit: %s", decision_type, action.id, exc)
            continue
        rows[action.group_id].append(
            _Row(action.id, action.request_timestamp, h, f, float(int(action.action)), pi, r)
        )
    return dict(rows), skipped


def _published_earlier(decision_type: str, W: datetime.date, snapshot_type: str):
    """``(ModelParameters, PolicyVersion)`` pairs of published versions of weeks < W, newest first."""
    from app.extensions import db
    from app.models import ModelParameters, PolicyVersion

    return (
        db.session.query(ModelParameters, PolicyVersion)
        .join(PolicyVersion, ModelParameters.policy_version_id == PolicyVersion.id)
        .filter(
            ModelParameters.snapshot_type == snapshot_type,
            ModelParameters.decision_type == decision_type,
            PolicyVersion.status == PolicyVersion.STATUS_PUBLISHED,
            PolicyVersion.study_week_monday < W,
        )
        .order_by(PolicyVersion.study_week_monday.desc(), PolicyVersion.id.desc(), ModelParameters.id.asc())
    )


def _warm_starts(decision_type: str, W: datetime.date, d: int) -> dict:
    """Per dyad: the local fit of the most recent published earlier version containing it (contract D8)."""
    out: dict[str, dict] = {}
    for mp, pv in _published_earlier(decision_type, W, SNAPSHOT_LOCAL_FIT):
        if mp.group_id in out:
            continue
        theta = np.asarray(mp.theta if mp.theta is not None else [], dtype=float)
        if theta.shape != (d,) or not np.all(np.isfinite(theta)):
            continue
        out[mp.group_id] = {
            "theta": theta,
            "local_fit_id": mp.id,
            "study_week_monday": pv.study_week_monday,
            "sample_size": int(mp.sample_size or 0),
        }
    return out


def _previous_hyper(decision_type: str, W: datetime.date, p_f: int):
    """``(Theta, tau, hyper row id, week)`` of the most recent published earlier version, else ``None``."""
    for mp, pv in _published_earlier(decision_type, W, SNAPSHOT_HYPER):
        theta = np.asarray(mp.theta if mp.theta is not None else [], dtype=float)
        tau = np.asarray((mp.metadata_json or {}).get("tau2") or [], dtype=float)
        if theta.shape == (p_f,) and tau.shape == (2,) and np.all(np.isfinite(theta)) and np.all(tau > 0):
            return theta, tau, mp.id, pv.study_week_monday
    return None


def _latest_upload(ctx: _Context, group_id: str, after: datetime.datetime | None = None):
    """The dyad's latest upload stamped in ``(after, request_timestamp]`` and received by ``started_wall``."""
    from app.models import DataUpload

    q = DataUpload.query.filter(
        DataUpload.group_id == group_id,
        DataUpload.request_timestamp <= ctx.request_timestamp,
        DataUpload.created_at <= ctx.started_wall,
    )
    if after is not None:
        q = q.filter(DataUpload.request_timestamp > after)
    return q.order_by(DataUpload.request_timestamp.desc(), DataUpload.id.desc()).first()


def _in_warmup(ctx: _Context, group_id: str, week_monday_date: datetime.date) -> bool:
    from app.study_calendar import warmup_reason

    consent = ctx.consents.get(group_id)
    if consent is None:
        return True
    return (
        warmup_reason(consent, week_monday_date, ctx.consents, n_warm=ctx.hp.n_warm, t_warm=ctx.hp.t_warm)
        is not None
    )


class _PrevPolicy:
    """The agent's policy in force for week W - 1, for ``pi_check`` at boundary successors (contract D8)."""

    def __init__(self, ctx: _Context, decision_type: str):
        self.ctx = ctx
        self.decision_type = decision_type
        self.prev_monday = ctx.W - datetime.timedelta(days=7)
        self.version = week_policy(decision_type, self.prev_monday)
        self._cache: dict[str, tuple | None] = {}

    def posterior(self, group_id: str):
        if group_id not in self._cache:
            arrays = None
            if self.version is not None and not _in_warmup(self.ctx, group_id, self.prev_monday):
                row = posterior_row(self.version, group_id)
                if row is not None:
                    arrays = posterior_arrays(row)
            self._cache[group_id] = arrays
        return self._cache[group_id]

    def probability(self, group_id: str, f: np.ndarray) -> float:
        arrays = self.posterior(group_id)
        if arrays is None:
            return 0.5
        pi, _, _ = probability_at(self.decision_type, f, arrays[0], arrays[1], hp=self.ctx.hp)
        return float(pi)


def _successors(ctx: _Context, decision_type: str, group_id: str, last: _Row, prev: _PrevPolicy):
    """Candidates for the last row's successor (contract D8) and their metadata."""
    upload = _latest_upload(ctx, group_id, after=last.request_timestamp)
    if upload is None:
        return [], None
    from app.study_calendar import week_monday

    ds = build_design_state(group_id, decision_type, upload, upload.request_timestamp, hp=ctx.hp)
    # A successor in a later calendar week than the last decision (normally the
    # week-closing upload) precedes a game decision that has not been drawn.
    boundary = week_monday(upload.request_timestamp) > week_monday(last.request_timestamp)
    meta: dict[str, Any] = {
        "upload_id": upload.id,
        "upload_rid": upload.rid,
        "upload_timestamp": upload.request_timestamp.isoformat(),
        "boundary": bool(boundary),
        "prev_policy_version_id": prev.version.id if prev.version is not None else None,
    }
    candidates: list[SuccessorCandidate] = []
    if boundary and decision_type in F.GAME_INDEX:
        q = float(ctx.q_rel.get(group_id, 0.5))
        meta["q_rel"] = q
        for a in (0, 1):
            h, f = F.hf_from_design_state(decision_type, F.with_game(decision_type, ds, a))
            pi = prev.probability(group_id, f)
            candidates.append(SuccessorCandidate(weight=q if a == 1 else 1.0 - q, h=h, f=f, pi=pi))
    else:
        h, f = F.hf_from_design_state(decision_type, ds)
        candidates.append(SuccessorCandidate(weight=1.0, h=h, f=f, pi=prev.probability(group_id, f)))
    meta["weights"] = [c.weight for c in candidates]
    meta["pi_check"] = [c.pi for c in candidates]
    meta["raw"] = ds["raw"]
    return candidates, meta


def _model_row(ctx: _Context, decision_type: str, snapshot_type: str, version_id: int, **kwargs):
    from app.models import ModelParameters

    return ModelParameters(
        snapshot_type=snapshot_type,
        decision_type=decision_type,
        study_week_monday=ctx.W,
        policy_version_id=version_id,
        update_rid=ctx.update_rid,
        hyperparams_hash=ctx.hp.sha256,
        **kwargs,
    )


# ------------------------------------------------------------------- agents


def _fit_agent(ctx: _Context, decision_type: str, rows_by_dyad: dict):
    """Local fits, EB and posteriors of one agent (no database writes)."""
    hp = ctx.hp
    a = hp.agent(decision_type)
    d = a.p_h + a.p_f
    warm = _warm_starts(decision_type, ctx.W, d)
    prev_policy = _PrevPolicy(ctx, decision_type) if a.gamma > 0.0 else None
    fits: dict[str, tuple[LocalFitResult, dict]] = {}
    for gid in sorted(rows_by_dyad):
        rows = rows_by_dyad[gid]
        H = np.vstack([r.h for r in rows])
        Fm = np.vstack([r.f for r in rows])
        A = np.array([r.a for r in rows])
        PI = np.array([r.pi for r in rows])
        R = np.array([r.r for r in rows])
        ws = warm.get(gid)
        successors, succ_meta = ([], None)
        if a.gamma > 0.0:
            successors, succ_meta = _successors(ctx, decision_type, gid, rows[-1], prev_policy)
        res = fit_local(
            H,
            Fm,
            A,
            PI,
            R,
            successors,
            theta_check=ws["theta"] if ws else None,
            gamma=a.gamma,
            lambda_main=a.lambda_main,
            lambda_mod=a.lambda_mod,
            sigma_sq=a.sigma_sq,
            eps=hp.eps_ridge,
            j_backups=hp.j_backups,
        )
        meta = {
            "p_h": a.p_h,
            "p_f": a.p_f,
            "gamma": a.gamma,
            "spectral_norm_GXtXplus": res.spectral_norm,
            "no_successor": res.no_successor,
            "successor": succ_meta,
            "warm_start": (
                {
                    "source": "previous",
                    "local_fit_id": ws["local_fit_id"],
                    "study_week_monday": ws["study_week_monday"],
                }
                if ws
                else {"source": "zero"}
            ),
            "theta_change_inf": res.theta_change_inf,
            "n_new_rows": res.n - (ws["sample_size"] if ws else 0),
            "backups": res.backups,
            "action_ids": [r.action_id for r in rows],
        }
        fits[gid] = (res, meta)

    gids = sorted(fits)
    prev = _previous_hyper(decision_type, ctx.W, a.p_f)
    eb: EBResult = hyper_update(
        [fits[g][0].alpha for g in gids],
        [fits[g][0].cov_aa for g in gids],
        tau0=(a.tau0_act, a.tau0_mod),
        nu0=hp.nu0,
        lambda0=hp.lambda0,
        eps=hp.eps_ridge,
        k_max=hp.k_max,
        eps_tol=hp.eps_tol,
        warm_tau=prev[1] if prev else None,
        previous=(prev[0], prev[1]) if prev else None,
        n_w=len(gids),
    )
    active = set(ctx.active)
    posts: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for gid in gids:
        if gid in active:
            res = fits[gid][0]
            posts[gid] = posterior(res.alpha, res.cov_aa, eb.theta, eb.tau, hp.eps_ridge)
    c3 = check_policy(
        decision_type,
        {g: (fits[g][0].theta, fits[g][0].cov) for g in gids},
        eb.theta,
        eb.tau,
        eb.floor,
        eb.sigma0,
        {g: (p[0], p[1]) for g, p in posts.items()},
        eps=hp.eps_ridge,
    )
    return fits, eb, prev, posts, c3


def _rel_q_from_posteriors(ctx: _Context, posts: dict) -> dict:
    """q^REL per active dyad from this run's REL posteriors (1/2 in per-dyad warm-up or without data)."""
    from app.study_calendar import per_dyad_warmup

    out: dict[str, float] = {}
    for gid in ctx.active:
        consent = ctx.consents.get(gid)
        if consent is None or per_dyad_warmup(consent, ctx.W, ctx.hp.t_warm) or gid not in posts:
            out[gid] = 0.5
            continue
        upload = _latest_upload(ctx, gid)
        if upload is None:
            out[gid] = 0.5
            continue
        ds = build_design_state(gid, F.REL, upload, ctx.request_timestamp, hp=ctx.hp)
        _, f = F.hf_from_design_state(F.REL, ds)
        pi, _, _ = probability_at(F.REL, f, posts[gid][0], posts[gid][1], hp=ctx.hp)
        out[gid] = float(pi)
    return out


def _rel_q_from_version(ctx: _Context, version) -> dict:
    """q^REL per active dyad from a stored REL version (the one in force for week W)."""
    posts = {}
    if version is not None:
        for gid in ctx.active:
            row = posterior_row(version, gid)
            if row is not None:
                alpha, cov = posterior_arrays(row)
                posts[gid] = (alpha, cov, None)
    return _rel_q_from_posteriors(ctx, posts)


def _agent_summary_base(decision_type: str) -> dict:
    return {
        "decision_type": decision_type,
        "status": AGENT_SKIPPED,
        "policy_version_id": None,
        "reason": None,
        "n_dyads_fit": 0,
        "n_posteriors": 0,
        "rows": 0,
        "rows_skipped_invalid": 0,
        "published_at_study": None,
        "c3": None,
        "eb": None,
        "l1": None,
        "l2": None,
        "l3": None,
        "no_successor_dyads": [],
        "dyads_without_posterior": {},
        "error": None,
    }


def _run_agent(ctx: _Context, decision_type: str, rows_by_dyad: dict, skipped: int) -> dict:
    """Fit, gate and atomically publish one agent; never raises (failures become a ``failed`` version)."""
    from app.clock import server_now
    from app.extensions import db
    from app.models import PolicyVersion
    from app.study_calendar import per_dyad_warmup

    hp = ctx.hp
    a = hp.agent(decision_type)
    summary = _agent_summary_base(decision_type)
    summary["rows"] = int(sum(len(v) for v in rows_by_dyad.values()))
    summary["rows_skipped_invalid"] = int(skipped)
    if not rows_by_dyad:
        summary["reason"] = SKIP_NO_DATA
        return summary
    try:
        fits, eb, prev, posts, c3 = _fit_agent(ctx, decision_type, rows_by_dyad)
        published = bool(c3["passed"])
        status = PolicyVersion.STATUS_PUBLISHED if published else PolicyVersion.STATUS_REJECTED
        reason = None if published else "C3: " + "; ".join(c3["failures"][:10])
        q_this = _rel_q_from_posteriors(ctx, posts) if decision_type == F.REL else {}

        version = PolicyVersion(
            decision_type=decision_type,
            study_week_monday=ctx.W,
            update_rid=ctx.update_rid,
            status=status,
            reason=reason,
            hyperparams_hash=hp.sha256,
            n_dyads=len(posts),
        )
        db.session.add(version)
        db.session.flush()
        fit_ids: dict[str, int] = {}
        for gid in sorted(fits):
            res, meta = fits[gid]
            row = _model_row(
                ctx,
                decision_type,
                SNAPSHOT_LOCAL_FIT,
                version.id,
                group_id=gid,
                sample_size=res.n,
                feature_dim=a.p_h + a.p_f,
                theta=_jsonable(res.theta),
                covariance=_jsonable(res.cov),
                metadata_json=_jsonable(meta),
            )
            db.session.add(row)
            db.session.flush()
            fit_ids[gid] = row.id
        hyper_meta = {
            "N_w": eb.n_w,
            "tau2": eb.tau,
            "floor": eb.floor,
            "converged": eb.converged,
            "iterations": eb.iterations,
            "grad_inf": eb.grad_inf,
            "objective": eb.objective,
            "fallback_used": eb.fallback_used,
            "termination": eb.termination,
            "newton_steps": eb.newton_steps,
            "solver_theta": eb.solver_theta,
            "solver_tau2": eb.solver_tau,
            "warm_start_tau2": eb.warm_start_tau,
            "tau0": eb.tau0,
            "previous_hyper_id": prev[2] if prev else None,
            "at_floor": eb.at_floor(L1_REL),
            "c3": c3,
        }
        db.session.add(
            _model_row(
                ctx,
                decision_type,
                SNAPSHOT_HYPER,
                version.id,
                sample_size=eb.n_w,
                feature_dim=a.p_f,
                theta=_jsonable(eb.theta),
                covariance=_jsonable(eb.sigma0),
                metadata_json=_jsonable(hyper_meta),
            )
        )
        for gid in sorted(posts):
            mean, cov, weight = posts[gid]
            meta = {"own_data_weight": weight, "local_fit_id": fit_ids[gid]}
            if decision_type == F.REL:
                meta["q_rel"] = q_this.get(gid)
            db.session.add(
                _model_row(
                    ctx,
                    decision_type,
                    SNAPSHOT_POSTERIOR,
                    version.id,
                    group_id=gid,
                    sample_size=fits[gid][0].n,
                    feature_dim=a.p_f,
                    theta=_jsonable(mean),
                    covariance=_jsonable(cov),
                    metadata_json=_jsonable(meta),
                )
            )
        if published:
            version.published_at = server_now()
            version.published_at_study = _elapsed_study_time(ctx)
        db.session.commit()
    except Exception as exc:  # noqa: BLE001 - one agent's failure must not stop the others (F-U2)
        db.session.rollback()
        logger.exception("inf_lsvi_eb update %s: %s failed", ctx.update_rid, decision_type)
        summary["status"] = PolicyVersion.STATUS_FAILED
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["reason"] = summary["error"]
        try:
            failed = PolicyVersion(
                decision_type=decision_type,
                study_week_monday=ctx.W,
                update_rid=ctx.update_rid,
                status=PolicyVersion.STATUS_FAILED,
                reason=summary["error"][:4000],
                hyperparams_hash=hp.sha256,
                n_dyads=0,
            )
            db.session.add(failed)
            db.session.commit()
            summary["policy_version_id"] = failed.id
        except Exception:  # noqa: BLE001
            db.session.rollback()
            logger.exception("could not record the failed %s version", decision_type)
        return summary

    if decision_type == F.REL:
        ctx.q_rel = q_this if published else {}
        ctx.q_source = "this_run" if published else "none"
    norms = {g: fits[g][0].spectral_norm for g in sorted(fits)}
    finite_norms = [v for v in norms.values() if v is not None and math.isfinite(v)]
    weights = [p[2] for p in posts.values() if math.isfinite(p[2])]
    summary.update(
        {
            "status": version.status,
            "policy_version_id": version.id,
            "reason": version.reason,
            "n_dyads_fit": len(fits),
            "n_posteriors": len(posts),
            "published_at_study": version.published_at_study,
            "c3": c3,
            "eb": {
                "N_w": eb.n_w,
                "theta": eb.theta,
                "tau2": eb.tau,
                "floor": eb.floor,
                "converged": eb.converged,
                "iterations": eb.iterations,
                "grad_inf": eb.grad_inf,
                "objective": eb.objective,
                "fallback_used": eb.fallback_used,
                "termination": eb.termination,
            },
            "l1": {
                "at_floor": dict(zip(("act", "mod"), eb.at_floor(L1_REL))),
                "tau2": eb.tau,
                "floor": eb.floor,
                "ratio": eb.tau / eb.floor,
            },
            "l2": {
                "mean_own_data_weight": float(np.mean(weights)) if weights else None,
                "n_dyads": len(weights),
                "by_dyad": {g: posts[g][2] for g in sorted(posts)},
            },
            "l3": (
                {
                    "threshold": L3_THRESHOLD,
                    "max_spectral_norm": max(finite_norms) if finite_norms else None,
                    "dyads_above_threshold": [g for g, v in norms.items() if v is not None and v > L3_THRESHOLD],
                    "by_dyad": norms,
                    "theta_change_inf": {g: fits[g][0].theta_change_inf for g in sorted(fits)},
                }
                if a.gamma > 0.0
                else None
            ),
            "no_successor_dyads": [g for g in sorted(fits) if fits[g][0].no_successor],
            "dyads_without_posterior": {
                g: "no_rows"
                for g in ctx.active
                if g not in posts
                and ctx.consents.get(g) is not None
                and not per_dyad_warmup(ctx.consents[g], ctx.W, hp.t_warm)
            },
        }
    )
    return summary


# ------------------------------------------------------------------- entry


def _record_first_learning_update(ctx: _Context) -> tuple[bool, int | None]:
    """Freeze the hyperparameters and write ``noise_init`` once (contract D14, D16; Monitoring-Spec C10)."""
    from app.extensions import db
    from app.hyperparams import freeze_if_needed
    from app.models import NoiseInit

    hp = ctx.hp
    frozen = freeze_if_needed(
        "first learning update", hp=hp, in_force_from=ctx.request_timestamp, commit=True
    )
    wrote = False
    for decision_type in AGENT_ORDER:
        if NoiseInit.query.filter_by(decision_type=decision_type).first() is not None:
            continue
        a = hp.agent(decision_type)
        db.session.add(
            NoiseInit(
                decision_type=decision_type,
                sigma_sq=a.sigma_sq,
                sigma_bar_sq=a.sigma_bar_sq,
                tau0_act=a.tau0_act,
                tau0_mod=a.tau0_mod,
                source="design",
                n_week1_rewards=None,
                s2=None,
                clip_used=False,
                study_week_monday=ctx.W,
            )
        )
        wrote = True
    if wrote:
        db.session.commit()
    return wrote, (frozen.version if frozen is not None else None)


def run_weekly_update(
    update_rid: str,
    request_timestamp: datetime.datetime,
    started_wall: datetime.datetime,
) -> dict:
    """
    One weekly update (contract D14). Returns the summary consumed by
    ``app.monitoring.on_update_finished`` and by ``/update``:

    ``{"update_rid", "study_week_monday", "request_timestamp", "status",
    "learning", "skip_reason", "enrolled", "active", "first_learning_update",
    "noise_init_written", "hyperparams_hash", "hyperparams_version",
    "published_at_study", "q_rel", "q_rel_source", "agents": {decision_type: {...}}}``

    ``status`` is ``"failed"`` when any agent's fit raised (its version is
    recorded as ``failed``; F-U2), else ``"completed"`` (C3 rejections report
    themselves through ``agents[...]["c3"]``). Per agent: ``status``
    (``published`` / ``rejected`` / ``failed`` / ``skipped``),
    ``policy_version_id``, ``reason``, ``n_dyads_fit`` (N_W),
    ``n_posteriors``, ``published_at_study``, ``c3`` (``{"passed", "checks",
    "failures"}``), ``eb`` (N_w, Theta, tau2, floor, converged, iterations,
    grad_inf, objective, fallback_used, termination), ``l1`` (blocks within
    1% of the floor), ``l2`` (mean own-data weight
    ``tr(Sigma_EB (Sigma_hat^aa)^{-1}) / p_f`` over the dyads in trial),
    ``l3`` (AYA/CP: spectral norm of ``G X' X_+`` per dyad and those above 1),
    ``no_successor_dyads`` and ``dyads_without_posterior`` (B5).

    Requires an application context. Values are JSON-ready.
    """
    from app.clock import server_now, to_study_local
    from app.hyperparams import get_hyperparams
    from app.models import Group
    from app.study_calendar import (
        active_dyads,
        cohort_warmup_active,
        consent_start_of,
        enrolled_dyads,
        week_monday,
        weekly_deadline,
    )

    hp = get_hyperparams()
    ts = to_study_local(request_timestamp)
    started = started_wall if started_wall is not None else server_now()
    W = week_monday(ts)
    consents = {g.group_id: consent_start_of(g) for g in Group.query.order_by(Group.group_id.asc()).all()}
    ctx = _Context(
        hp=hp,
        update_rid=update_rid,
        request_timestamp=ts,
        started_wall=started,
        W=W,
        week_start=datetime.datetime.combine(W, datetime.time.min),
        consents=consents,
        enrolled=enrolled_dyads(W, consents),
        active=active_dyads(W, hp.trial_weeks, consents),
    )
    summary: dict[str, Any] = {
        "update_rid": update_rid,
        "study_week_monday": W,
        "request_timestamp": ts,
        "status": STATUS_COMPLETED,
        "learning": False,
        "skip_reason": None,
        "enrolled": ctx.enrolled,
        "active": ctx.active,
        "first_learning_update": False,
        "noise_init_written": False,
        "hyperparams_hash": hp.sha256,
        "hyperparams_version": None,
        "published_at_study": None,
        "q_rel": {},
        "q_rel_source": None,
        "agents": {dt: _agent_summary_base(dt) for dt in AGENT_ORDER},
    }

    if cohort_warmup_active(W, consents, n_warm=hp.n_warm):
        summary["skip_reason"] = SKIP_COHORT
        for dt in AGENT_ORDER:
            summary["agents"][dt]["reason"] = SKIP_COHORT
        return _jsonable(summary)

    enrolled = set(ctx.enrolled)
    loaded = {dt: _load_rows(dt, enrolled, ctx.week_start) for dt in AGENT_ORDER}
    if not any(rows for rows, _ in loaded.values()):
        summary["skip_reason"] = SKIP_NO_DATA
        for dt in AGENT_ORDER:
            summary["agents"][dt]["reason"] = SKIP_NO_DATA
            summary["agents"][dt]["rows_skipped_invalid"] = loaded[dt][1]
        return _jsonable(summary)

    summary["learning"] = True
    wrote, version_no = _record_first_learning_update(ctx)
    summary["first_learning_update"] = wrote
    summary["noise_init_written"] = wrote
    summary["hyperparams_version"] = version_no

    for dt in AGENT_ORDER:
        rows, skipped = loaded[dt]
        summary["agents"][dt] = _run_agent(ctx, dt, rows, skipped)
        if dt == F.REL:
            # q^REL comes from the REL policy that governs week W: this run's
            # version if it was published by the deadline, else the version in
            # force (F-U2), else 1/2 (contract D8, D13).
            in_force = policy_in_force(F.REL, weekly_deadline(W, _deadline()))
            this_id = summary["agents"][F.REL]["policy_version_id"]
            this_published = summary["agents"][F.REL]["status"] == "published"
            if not (this_published and in_force is not None and in_force.id == this_id):
                ctx.q_rel = _rel_q_from_version(ctx, in_force)
                ctx.q_source = f"version:{in_force.id}" if in_force is not None else "default"

    summary["q_rel"] = dict(ctx.q_rel)
    summary["q_rel_source"] = ctx.q_source
    published = [a["published_at_study"] for a in summary["agents"].values() if a["published_at_study"] is not None]
    summary["published_at_study"] = max(published) if published else None
    if any(a["status"] == STATUS_FAILED for a in summary["agents"].values()):
        summary["status"] = STATUS_FAILED
    return _jsonable(summary)


def _deadline() -> str:
    from app.learner.policy import _deadline_spec

    return _deadline_spec()


__all__ = ["AGENT_ORDER", "run_weekly_update"]
