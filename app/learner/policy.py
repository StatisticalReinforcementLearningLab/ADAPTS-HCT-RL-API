"""
Policy in force and the learned randomization probability (implementation
contract D10, D13, D21; main_new.tex §3.6, Algorithm 3 line 12; API-Spec §2.4
"Publication deadline"; Monitoring-Spec §7 F-U2, F-A3).

A ``policy_versions`` row is *eligible* for a decision at study time ``t`` in
calendar week ``W`` iff ``status == "published"`` and
``published_at_study <= min(t, Monday(W) 05:45)`` (the deadline is
``MONITOR_WEEK_READY_BY``). The version in force for ``(agent, t)`` is the
latest eligible version of week ``W`` (a rerun supersedes earlier runs of the
week), else the latest eligible version of an earlier week (F-U2), else none.
A late version is therefore never activated midweek.

The dyad's policy is the ``model_parameters`` row with
``snapshot_type = "posterior"`` of that version; without one the decision
takes the ``no_policy`` fallback (:class:`NoPolicy`; F-A3).
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np

from app.learner import features as F
from app.learner.allocation import agent_probability, contrast_moments

SNAPSHOT_LOCAL_FIT = "local_fit"
SNAPSHOT_HYPER = "hyper"
SNAPSHOT_POSTERIOR = "posterior"
DEFAULT_DEADLINE = "MON 05:45"


class PolicyDecision(NamedTuple):
    """A learned allocation (contract D10, D13) as returned by :func:`learned_probability`."""

    action_prob: float
    mu: float
    v: float
    model_param: list[float]
    policy_version_id: int
    policy_week: datetime.date
    model_parameters_id: int
    hyperparams_hash: str


class NoPolicy(Exception):
    """No in-force policy (or no posterior for the dyad) -> ``no_policy`` fallback (contract D12, D13)."""


def _deadline_spec() -> str:
    try:
        from flask import current_app, has_app_context

        if has_app_context():
            return str(current_app.config.get("MONITOR_WEEK_READY_BY") or DEFAULT_DEADLINE)
    except Exception:  # noqa: BLE001 - outside an app the default applies
        pass
    return DEFAULT_DEADLINE


def publication_cutoff(t: datetime.datetime) -> datetime.datetime:
    """``min(t, Monday(W) 05:45)`` for a decision at ``t`` in week ``W`` (contract D13)."""
    from app.study_calendar import week_monday, weekly_deadline

    deadline = weekly_deadline(week_monday(t), _deadline_spec())
    return min(t, deadline)


def policy_in_force(decision_type: str, t: datetime.datetime):
    """The ``PolicyVersion`` in force for ``decision_type`` at study time ``t`` (contract D13), or ``None``."""
    from app.models import PolicyVersion
    from app.study_calendar import week_monday

    F.check_decision_type(decision_type)
    W = week_monday(t)
    cutoff = publication_cutoff(t)
    base = PolicyVersion.query.filter(
        PolicyVersion.decision_type == decision_type,
        PolicyVersion.status == PolicyVersion.STATUS_PUBLISHED,
        PolicyVersion.published_at_study.isnot(None),
        PolicyVersion.published_at_study <= cutoff,
    )
    version = (
        base.filter(PolicyVersion.study_week_monday == W).order_by(PolicyVersion.id.desc()).first()
    )
    if version is None:
        version = (
            base.filter(PolicyVersion.study_week_monday < W)
            .order_by(PolicyVersion.study_week_monday.desc(), PolicyVersion.id.desc())
            .first()
        )
    return version


def week_policy(decision_type: str, week_monday_date: datetime.date):
    """The version in force for the bulk of week ``W``: :func:`policy_in_force` at Monday(W) 05:45."""
    from app.study_calendar import weekly_deadline

    return policy_in_force(decision_type, weekly_deadline(week_monday_date, _deadline_spec()))


def posterior_row(version, group_id: str):
    """The dyad's ``posterior`` row of ``version`` (``None`` if absent)."""
    from app.models import ModelParameters

    if version is None:
        return None
    return (
        ModelParameters.query.filter_by(
            snapshot_type=SNAPSHOT_POSTERIOR, policy_version_id=version.id, group_id=group_id
        )
        .order_by(ModelParameters.id.desc())
        .first()
    )


def posterior_arrays(row) -> tuple[np.ndarray, np.ndarray]:
    """``(alpha_EB, Sigma_EB)`` of a posterior row as float arrays."""
    alpha = np.asarray(row.theta, dtype=float)
    cov = np.asarray(row.covariance, dtype=float)
    if alpha.ndim != 1 or cov.shape != (alpha.size, alpha.size):
        raise ValueError(f"posterior row {row.id} has inconsistent shapes {alpha.shape} / {cov.shape}.")
    return alpha, cov


def probability_at(decision_type: str, f: np.ndarray, alpha: np.ndarray, cov: np.ndarray, *, hp=None):
    """``(pi, mu, v)`` of eqs. (contrast-post), (alloc-mc) for an advantage map ``f``."""
    mu, v = contrast_moments(f, alpha, cov)
    return agent_probability(decision_type, mu, v, hp=hp), mu, v


def model_param_vector(alpha: np.ndarray, cov: np.ndarray) -> list[float]:
    """The opaque ``model_param`` of API-Spec §2.2: ``alpha_EB`` then ``Sigma_EB`` row-major."""
    return [float(x) for x in np.asarray(alpha, dtype=float)] + [
        float(x) for x in np.asarray(cov, dtype=float).ravel()
    ]


def _fault(name: str):
    try:
        from app import fault_setting

        return fault_setting(name)
    except Exception:  # noqa: BLE001 - no app / unknown key: no fault
        return None


def learned_probability(
    group_id: str,
    decision_type: str,
    design_state: Mapping,
    decision_time: datetime.datetime,
) -> PolicyDecision:
    """
    ``Pr(A = 1)`` for a learned decision (contract D10, D13): the policy in
    force at ``decision_time``, the dyad's posterior in it, and the Monte Carlo
    allocation at ``f`` of ``design_state``. Raises :class:`NoPolicy` when no
    version is in force or the dyad has no posterior; any other exception is
    a ``learner_error`` for the route.

    Fault injection (contract D21; only with ``FAULT_INJECTION_ENABLED``):
    ``FAULT_ACTION_CRASH`` makes this raise for the listed decision types and
    ``FAULT_FORCE_PI`` overwrites the returned ``action_prob``.
    """
    from app.clock import to_study_local
    from app.hyperparams import get_hyperparams

    F.check_decision_type(decision_type)
    crash = _fault("FAULT_ACTION_CRASH") or frozenset()
    if decision_type in crash:
        raise RuntimeError(f"injected learner crash for {decision_type} (FAULT_ACTION_CRASH)")

    hp = get_hyperparams()
    t = to_study_local(decision_time)
    _, f = F.hf_from_design_state(decision_type, design_state)
    version = policy_in_force(decision_type, t)
    if version is None:
        raise NoPolicy(f"no published {decision_type} policy is in force at {t.isoformat()}.")
    row = posterior_row(version, group_id)
    if row is None:
        raise NoPolicy(
            f"policy version {version.id} ({decision_type}, week {version.study_week_monday}) "
            f"has no posterior for {group_id}."
        )
    alpha, cov = posterior_arrays(row)
    if alpha.size != f.size:
        raise ValueError(f"posterior {row.id} has {alpha.size} coefficients but f has {f.size}.")
    pi, mu, v = probability_at(decision_type, f, alpha, cov, hp=hp)

    forced = _fault("FAULT_FORCE_PI")
    if forced is not None:
        pi = float(forced)

    return PolicyDecision(
        action_prob=float(pi),
        mu=float(mu),
        v=float(v),
        model_param=model_param_vector(alpha, cov),
        policy_version_id=int(version.id),
        policy_week=version.study_week_monday,
        model_parameters_id=int(row.id),
        hyperparams_hash=hp.sha256,
    )


__all__ = [
    "SNAPSHOT_LOCAL_FIT",
    "SNAPSHOT_HYPER",
    "SNAPSHOT_POSTERIOR",
    "PolicyDecision",
    "NoPolicy",
    "publication_cutoff",
    "policy_in_force",
    "week_policy",
    "posterior_row",
    "posterior_arrays",
    "probability_at",
    "model_param_vector",
    "learned_probability",
]
