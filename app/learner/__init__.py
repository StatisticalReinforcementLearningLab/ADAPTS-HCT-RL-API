"""
The ``inf_lsvi_eb`` learner (main_new.tex Sections 3-5, Algorithms 1-4;
implementation contract D6-D16, §3.1).

Interface fixed by the implementation contract §3.1. The routes call these
functions directly when ``RL_ALGORITHM == "inf_lsvi_eb"``:

- ``/action``: :func:`build_design_state` -> :func:`standardized_g` ->
  :func:`learned_probability` (raises :class:`NoPolicy` for the ``no_policy``
  fallback; any other exception is the ``learner_error`` fallback).
- ``/update``: the reward derivation pairs each decision with its outcome
  upload through :func:`find_outcome_upload` and scores it with
  :func:`compute_reward` (REL with ``w_adh > 0`` also needs
  :func:`aya_week_mean_level`); then :func:`run_weekly_update`.

Modules:

- :mod:`app.learner.features` — design coordinates, ranges, standardization,
  h/f split (D7);
- :mod:`app.learner.state` — design state from an upload snapshot (D7);
- :mod:`app.learner.rewards` — rewards and outcome pairing (D6);
- :mod:`app.learner.local_fit` — Inf-LSVI local fit (D8);
- :mod:`app.learner.eb` — empirical-Bayes pooling and plug-in posterior (D9);
- :mod:`app.learner.allocation` — Monte Carlo allocation (D10);
- :mod:`app.learner.validity` — the C3 publication gate (Monitoring-Spec C3);
- :mod:`app.learner.policy` — the policy in force and ``learned_probability`` (D13);
- :mod:`app.learner.update` — the weekly update (D14, D15).
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

from app.hyperparams import get_hyperparams as _get_hyperparams
from app.learner import features as _features
from app.learner import policy as _policy
from app.learner import rewards as _rewards
from app.learner import state as _state
from app.learner import update as _update
from app.learner.policy import NoPolicy, PolicyDecision, policy_in_force
from app.learner.rewards import action_slot, find_outcome_upload, is_missing
from app.learner.state import aya_week_mean_level

if TYPE_CHECKING:  # pragma: no cover
    from app.hyperparams import Hyperparams
    from app.models import DataUpload

AGENTS = ("aya_message", "cp_message", "dyad_game")


def get_hyperparams() -> "Hyperparams":
    """The running app's resolved hyperparameter set (thin re-export of ``app.hyperparams``)."""
    return _get_hyperparams()


def build_design_state(
    group_id: str,
    decision_type: str,
    upload: "DataUpload",
    decision_time: datetime.datetime,
) -> dict:
    """
    Raw design state of contract D7 from the snapshot ``upload``. Returns
    ``{"names": [...], "raw": [...], "upload_id", "upload_rid",
    "imputed": {...}, "clipped": [...]}`` plus bookkeeping keys
    (``decision_type``, ``upload_timestamp``, ``slot``, ``week_in_study``,
    ``day_in_study``; REL also ``aya_week_mean_level_n``). Store it as
    ``actions.raw_context["design_state"]`` for every decision built from a
    snapshot, warm-up included, so the decision can enter the fit.
    """
    return _state.build_design_state(group_id, decision_type, upload, decision_time)


def standardized_g(decision_type: str, design_state: dict) -> list[float]:
    """Standardized full ``g`` (contract D7; main_new.tex eq. (range-std)), clipped to [-1, 1]."""
    return _features.standardized_g(decision_type, design_state)


def compute_reward(
    decision_type: str,
    outcome_upload_data: dict,
    *,
    aya_week_mean_level: float | None = None,
) -> tuple[float, float]:
    """Reward of contract D6 from the outcome upload's ``data``: ``(normalized, native)``."""
    return _rewards.compute_reward(decision_type, outcome_upload_data, aya_week_mean_level=aya_week_mean_level)


def aya_level(previous_med_adherence) -> int:
    """AYA reward level (contract D6): 2 if adherence == 1, 1 if 0, 0 if "miss"/null."""
    return _rewards.aya_level(previous_med_adherence)


def learned_probability(
    group_id: str,
    decision_type: str,
    design_state: dict,
    decision_time: datetime.datetime,
) -> PolicyDecision:
    """
    Pr(A = 1) under the policy in force (contract D13) by the Monte Carlo
    allocation (contract D10). Raises :class:`NoPolicy`; any other exception
    is a ``learner_error``.
    """
    return _policy.learned_probability(group_id, decision_type, design_state, decision_time)


def run_weekly_update(
    update_rid: str,
    request_timestamp: datetime.datetime,
    started_wall: datetime.datetime,
) -> dict:
    """
    One weekly update (contract D14): REL, then AYA, then CP; local fits, EB,
    posteriors, C3 gate, atomic publish. Returns
    ``{"study_week_monday", "agents": {...}, "status", ...}`` (see
    :func:`app.learner.update.run_weekly_update` for every key).
    """
    return _update.run_weekly_update(update_rid, request_timestamp, started_wall)


__all__ = [
    "AGENTS",
    "get_hyperparams",
    "PolicyDecision",
    "NoPolicy",
    "build_design_state",
    "standardized_g",
    "compute_reward",
    "aya_level",
    "learned_probability",
    "run_weekly_update",
    "find_outcome_upload",
    "action_slot",
    "aya_week_mean_level",
    "is_missing",
    "policy_in_force",
]
