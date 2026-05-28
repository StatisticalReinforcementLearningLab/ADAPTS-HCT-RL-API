"""
Hybrid learner: EB-Gradient for AYA/CP, fully-pooled Inf-LSVI for REL.

Per-agent rationale:
  - AYA (168 obs/dyad) and CP (84 obs/dyad) have enough per-dyad data
    that the EB-Gradient partial-pooling estimator works well.
  - REL has only ~14 weekly obs/dyad against D ≈ 44 features ---
    severely under-determined for any per-dyad fit. The cleanest
    estimator is a single fully-pooled Inf-LSVI fit on the
    concatenation of all dyads' REL trajectories, with a deliberately
    loose cold-start prior (τ = 10) so the prior is uninformative
    against the cohort-pooled sample size.

Implementation: dispatch every method by ``decision_type``. Holds
instances of both learners and forwards each call to the right one.
"""

from __future__ import annotations

import numpy as np

from app.algorithms.base import RLAlgorithm
from app.algorithms.eb_gradient import ThreeAgentEmpiricalBayesGradientAlgorithm
from app.algorithms.inf_lsvi_pool import ThreeAgentInfLsviPooledAlgorithm
from app.deterministic_sampler import DeterministicSampleStream
from app.logging_config import get_rl_logger


_POOL_AGENTS = frozenset({"dyad_game"})


class HybridRelPoolAlgorithm(RLAlgorithm):
    """EB-Gradient for AYA/CP + Inf-LSVI fully-pooled for REL."""

    def __init__(
        self,
        seed: int | None = None,
        app=None,
        sampler: DeterministicSampleStream | None = None,
    ):
        super().__init__(seed)
        self.logger = get_rl_logger()
        self.app = app
        self.sampler = sampler
        # Both learners share the same sampler so action-Bernoulli draws
        # stay byte-for-byte reproducible.
        self.eb = ThreeAgentEmpiricalBayesGradientAlgorithm(
            seed=seed, app=app, sampler=sampler
        )
        self.pool = ThreeAgentInfLsviPooledAlgorithm(
            seed=seed, app=app, sampler=sampler
        )
        self.logger.info(
            "HybridRelPool initialized: EB-Gradient for {AYA, CP}, "
            "fully-pooled Inf-LSVI for REL (pool agents: %s)",
            sorted(_POOL_AGENTS),
        )

    # ------------------------------------------------------------ dispatch

    def _route(self, decision_type: str) -> RLAlgorithm:
        return self.pool if decision_type in _POOL_AGENTS else self.eb

    def get_action(self, group_id, state, parameters, decision_type, decision_idx):
        return self._route(decision_type).get_action(
            group_id, state, parameters, decision_type, decision_idx
        )

    def update(self, old_params, data):
        # Split records by decision_type and dispatch each batch to the
        # appropriate learner. Both learners are designed to handle a
        # mixed-agent records list themselves, but routing per-agent keeps
        # each learner's hyper-snapshot table consistent.
        records = data.get("records", [])
        if not records:
            ok, params = self.eb.update(old_params, data)
            return ok, params

        by_route: dict[str, list] = {"eb": [], "pool": []}
        for r in records:
            key = "pool" if r.get("decision_type") in _POOL_AGENTS else "eb"
            by_route[key].append(r)

        params = old_params
        ok_overall = True
        if by_route["eb"]:
            ok, params = self.eb.update(
                params, {"records": by_route["eb"],
                         "current_index": data.get("current_index", {})}
            )
            ok_overall = ok_overall and ok
        if by_route["pool"]:
            ok, params = self.pool.update(
                params, {"records": by_route["pool"],
                         "current_index": data.get("current_index", {})}
            )
            ok_overall = ok_overall and ok
        return ok_overall, params

    def make_state(self, context):
        return self._route(context.get("decision_type")).make_state(context)

    def make_reward(self, user_id, state, action, outcome):
        return self._route(outcome.get("decision_type")).make_reward(
            user_id, state, action, outcome
        )
