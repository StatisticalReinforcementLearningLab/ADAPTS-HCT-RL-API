"""
Random-baseline RL algorithm: Bernoulli(0.5) on every decision regardless
of state. Reuses the production protocol's state encoding and reward
construction so the StudyData rows it produces are identical-shape to the
ones produced by the empirical_bayes learner.

Used as the comparison policy for the sanity-check experiment (see
``/tmp/adapts_run/sanity_check.py``).
"""

from __future__ import annotations

import numpy as np

from app.algorithms.base import RLAlgorithm
from app.logging_config import get_rl_logger
from app.protocol import (
    compute_reward, encode_state, validate_context, validate_outcome,
)
from app.standardization import fetch_baselines


class RandomBaselineAlgorithm(RLAlgorithm):
    """Bernoulli(0.5) sampler. Ignores state. State / reward use the
    production pipeline so row schemas match EB."""

    def __init__(self, seed: int | None = None):
        super().__init__(seed)
        self.logger = get_rl_logger()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.logger.info("RandomBaseline (Bernoulli 0.5) initialized.")

    def get_action(self, group_id, state, parameters, decision_type, decision_idx):
        rng_state = self.rng.bit_generator.state
        action = int(self.rng.binomial(1, 0.5))
        return action, 0.5, rng_state

    def update(self, old_params, data):
        # No-op update; preserve parameters unchanged.
        return True, old_params

    def make_state(self, context):
        decision_type = context.get("decision_type")
        valid, msg = validate_context(decision_type, context)
        if not valid:
            return False, msg
        group_id = context.get("group_id")
        baselines = fetch_baselines(group_id, decision_type) if group_id else None
        return True, encode_state(decision_type, context, baselines=baselines)

    def make_reward(self, user_id, state, action, outcome):
        decision_type = outcome.get("decision_type")
        valid, msg = validate_outcome(decision_type, outcome)
        if not valid:
            return False, msg
        return True, compute_reward(decision_type, action, outcome)
