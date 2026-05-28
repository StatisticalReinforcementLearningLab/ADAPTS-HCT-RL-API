"""
Always-none RL algorithm: action=0 with probability 1.0 on every decision.
Mirror of AlwaysSendAlgorithm for the opposite oracle in the sanity-check
comparison. Used as a lower-bound baseline when "never send" is the
worst-case fixed policy.
"""

from __future__ import annotations

from app.algorithms.base import RLAlgorithm
from app.logging_config import get_rl_logger
from app.protocol import (
    compute_reward, encode_state, validate_context, validate_outcome,
)
from app.standardization import fetch_baselines


class AlwaysNoneAlgorithm(RLAlgorithm):
    """Returns (action=0, action_prob=1.0) on every call."""

    def __init__(self, seed: int | None = None):
        super().__init__(seed)
        self.logger = get_rl_logger()
        self.seed = seed
        self.logger.info("AlwaysNone (action=0) initialized.")

    def get_action(self, group_id, state, parameters, decision_type, decision_idx):
        return 0, 1.0, {"mode": "always_none"}

    def update(self, old_params, data):
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
