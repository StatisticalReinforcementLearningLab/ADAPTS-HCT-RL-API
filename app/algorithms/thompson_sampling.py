"""
Thompson Sampling algorithm for ADAPTS-HCT.

Each (group_id, decision_type) pair has its own independent Thompson Sampling bandit.
- Reward: first entry of context (cur_var) or from outcome
- State: other entries (past3_vars) used as context for the linear model

Uses Bayesian linear regression: E[r|a,x] = x^T theta_a with Gaussian prior.
"""

import numpy as np
from app.algorithms.base import RLAlgorithm
from app.logging_config import get_rl_logger
from app.extensions import db
from app.models import ThompsonSamplingParams

# State dimension: [1, past3_vars[0], past3_vars[1], past3_vars[2]]
STATE_DIM = 4
LAMBDA_PRIOR = 1.0  # Prior variance scaling
SIGMA_NOISE = 1.0  # Observation noise std


def _default_params():
    """Default prior for each action: N(0, lambda*I)."""
    return {
        "action_0": {
            "n": 0,
            "sum_xx": [[0.0] * STATE_DIM for _ in range(STATE_DIM)],
            "sum_xr": [0.0] * STATE_DIM,
        },
        "action_1": {
            "n": 0,
            "sum_xx": [[0.0] * STATE_DIM for _ in range(STATE_DIM)],
            "sum_xr": [0.0] * STATE_DIM,
        },
    }


def _state_vector(context: dict) -> np.ndarray:
    """Build state vector x = [1, past3_vars[0], past3_vars[1], past3_vars[2]]."""
    past3 = context.get("past3_vars", [0.0, 0.0, 0.0])
    past3 = list(past3)[:3]
    while len(past3) < 3:
        past3.append(0.0)
    return np.array([1.0, float(past3[0]), float(past3[1]), float(past3[2])], dtype=np.float64)


def _posterior_mean_cov(params: dict, action: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute posterior mean and covariance for the given action."""
    pa = params[f"action_{action}"]
    n = pa["n"]
    sum_xx = np.array(pa["sum_xx"])
    sum_xr = np.array(pa["sum_xr"])

    prior_prec = (1.0 / LAMBDA_PRIOR) * np.eye(STATE_DIM)
    lik_prec = (1.0 / (SIGMA_NOISE**2)) * sum_xx if n > 0 else np.zeros((STATE_DIM, STATE_DIM))
    post_prec = prior_prec + lik_prec
    post_cov = np.linalg.inv(post_prec)

    prior_term = prior_prec @ np.zeros(STATE_DIM)
    lik_term = (1.0 / (SIGMA_NOISE**2)) * sum_xr if n > 0 else np.zeros(STATE_DIM)
    post_mean = post_cov @ (prior_term + lik_term)

    return post_mean, post_cov


def _sample_theta(params: dict, action: int, rng: np.random.Generator) -> np.ndarray:
    """Sample theta from posterior for the given action."""
    mu, cov = _posterior_mean_cov(params, action)
    return rng.multivariate_normal(mu, cov)


def _update_params(params: dict, action: int, x: np.ndarray, r: float) -> dict:
    """Update sufficient statistics for the given action."""
    pa = params[f"action_{action}"].copy()
    pa["n"] = pa["n"] + 1
    xx = np.outer(x, x)
    pa["sum_xx"] = (np.array(pa["sum_xx"]) + xx).tolist()
    pa["sum_xr"] = (np.array(pa["sum_xr"]) + x * r).tolist()
    params[f"action_{action}"] = pa
    return params


def _prob_action_1(params: dict, x: np.ndarray, rng: np.random.Generator, n_samples: int = 100) -> float:
    """Estimate P(action=1 is better) via sampling."""
    wins = 0
    for _ in range(n_samples):
        theta_0 = _sample_theta(params, 0, rng)
        theta_1 = _sample_theta(params, 1, rng)
        if x @ theta_1 > x @ theta_0:
            wins += 1
    return wins / n_samples


class ThompsonSamplingAlgorithm(RLAlgorithm):
    """
    Thompson Sampling with linear context: each (group_id, decision_type) has its own bandit.
    State = past3_vars (with intercept), reward = cur_var or from outcome.
    """

    def __init__(self, seed: int = None, app=None):
        super().__init__(seed)
        self.logger = get_rl_logger()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.app = app
        self.logger.info("Thompson Sampling algorithm initialized.")

    def _get_params(self, group_id: str, decision_type: str) -> dict:
        """Get TS params for (group_id, decision_type), or default."""
        if self.app is None:
            return _default_params()
        with self.app.app_context():
            row = ThompsonSamplingParams.query.filter_by(
                group_id=group_id, decision_type=decision_type
            ).first()
            if row is None:
                return _default_params()
            return row.params

    def _save_params(self, group_id: str, decision_type: str, params: dict):
        """Save TS params for (group_id, decision_type)."""
        if self.app is None:
            return
        with self.app.app_context():
            row = ThompsonSamplingParams.query.filter_by(
                group_id=group_id, decision_type=decision_type
            ).first()
            if row is None:
                row = ThompsonSamplingParams(group_id=group_id, decision_type=decision_type, params=params)
                db.session.add(row)
            else:
                row.params = params
                row.updated_at = __import__("datetime").datetime.now()
            db.session.commit()

    def get_action(
        self, group_id: str, state, parameters: dict, decision_type: str, decision_idx: int
    ) -> tuple[int, float, dict]:
        """Sample action from Thompson Sampling posterior."""
        self.logger.info(
            "Getting action for group_id=%s decision_type=%s decision_idx=%d",
            group_id, decision_type, decision_idx,
        )
        rng_state = self.rng.bit_generator.state

        # State from make_state: list of past3_vars [p0, p1, p2] -> x = [1, p0, p1, p2]
        if isinstance(state, (list, tuple)) and len(state) >= 3:
            x = np.array([1.0, float(state[0]), float(state[1]), float(state[2])], dtype=np.float64)
        elif isinstance(state, dict):
            x = _state_vector(state)
        else:
            x = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        params = self._get_params(group_id, decision_type)

        theta_0 = _sample_theta(params, 0, self.rng)
        theta_1 = _sample_theta(params, 1, self.rng)
        val_0 = float(x @ theta_0)
        val_1 = float(x @ theta_1)

        action = 1 if val_1 > val_0 else 0
        prob_a1 = _prob_action_1(params, x, self.rng)
        prob = prob_a1 if action == 1 else (1.0 - prob_a1)

        self.logger.info(
            "TS action=%d for group_id=%s decision_type=%s prob=%.3f",
            action, group_id, decision_type, prob,
        )
        return action, float(prob), rng_state

    def update(self, old_params: dict, data: dict) -> tuple[bool, dict]:
        """Update each (group_id, decision_type) bandit with its data."""
        try:
            group_ids = data.get("group_ids", [])
            decision_types = data.get("decision_types", [])
            past3_vars_list = data.get("past3_vars", [])
            rewards = data.get("rewards", [])

            if not group_ids:
                return True, {"probability_of_action": old_params.get("probability_of_action", 0.5)}

            # Group updates by (group_id, decision_type)
            updates = {}
            for i in range(len(group_ids)):
                gid = group_ids[i]
                dtype = decision_types[i] if i < len(decision_types) else "aya_message"
                past3 = past3_vars_list[i] if i < len(past3_vars_list) else [0, 0, 0]
                r = rewards[i] if i < len(rewards) else 0.0

                # Need action for this row - data should include actions
                actions = data.get("actions", [])
                action = actions[i] if i < len(actions) else 0

                key = (gid, dtype)
                if key not in updates:
                    updates[key] = self._get_params(gid, dtype)

                x = np.array([1.0, float(past3[0]) if len(past3) > 0 else 0,
                              float(past3[1]) if len(past3) > 1 else 0,
                              float(past3[2]) if len(past3) > 2 else 0], dtype=np.float64)
                updates[key] = _update_params(updates[key], action, x, float(r))

            for (gid, dtype), params in updates.items():
                self._save_params(gid, dtype, params)

            return True, {"probability_of_action": old_params.get("probability_of_action", 0.5)}

        except Exception as e:
            self.logger.error("Thompson Sampling update error: %s", e)
            return False, old_params

    def make_state(self, context: dict) -> tuple[bool, list]:
        """State = past3_vars (other entries); first entry cur_var is reward."""
        try:
            past3 = context.get("past3_vars", [0, 0, 0])
            past3 = list(past3)[:3]
            while len(past3) < 3:
                past3.append(0.0)
            state = [float(x) for x in past3]
            return True, state
        except Exception as e:
            self.logger.error("make_state error: %s", e)
            return False, []

    def make_reward(self, user_id: str, state, action: int, outcome: dict) -> tuple[bool, float]:
        """Reward: first entry of context (cur_var) or outcome clicks."""
        try:
            if "reward" in outcome:
                return True, float(outcome["reward"])
            if "cur_var" in outcome:
                return True, float(outcome["cur_var"])
            if "clicks" in outcome:
                return True, float(outcome["clicks"])
            return True, 0.0
        except Exception as e:
            self.logger.error("make_reward error: %s", e)
            return False, 0.0
