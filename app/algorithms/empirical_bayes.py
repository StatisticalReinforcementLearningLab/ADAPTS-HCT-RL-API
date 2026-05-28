"""
Three-agent empirical Bayes learner for ADAPTS-HCT.

Implements the algorithm described in Study_Design/main.tex:
- three agent streams: AYA messages, care-partner messages, weekly game
- local per-dyad fits via discounted linear RLSVI (Algorithm 1)
- empirical Bayes pooling across dyads within each stream (Algorithm 3)
- posterior shrinkage for action selection
- warmup override: groups flagged `warmup=True` get a Bernoulli(0.5) action,
  bypassing the learner so their data can seed the EB prior
- per-agent discount factors and per-agent EB refresh cadence
- per-dyad week-1 standardization baselines applied at feature-build time

All randomness is consumed from a shared ``DeterministicSampleStream`` (a
pre-sampled buffer of standard normals + uniforms with a cursor). Every
consumption stamps the (start, end) cursor positions onto the row that
triggered it (Action.random_state for actions, EB snapshot metadata_json
for fits and perturbations) so the run is byte-for-byte reproducible from
the original buffer.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from app.algorithms.base import RLAlgorithm
from app.deterministic_sampler import (
    DeterministicSampleStream,
    closed_form_action_prob,
)
from app.extensions import db
from app.feature_builder import ProtocolRLFeatureBuilder, tailoring_mask
from app.logging_config import get_rl_logger
from app.models import ModelParameters, Group, StandardizationBaseline
from app.protocol import compute_reward, encode_state, validate_context, validate_outcome
from app.standardization import (
    compute_week1_baselines_for_dyad,
    fetch_baselines,
    filter_week1_records,
)


# Per-agent discount factors. main.tex Algorithm 2 calls Inf-LSVI with these.
GAMMA_BY_AGENT: dict[str, float] = {
    "aya_message": 13.0 / 14.0,
    "cp_message": 6.0 / 7.0,
    # REL is a contextual bandit: with only 14 weekly decisions per dyad,
    # the "one-week-ahead" horizon rule (γ such that 1/(1-γ) = decisions/week)
    # collapses to γ = 0. The Bellman target reduces to y_k = r_k.
    "dyad_game": 0.0,
}

# REL agent has only 14 weekly samples per dyad — pooling weekly is unstable
# early on. Refresh the EB hyperparameters every Nth update for REL; AYA/CP
# refresh on every update.
EB_REFRESH_EVERY: dict[str, int] = {
    "aya_message": 1,
    "cp_message": 1,
    "dyad_game": 4,
}

# Prior block scales (main.tex Appendix B Table 4). τ_m² on the main block,
# τ_x² on the interaction block; within the interaction block, prognostic
# features get τ_x²/2 per the prognostic-halving rule.
TAU_M_SQ_BY_AGENT: dict[str, float] = {
    "aya_message": 0.73,
    "cp_message": 0.20,
    # REL is fully-pooled (HybridRelPoolAlgorithm). τ=10.0 is uninformative
    # so the pooled data dominates the action coefficient from very early
    # in the trial; this gives the fastest possible kick-off for REL
    # commitment subject to the data quantity available.
    "dyad_game": 10.0,
}
TAU_X_SQ_BY_AGENT: dict[str, float] = {
    "aya_message": 0.073,
    "cp_message": 0.040,
    "dyad_game": 10.0,
}

# Probit-TS inverse temperature (main.tex §sec:probit-ts). Open knob; default 1.
ETA_BY_AGENT: dict[str, float] = {
    "aya_message": 1.0,
    "cp_message": 1.0,
    "dyad_game": 1.0,
}

# Anchor-shrinkage strength: α_N = N / (N + N_0). main.tex Appendix B.
N_0_ANCHOR = 10

SIGMA_NOISE = 1.0
MIN_COV_JITTER = 1e-6

# Legacy scalar prior. Retained for backward compatibility of any external
# imports; the live learner uses the block-diagonal Σ_0 from
# `_prior_covariance` below.
LAMBDA_PRIOR = 1.0


def _prior_covariance(decision_type: str) -> np.ndarray:
    """
    Block-diagonal Σ_0^g.

    Feature layout (shared missing indicator I across all variables):
        φ(s,a) = [1, a, I, (v_j*I)_{j=1..J}, a*I, (a*v_j*I)_{j=1..J}]
    Main block B_m: {intercept, action, I, v_j*I for each j} → variance τ_m².
    Interaction block B_x: {a*I, a*v_j*I for each j} → variance τ_x² for
    tailoring variables, τ_x²/2 for prognostic variables. The shared I
    interaction (a*I) gets the unweighted τ_x².
    """
    fb = ProtocolRLFeatureBuilder(decision_type)
    tau_m_sq = TAU_M_SQ_BY_AGENT[decision_type]
    tau_x_sq = TAU_X_SQ_BY_AGENT[decision_type]
    mask = tailoring_mask(decision_type)  # length n_vars
    # Main block: 1, a, I, then one v_j*I per variable
    diag = [tau_m_sq, tau_m_sq, tau_m_sq]  # intercept, action, I
    diag.extend([tau_m_sq] * fb.n_vars)    # v_j*I (one per variable)
    # Interaction block: a*I, then a*v_j*I per variable
    diag.append(tau_x_sq)                  # a*I (shared indicator)
    for j in range(fb.n_vars):
        diag.append(tau_x_sq if mask[j] else tau_x_sq / 2.0)
    return np.diag(np.asarray(diag, dtype=np.float64))


class ThreeAgentEmpiricalBayesAlgorithm(RLAlgorithm):
    def __init__(
        self,
        seed: int | None = None,
        app=None,
        sampler: DeterministicSampleStream | None = None,
    ):
        super().__init__(seed)
        self.logger = get_rl_logger()
        self.seed = seed
        self.app = app
        if sampler is None:
            raise ValueError(
                "ThreeAgentEmpiricalBayesAlgorithm requires a "
                "DeterministicSampleStream. Generate one with "
                "`flask init-buffer` and configure SAMPLE_BUFFER_PATH."
            )
        self.sampler = sampler
        # Per-agent counter of how many EB-refresh checkpoints have been seen,
        # used to gate REL refreshes to every Nth call.
        self._update_call_counts: dict[str, int] = defaultdict(int)
        self.logger.info(
            "Three-agent EB algorithm initialized "
            "(deterministic sampler: %d normals, %d uniforms; cursor=%s)",
            self.sampler.n_normals,
            self.sampler.n_uniforms,
            self.sampler.cursor(),
        )

    # ------------------------------------------------------------------ action

    def get_action(
        self,
        group_id: str,
        state,
        parameters: dict,
        decision_type: str,
        decision_idx: int,
    ) -> tuple[int, float, dict]:
        try:
            state_vec = np.asarray(state, dtype=np.float64)
            fb = ProtocolRLFeatureBuilder(decision_type)
            phi_dim = fb.phi_dim

            # Warmup override: purely-randomized actions during each dyad's
            # first 7 days of enrollment (~14 AYA / 7 CP / 1 REL decisions),
            # for the standardization baseline and to seed the EB pool.
            # Consumes ONE uniform primitive — recorded by cursor diff.
            if self._is_warmup(group_id, decision_type, decision_idx):
                cursor_start = self.sampler.cursor()
                action = int(self.sampler.draw_bernoulli(0.5))
                cursor_end = self.sampler.cursor()
                random_state = {
                    "mode": "warmup",
                    "sampler_cursor_start": cursor_start,
                    "sampler_cursor_end": cursor_end,
                }
                self.logger.info(
                    "EB warmup action=%d group_id=%s decision_type=%s "
                    "decision_idx=%d cursor=%s",
                    action,
                    group_id,
                    decision_type,
                    decision_idx,
                    cursor_end,
                )
                return action, 0.5, random_state

            posterior = self._load_latest_snapshot("posterior", decision_type, group_id=group_id)
            hyper = self._load_latest_snapshot("hyper", decision_type)

            if posterior is not None and len(posterior.theta) == phi_dim:
                mean = np.asarray(posterior.theta, dtype=np.float64)
                cov = np.asarray(posterior.covariance, dtype=np.float64)
                source = "posterior"
            elif hyper is not None and len(hyper.theta) == phi_dim:
                mean = np.asarray(hyper.theta, dtype=np.float64)
                cov = np.asarray(hyper.covariance, dtype=np.float64)
                source = "hyper"
            else:
                mean = np.zeros(phi_dim, dtype=np.float64)
                cov = _prior_covariance(decision_type)
                source = "prior"

            cov = self._stabilize_covariance(cov)

            # Probit-TS marginal allocation probability (closed form, η = eta).
            # No theta sampling — consumes ONE uniform primitive for the Bernoulli draw.
            eta = ETA_BY_AGENT.get(decision_type, 1.0)
            prob_action_1 = closed_form_action_prob(
                state_vec, mean, cov, fb.expand_base_to_phi, eta=eta
            )

            cursor_start = self.sampler.cursor()
            action = int(self.sampler.draw_bernoulli(prob_action_1))
            cursor_end = self.sampler.cursor()

            prob = prob_action_1 if action == 1 else (1.0 - prob_action_1)

            random_state = {
                "mode": "probit_ts",
                "source": source,
                "eta": eta,
                "sampler_cursor_start": cursor_start,
                "sampler_cursor_end": cursor_end,
            }

            self.logger.info(
                "EB action=%d group_id=%s decision_type=%s decision_idx=%d "
                "prob=%.6f source=%s cursor=%s",
                action,
                group_id,
                decision_type,
                decision_idx,
                prob,
                source,
                cursor_end,
            )
            return action, float(prob), random_state
        except Exception as exc:
            self.logger.error("Empirical Bayes action selection failed: %s", exc)
            raise

    # ------------------------------------------------------------------ update

    def update(self, old_params: dict, data: dict) -> tuple[bool, dict]:
        try:
            records = data.get("records", [])
            if not records:
                return True, {"probability_of_action": old_params.get("probability_of_action", 0.5)}

            grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
            for record in records:
                grouped[record["decision_type"]][record["group_id"]].append(record)

            for decision_type, group_records in grouped.items():
                self._update_call_counts[decision_type] += 1
                local_fits: dict[str, dict] = {}

                for group_id, rows in group_records.items():
                    ordered_rows = sorted(rows, key=lambda row: row["agent_decision_index"])

                    # Persist per-dyad week-1 baselines once enough data exist.
                    self._maybe_persist_baselines(group_id, decision_type, ordered_rows)
                    baselines = fetch_baselines(group_id, decision_type)

                    previous = self._load_latest_snapshot(
                        "local_fit", decision_type, group_id=group_id
                    )
                    fit_summary = self._fit_local_model(
                        decision_type, ordered_rows, previous, baselines
                    )
                    fit_summary["agent_decision_index"] = ordered_rows[-1]["agent_decision_index"]
                    local_fits[group_id] = fit_summary
                    self._save_snapshot(
                        snapshot_type="local_fit",
                        decision_type=decision_type,
                        agent_decision_index=fit_summary["agent_decision_index"],
                        group_id=group_id,
                        sample_size=fit_summary["sample_size"],
                        theta=fit_summary["theta_hat"].tolist(),
                        covariance=fit_summary["covariance"].tolist(),
                        perturbation=None,
                        metadata_json={
                            "update_decision_idx": ordered_rows[-1]["decision_idx"],
                            "sampler_cursor_start": fit_summary["sampler_cursor_start"],
                            "sampler_cursor_end": fit_summary["sampler_cursor_end"],
                        },
                    )

                max_agent_index = max(fit["agent_decision_index"] for fit in local_fits.values())
                if self._is_eb_refresh_point(decision_type):
                    eb_mean, eb_cov = self._estimate_hyperparameters(local_fits, decision_type)
                    self._save_snapshot(
                        snapshot_type="hyper",
                        decision_type=decision_type,
                        agent_decision_index=max_agent_index,
                        group_id=None,
                        sample_size=len(local_fits),
                        theta=eb_mean.tolist(),
                        covariance=eb_cov.tolist(),
                        perturbation=None,
                        metadata_json={
                            "active_groups": len(local_fits),
                            "update_call_index": self._update_call_counts[decision_type],
                        },
                    )
                else:
                    prev_hyper = self._load_latest_snapshot("hyper", decision_type)
                    if prev_hyper is None:
                        # First update for REL, no hyper yet; build one to keep
                        # downstream shrinkage well-defined.
                        eb_mean, eb_cov = self._estimate_hyperparameters(local_fits, decision_type)
                        self._save_snapshot(
                            snapshot_type="hyper",
                            decision_type=decision_type,
                            agent_decision_index=max_agent_index,
                            group_id=None,
                            sample_size=len(local_fits),
                            theta=eb_mean.tolist(),
                            covariance=eb_cov.tolist(),
                            perturbation=None,
                            metadata_json={
                                "active_groups": len(local_fits),
                                "update_call_index": self._update_call_counts[decision_type],
                                "bootstrap": True,
                            },
                        )
                    else:
                        eb_mean = np.asarray(prev_hyper.theta, dtype=np.float64)
                        eb_cov = np.asarray(prev_hyper.covariance, dtype=np.float64)

                for group_id, fit_summary in local_fits.items():
                    post_mean, post_cov = self._shrink_to_hyperprior(
                        fit_summary["theta_hat"],
                        fit_summary["covariance"],
                        eb_mean,
                        eb_cov,
                    )
                    self._save_snapshot(
                        snapshot_type="posterior",
                        decision_type=decision_type,
                        agent_decision_index=fit_summary["agent_decision_index"],
                        group_id=group_id,
                        sample_size=fit_summary["sample_size"],
                        theta=post_mean.tolist(),
                        covariance=post_cov.tolist(),
                        perturbation=None,
                        metadata_json={"update_decision_idx": fit_summary["decision_idx"]},
                    )

            return True, {"probability_of_action": old_params.get("probability_of_action", 0.5)}
        except Exception as exc:
            self.logger.error("Empirical Bayes update error: %s", exc)
            return False, old_params

    # ------------------------------------------------------------- delegation

    def make_state(self, context: dict) -> tuple[bool, list]:
        decision_type = context.get("decision_type")
        valid, error_message = validate_context(decision_type, context)
        if not valid:
            return False, error_message
        group_id = context.get("group_id")
        baselines = None
        if group_id is not None:
            baselines = fetch_baselines(group_id, decision_type) or None
        return True, encode_state(decision_type, context, baselines=baselines)

    def make_reward(self, user_id: str, state, action: int, outcome: dict) -> tuple[bool, float]:
        decision_type = outcome.get("decision_type")
        valid, error_message = validate_outcome(decision_type, outcome)
        if not valid:
            return False, error_message
        return True, compute_reward(decision_type, action, outcome)

    # ----------------------------------------------------------------- helpers

    # Per-dyad week-1 warm-up counts (~7 days for each agent's cadence).
    # Replaces the earlier cohort-level "first 5 dyads" warmup.
    _WARMUP_DECISIONS = {
        "aya_message": 14,   # 2/day × 7 days
        "cp_message": 7,     # 1/day × 7 days
        "dyad_game": 1,      # weekly: only the first weekly decision is random
    }

    def _is_warmup(self, group_id: str, decision_type: str, decision_idx: int) -> bool:
        threshold = self._WARMUP_DECISIONS.get(decision_type, 7)
        return decision_idx < threshold

    def _is_eb_refresh_point(self, decision_type: str) -> bool:
        every = EB_REFRESH_EVERY.get(decision_type, 1)
        return (self._update_call_counts[decision_type] % every) == 0

    def _maybe_persist_baselines(
        self, group_id: str, decision_type: str, ordered_rows: list[dict]
    ) -> None:
        existing = (
            StandardizationBaseline.query
            .filter_by(group_id=group_id, decision_type=decision_type)
            .first()
        )
        if existing is not None:
            return
        week1 = filter_week1_records(ordered_rows)
        if not week1:
            return
        compute_week1_baselines_for_dyad(group_id, decision_type, week1)

    def _fit_local_model(
        self,
        decision_type: str,
        records: list[dict],
        previous_snapshot,
        baselines: dict[str, dict[str, float]] | None,
    ) -> dict:
        fb = ProtocolRLFeatureBuilder(decision_type)
        feature_dim = fb.phi_dim
        base_dim = fb.base_dim
        state_dim = len(records[0].get("state", []))
        prev_theta = np.zeros(feature_dim, dtype=np.float64)
        gamma = GAMMA_BY_AGENT.get(decision_type, 0.9)
        prior_cov = _prior_covariance(decision_type)
        prior_precision = np.linalg.inv(prior_cov)

        if previous_snapshot is not None and previous_snapshot.feature_dim == feature_dim:
            prev_theta = np.asarray(previous_snapshot.theta, dtype=np.float64)

        x_rows = []
        y_vals = []
        for idx, record in enumerate(records):
            raw = record["raw_context"]
            phi = fb.phi(raw, int(record["action"]), baselines=baselines)
            reward = float(record["reward"])
            if idx + 1 < len(records):
                nxt = records[idx + 1]["raw_context"]
                phi_n0 = fb.phi(nxt, 0, baselines=baselines)
                phi_n1 = fb.phi(nxt, 1, baselines=baselines)
                q0 = float(phi_n0 @ prev_theta)
                q1 = float(phi_n1 @ prev_theta)
                target = reward + (gamma * max(q0, q1))
            else:
                target = reward
            x_rows.append(phi)
            y_vals.append(target)

        x_mat = np.vstack(x_rows)
        y_vec = np.asarray(y_vals, dtype=np.float64)
        # Inf-LSVI Bayesian linear regression: Σ⁻¹ = Σ_0⁻¹ + Xᵀ X / σ²
        precision = (x_mat.T @ x_mat) / (SIGMA_NOISE**2) + prior_precision
        cov = np.linalg.inv(precision)
        # Prior mean is θ_0 = 0 (main.tex), so mean = Σ Xᵀ y / σ².
        theta_hat = cov @ ((x_mat.T @ y_vec) / (SIGMA_NOISE**2))

        return {
            "sample_size": len(records),
            "state_dim": state_dim,
            "base_dim": base_dim,
            "feature_dim": feature_dim,
            "theta_hat": theta_hat,
            "covariance": cov,
            "perturbation": None,
            "decision_idx": records[-1]["decision_idx"],
            "sampler_cursor_start": self.sampler.cursor(),
            "sampler_cursor_end": self.sampler.cursor(),
        }

    def _estimate_hyperparameters(
        self, local_fits: dict[str, dict], decision_type: str
    ) -> tuple[np.ndarray, np.ndarray]:
        thetas = [fit["theta_hat"] for fit in local_fits.values()]
        covariances = [fit["covariance"] for fit in local_fits.values()]
        feature_dim = thetas[0].shape[0]
        n_dyads = len(thetas)

        weighted_precision = np.zeros((feature_dim, feature_dim), dtype=np.float64)
        weighted_mean_term = np.zeros(feature_dim, dtype=np.float64)
        for theta_hat, cov in zip(thetas, covariances):
            cov = self._stabilize_covariance(cov)
            inv_cov = np.linalg.inv(cov)
            weighted_precision += inv_cov
            weighted_mean_term += inv_cov @ theta_hat

        eb_cov_mean = np.linalg.inv(weighted_precision + MIN_COV_JITTER * np.eye(feature_dim))
        eb_mean = eb_cov_mean @ weighted_mean_term

        centered_cov = np.zeros((feature_dim, feature_dim), dtype=np.float64)
        for theta_hat in thetas:
            delta = theta_hat - eb_mean
            centered_cov += np.outer(delta, delta)
        centered_cov /= max(n_dyads, 1)

        avg_local_cov = sum(covariances) / max(n_dyads, 1)
        mom_cov = centered_cov - avg_local_cov

        # Anchor shrinkage (main.tex Algorithm 3 + Appendix B):
        # Σ̂_0 = α_N · MoM + (1 - α_N) · Σ_0^anchor, α_N = N / (N + N_0).
        anchor = _prior_covariance(decision_type)
        alpha = n_dyads / (n_dyads + N_0_ANCHOR)
        eb_cov = alpha * mom_cov + (1.0 - alpha) * anchor
        eb_cov = self._stabilize_covariance(eb_cov)
        return eb_mean, eb_cov

    def _shrink_to_hyperprior(
        self,
        local_theta: np.ndarray,
        local_cov: np.ndarray,
        eb_mean: np.ndarray,
        eb_cov: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        local_cov = self._stabilize_covariance(local_cov)
        eb_cov = self._stabilize_covariance(eb_cov)
        precision = np.linalg.inv(local_cov) + np.linalg.inv(eb_cov)
        post_cov = np.linalg.inv(precision)
        post_mean = post_cov @ (
            (np.linalg.inv(local_cov) @ local_theta) + (np.linalg.inv(eb_cov) @ eb_mean)
        )
        return post_mean, self._stabilize_covariance(post_cov)

    def _save_snapshot(
        self,
        snapshot_type: str,
        decision_type: str,
        agent_decision_index: int,
        group_id: str | None,
        sample_size: int,
        theta: list[float],
        covariance: list[list[float]],
        perturbation: list[float] | None,
        metadata_json: dict | None,
    ):
        if self.app is None:
            return
        with self.app.app_context():
            snapshot = ModelParameters(
                snapshot_type=snapshot_type,
                group_id=group_id,
                decision_type=decision_type,
                agent_decision_index=agent_decision_index,
                sample_size=sample_size,
                feature_dim=len(theta),
                theta=theta,
                covariance=covariance,
                perturbation=perturbation,
                metadata_json=metadata_json,
            )
            db.session.add(snapshot)
            db.session.commit()

    def _load_latest_snapshot(
        self, snapshot_type: str, decision_type: str, group_id: str | None = None
    ):
        if self.app is None:
            return None
        with self.app.app_context():
            query = ModelParameters.query.filter_by(
                snapshot_type=snapshot_type,
                decision_type=decision_type,
                group_id=group_id,
            ).order_by(ModelParameters.agent_decision_index.desc())
            return query.first()

    def _stabilize_covariance(self, cov: np.ndarray) -> np.ndarray:
        cov = np.asarray(cov, dtype=np.float64)
        cov = (cov + cov.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, MIN_COV_JITTER)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T
