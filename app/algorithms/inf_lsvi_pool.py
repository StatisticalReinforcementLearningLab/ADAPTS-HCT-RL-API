"""
Fully-pooled Inf-LSVI baseline: one shared Bayesian linear $Q$ per agent
fit on the concatenation of every dyad's (s, a, r) tuples — the opposite
end of the pooling spectrum from `inf_lsvi_local.py`.

Same Inf-LSVI mechanics (Bayesian linear regression on Bellman targets,
structural cold-start prior as the regularizer) and same generalized-
logistic smooth allocation as `eb_gradient.py`, but no per-dyad split:
every action selection uses the same population-level posterior. This
is the natural "full pooling" baseline that sits opposite the per-dyad
"no pooling" baseline in `inf_lsvi_local.py`.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from app.algorithms.base import RLAlgorithm
from app.algorithms.empirical_bayes import (
    GAMMA_BY_AGENT,
    MIN_COV_JITTER,
    SIGMA_NOISE,
    _prior_covariance,
)
from app.algorithms.eb_gradient import (
    DEFAULT_B,
    DEFAULT_C,
    DEFAULT_K,
    DEFAULT_LMAX,
    DEFAULT_LMIN,
    DEFAULT_MC_SAMPLES,
    DEFAULT_MC_SEED,
    smooth_allocation_prob,
)
from app.deterministic_sampler import DeterministicSampleStream
from app.extensions import db
from app.feature_builder import ProtocolRLFeatureBuilder
from app.logging_config import get_rl_logger
from app.models import ModelParameters, StandardizationBaseline
from app.protocol import compute_reward, encode_state, validate_context, validate_outcome
from app.standardization import (
    compute_week1_baselines_for_dyad,
    fetch_baselines,
    filter_week1_records,
)


class ThreeAgentInfLsviPooledAlgorithm(RLAlgorithm):
    """Fully-pooled Inf-LSVI: one shared posterior per agent across all dyads."""

    _WARMUP_DECISIONS = {
        "aya_message": 14,
        "cp_message": 7,
        "dyad_game": 1,
    }

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
                "ThreeAgentInfLsviPooledAlgorithm requires a "
                "DeterministicSampleStream."
            )
        self.sampler = sampler

        cfg = app.config if app is not None else {}
        self.lmin = float(cfg.get("SMOOTH_ALLOC_LMIN", DEFAULT_LMIN))
        self.lmax = float(cfg.get("SMOOTH_ALLOC_LMAX", DEFAULT_LMAX))
        self.c = float(cfg.get("SMOOTH_ALLOC_C", DEFAULT_C))
        self.b = float(cfg.get("SMOOTH_ALLOC_B", DEFAULT_B))
        self.k = float(cfg.get("SMOOTH_ALLOC_K", DEFAULT_K))
        n_mc = int(cfg.get("SMOOTH_ALLOC_MC_SAMPLES", DEFAULT_MC_SAMPLES))
        mc_seed = int(cfg.get("SMOOTH_ALLOC_MC_SEED", DEFAULT_MC_SEED))
        rng = np.random.default_rng(mc_seed)
        self.z_bank = rng.standard_normal(n_mc).astype(np.float64)

        self.logger.info(
            "Inf-LSVI (full pooling) algorithm initialized "
            "(ρ=GenLogistic(Lmin=%.2f, Lmax=%.2f, c=%.2f, b=%.3f, k=%.2f); "
            "MC samples=%d, seed=%d; sampler: %d normals, %d uniforms; cursor=%s)",
            self.lmin, self.lmax, self.c, self.b, self.k,
            n_mc, mc_seed,
            self.sampler.n_normals, self.sampler.n_uniforms,
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

            if self._is_warmup(group_id, decision_type, decision_idx):
                cursor_start = self.sampler.cursor()
                action = int(self.sampler.draw_bernoulli(0.5))
                cursor_end = self.sampler.cursor()
                self.logger.info(
                    "ILP warmup action=%d group_id=%s decision_type=%s "
                    "decision_idx=%d cursor=%s",
                    action, group_id, decision_type, decision_idx, cursor_end,
                )
                return action, 0.5, {
                    "mode": "warmup",
                    "sampler_cursor_start": cursor_start,
                    "sampler_cursor_end": cursor_end,
                }

            # Use the agent-wide pooled fit (group_id=None).
            pooled = self._load_latest_snapshot(
                "local_fit", decision_type, group_id=None
            )
            if pooled is not None and len(pooled.theta) == phi_dim:
                mean = np.asarray(pooled.theta, dtype=np.float64)
                cov = np.asarray(pooled.covariance, dtype=np.float64)
                source = "pooled_fit"
            else:
                mean = np.zeros(phi_dim, dtype=np.float64)
                cov = _prior_covariance(decision_type)
                source = "prior"

            cov = self._stabilize_covariance(cov)

            phi0 = fb.expand_base_to_phi(state_vec, 0)
            phi1 = fb.expand_base_to_phi(state_vec, 1)
            dphi = phi1 - phi0
            m = float(dphi @ mean)
            v = float(dphi @ cov @ dphi)

            prob = smooth_allocation_prob(
                m, v, self.z_bank,
                lmin=self.lmin, lmax=self.lmax,
                c=self.c, b=self.b, k=self.k,
            )

            cursor_start = self.sampler.cursor()
            action = int(self.sampler.draw_bernoulli(prob))
            cursor_end = self.sampler.cursor()
            chosen_prob = prob if action == 1 else (1.0 - prob)

            self.logger.info(
                "ILP action=%d group_id=%s decision_type=%s decision_idx=%d "
                "prob=%.6f m=%.4f v=%.4f source=%s cursor=%s",
                action, group_id, decision_type, decision_idx,
                chosen_prob, m, v, source, cursor_end,
            )
            return action, float(chosen_prob), {
                "mode": "smooth_logistic",
                "source": source,
                "m": m,
                "v": v,
                "sampler_cursor_start": cursor_start,
                "sampler_cursor_end": cursor_end,
            }
        except Exception as exc:
            self.logger.error("Inf-LSVI (pooled) action selection failed: %s", exc)
            raise

    # ------------------------------------------------------------------ update

    def update(self, old_params: dict, data: dict) -> tuple[bool, dict]:
        try:
            records = data.get("records", [])
            if not records:
                return True, {
                    "probability_of_action": old_params.get("probability_of_action", 0.5)
                }

            # Bucket by agent, ignore group_id (full pooling).
            by_agent: dict[str, list[dict]] = defaultdict(list)
            for record in records:
                by_agent[record["decision_type"]].append(record)

            # Also bucket by (agent, dyad) so we can compute per-dyad week-1
            # baselines (still per-dyad — only the Q-fit is pooled).
            by_agent_dyad: dict[str, dict[str, list[dict]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for record in records:
                by_agent_dyad[record["decision_type"]][record["group_id"]].append(record)

            for decision_type, agent_records in by_agent.items():
                # Persist per-dyad week-1 baselines (used by the feature
                # builder), then run ONE pooled Inf-LSVI fit per agent.
                for group_id, dyad_rows in by_agent_dyad[decision_type].items():
                    ordered_dyad = sorted(
                        dyad_rows, key=lambda r: r["agent_decision_index"]
                    )
                    self._maybe_persist_baselines(group_id, decision_type, ordered_dyad)

                # Order all records by (group_id, agent_decision_index) so the
                # Bellman target's "next state" still walks a single dyad's
                # trajectory, not across dyads.
                ordered = sorted(
                    agent_records,
                    key=lambda r: (r["group_id"], r["agent_decision_index"]),
                )

                previous = self._load_latest_snapshot(
                    "local_fit", decision_type, group_id=None
                )
                fit = self._fit_pooled_model(decision_type, ordered, previous)
                fit["agent_decision_index"] = ordered[-1]["agent_decision_index"]
                self._save_snapshot(
                    snapshot_type="local_fit",
                    decision_type=decision_type,
                    agent_decision_index=fit["agent_decision_index"],
                    group_id=None,
                    sample_size=fit["sample_size"],
                    theta=fit["theta_hat"].tolist(),
                    covariance=fit["covariance"].tolist(),
                    perturbation=None,
                    metadata_json={
                        "update_decision_idx": ordered[-1]["decision_idx"],
                        "n_dyads_in_fit": len(by_agent_dyad[decision_type]),
                        "sampler_cursor_start": fit["sampler_cursor_start"],
                        "sampler_cursor_end": fit["sampler_cursor_end"],
                    },
                )

            return True, {
                "probability_of_action": old_params.get("probability_of_action", 0.5)
            }
        except Exception as exc:
            self.logger.error("Inf-LSVI (pooled) update error: %s", exc)
            return False, old_params

    # ---------------------------------------------------------- delegation

    def make_state(self, context: dict) -> tuple[bool, list]:
        decision_type = context.get("decision_type")
        valid, err = validate_context(decision_type, context)
        if not valid:
            return False, err
        group_id = context.get("group_id")
        baselines = None
        if group_id is not None:
            baselines = fetch_baselines(group_id, decision_type) or None
        return True, encode_state(decision_type, context, baselines=baselines)

    def make_reward(self, user_id: str, state, action: int, outcome: dict) -> tuple[bool, float]:
        decision_type = outcome.get("decision_type")
        valid, err = validate_outcome(decision_type, outcome)
        if not valid:
            return False, err
        return True, compute_reward(decision_type, action, outcome)

    # ---------------------------------------------------------------- helpers

    def _is_warmup(self, group_id: str, decision_type: str, decision_idx: int) -> bool:
        # Warm-up is decided server-side in the /action route (API-Spec §3.2);
        # the route draws the warm-up action and skips get_action.
        return False

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

    def _fit_pooled_model(
        self,
        decision_type: str,
        records: list[dict],
        previous_snapshot,
    ) -> dict:
        """Bayesian linear regression on the *concatenation* of every dyad's
        Bellman-target rows. The Bellman next-state uses each dyad's own
        successor (so we don't cross-link trajectories), but the parameter
        vector being fit is shared across dyads."""
        fb = ProtocolRLFeatureBuilder(decision_type)
        feature_dim = fb.phi_dim
        gamma = GAMMA_BY_AGENT.get(decision_type, 0.9)
        prev_theta = np.zeros(feature_dim, dtype=np.float64)
        if previous_snapshot is not None and previous_snapshot.feature_dim == feature_dim:
            prev_theta = np.asarray(previous_snapshot.theta, dtype=np.float64)

        # Group rows by dyad so the per-dyad successor lookup is correct.
        by_dyad: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            by_dyad[r["group_id"]].append(r)
        for gid in by_dyad:
            by_dyad[gid].sort(key=lambda r: r["agent_decision_index"])

        x_rows: list[np.ndarray] = []
        y_vals: list[float] = []
        for gid, ordered in by_dyad.items():
            baselines = fetch_baselines(gid, decision_type)
            for idx, record in enumerate(ordered):
                raw = record["raw_context"]
                phi_t = fb.phi(raw, int(record["action"]), baselines=baselines)
                reward = float(record["reward"])
                if idx + 1 < len(ordered):
                    nxt = ordered[idx + 1]["raw_context"]
                    q0 = float(fb.phi(nxt, 0, baselines=baselines) @ prev_theta)
                    q1 = float(fb.phi(nxt, 1, baselines=baselines) @ prev_theta)
                    target = reward + gamma * max(q0, q1)
                else:
                    target = reward
                x_rows.append(phi_t)
                y_vals.append(target)

        X = np.vstack(x_rows)
        y = np.asarray(y_vals, dtype=np.float64)
        S = (X.T @ X) / (SIGMA_NOISE ** 2)
        b = (X.T @ y) / (SIGMA_NOISE ** 2)
        anchor = _prior_covariance(decision_type)
        prior_prec = np.linalg.inv(anchor)
        cov = np.linalg.inv(S + prior_prec)
        theta_hat = cov @ b

        return {
            "sample_size": len(records),
            "feature_dim": feature_dim,
            "theta_hat": theta_hat,
            "covariance": cov,
            "decision_idx": records[-1]["decision_idx"],
            "sampler_cursor_start": self.sampler.cursor(),
            "sampler_cursor_end": self.sampler.cursor(),
        }

    # ---- persistence wrappers (identical to inf_lsvi_local) ----------------

    def _save_snapshot(self, snapshot_type, decision_type, agent_decision_index,
                       group_id, sample_size, theta, covariance, perturbation, metadata_json):
        if self.app is None:
            return
        with self.app.app_context():
            snap = ModelParameters(
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
            db.session.add(snap)
            db.session.commit()

    def _load_latest_snapshot(self, snapshot_type, decision_type, group_id=None):
        if self.app is None:
            return None
        with self.app.app_context():
            q = ModelParameters.query.filter_by(
                snapshot_type=snapshot_type,
                decision_type=decision_type,
                group_id=group_id,
            ).order_by(ModelParameters.agent_decision_index.desc())
            return q.first()

    def _stabilize_covariance(self, cov: np.ndarray) -> np.ndarray:
        cov = np.asarray(cov, dtype=np.float64)
        cov = (cov + cov.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, MIN_COV_JITTER)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T
