"""
Three-agent empirical Bayes learner with MAP marginal-likelihood
hyperparameter estimation and a generalized-logistic smooth allocation
function. See Prior_Construction/Prior_Construction_Note.tex §EB-Gradient
and §Generalized-logistic smooth allocation for the derivations.

Differences vs the MoM-based `empirical_bayes.py`:

  * No anchor shrinkage. The EB hyper-mean $\\hat\\theta_0$ and diagonal
    hyper-variance $\\hat\\Sigma_0 = \\text{diag}(\\exp \\eta)$ are
    estimated by maximizing the marginal log-likelihood
        $\\ell(\\theta_0,\\Sigma_0) = -\\tfrac12 \\sum_i
            [\\log\\det(\\Sigma_0+V_i) + (\\hat\\theta_i-\\theta_0)^T
             (\\Sigma_0+V_i)^{-1}(\\hat\\theta_i-\\theta_0)]$.
    $\\theta_0$ is solved in closed form (GLS) at every iteration; only
    $\\eta = \\log\\tau^2$ is optimized via Adam.
  * Smooth allocation function: $\\pi = \\E_{z \\sim N(0,1)}[\\rho(m +
    \\sqrt v \\, z)]$ with $\\rho$ the generalized logistic
        $\\rho(x) = L_{\\min} + (L_{\\max}-L_{\\min})/(1+c e^{-bx})^k$.
    The expectation is estimated by Monte Carlo using a pre-sampled
    bank of $M$ standard normals stored on the algorithm; the same bank
    is reused at every decision so the allocation is itself a fixed
    deterministic mapping from $(m,v)$ to $\\pi$.

Persistence, warmup, refresh cadence, snapshot schema, and the
deterministic sample buffer (one Bernoulli per /action) are unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from app.algorithms.base import RLAlgorithm
from app.algorithms.empirical_bayes import (
    EB_REFRESH_EVERY,
    GAMMA_BY_AGENT,
    MIN_COV_JITTER,
    SIGMA_NOISE,
    TAU_M_SQ_BY_AGENT,
    TAU_X_SQ_BY_AGENT,
    _prior_covariance,
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


# ---------------------------------------------------------- smooth allocation

DEFAULT_LMIN = 0.2
DEFAULT_LMAX = 0.8
DEFAULT_C = 5.0
DEFAULT_B = 20.0
DEFAULT_K = 1.0
DEFAULT_MC_SAMPLES = 500
DEFAULT_MC_SEED = 12345  # separate from the action-Bernoulli buffer seed


# --------------------------------------------------------- MAP optimization

# Adam settings for gradient descent on the MAP marginal log-likelihood
#   ℓ_MAP(θ_0, Σ_0) = ℓ(θ_0, Σ_0) + Σ_d log p(τ_d²)
# under the hierarchical model
#   θ_i ~ N(θ_0, Σ_0),   θ̂_{i,k} | θ_i ~ N(θ_i, Σ̂_{i,k}),
# with an inverse-Gamma prior  τ_d² ~ InvGamma(ν₀/2, ν₀·τ₀_d²/2)
# on each diagonal entry. θ_0 has a closed-form GLS given Σ_0; only
# η = log τ² is optimized via gradient steps. See Prior_Construction_Note.tex
# §EB-Gradient and Study_Design/main.tex Algorithm 3.
MAP_ITERATIONS = 200
MAP_STEP = 0.05
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

# Bounds on log τ² as numerical guards. The InvGamma prior keeps τ² near
# τ₀² for small N, so the floor / ceil are only safety rails. exp(-10) ≈
# 4.5e-5; exp(4) ≈ 54.6 — generous range above and below τ₀² = 10.
ETA_FLOOR = -10.0
ETA_CEIL = 4.0

# Inverse-Gamma prior parameters (defaults; overridable from app.config).
# τ₀² = 10 is two orders of magnitude above any local-fit variance, so the
# cold pool is effectively uninformative. ν₀ = 5 gives data-prior parity at
# N = 5 (the cohort warm-up window). See Study_Design/main.tex
# Eq. eb-gradient-prior.
DEFAULT_PRIOR_TAU0_SQ = 10.0
DEFAULT_PRIOR_NU0 = 5.0


def _initial_eta(decision_type: str, tau0_sq: float) -> np.ndarray:
    """Initialize η = log τ₀² · 1_D, where τ₀² is the InvGamma prior
    location. The cold-start hyper-prior is intentionally large and
    diagonal so that for small N the EB posterior collapses to the
    per-dyad Inf-LSVI fit (Σ_0⁻¹ ≪ Σ̂_i⁻¹ in the posterior shrinkage)."""
    fb = ProtocolRLFeatureBuilder(decision_type)
    return np.full(fb.phi_dim, np.log(float(tau0_sq)), dtype=np.float64)


def smooth_allocation_prob(
    m: float,
    v: float,
    z_bank: np.ndarray,
    lmin: float = DEFAULT_LMIN,
    lmax: float = DEFAULT_LMAX,
    c: float = DEFAULT_C,
    b: float = DEFAULT_B,
    k: float = DEFAULT_K,
) -> float:
    """
    Monte Carlo estimate of E_{z~N(0,1)}[ρ(m + sqrt(v) z)].

    Uses a fixed pre-sampled z_bank — never consumes from the buffer cursor
    and never advances a counter, so calling this twice with the same
    (m, v) returns identical π.
    """
    v = max(float(v), 0.0)
    samples = float(m) + np.sqrt(v) * z_bank
    # generalized logistic ρ
    expo = np.exp(-b * samples)
    denom = (1.0 + c * expo) ** k
    rho = lmin + (lmax - lmin) / denom
    pi = float(np.mean(rho))
    # numerical safety: floor / ceil at L_min / L_max
    return float(min(max(pi, lmin), lmax))


# -------------------------------------------------------------- main class

class ThreeAgentEmpiricalBayesGradientAlgorithm(RLAlgorithm):
    """EB-Gradient: same three-agent skeleton as ThreeAgentEmpiricalBayesAlgorithm,
    but uses MAP marginal-likelihood for the hyper-parameters and a
    generalized-logistic smooth allocation function."""

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
                "ThreeAgentEmpiricalBayesGradientAlgorithm requires a "
                "DeterministicSampleStream. Generate one with "
                "`flask init-buffer` and configure SAMPLE_BUFFER_PATH."
            )
        self.sampler = sampler
        self._update_call_counts: dict[str, int] = defaultdict(int)

        cfg = (app.config if app is not None else {})
        self.lmin = float(cfg.get("SMOOTH_ALLOC_LMIN", DEFAULT_LMIN))
        self.lmax = float(cfg.get("SMOOTH_ALLOC_LMAX", DEFAULT_LMAX))
        self.c = float(cfg.get("SMOOTH_ALLOC_C", DEFAULT_C))
        self.b = float(cfg.get("SMOOTH_ALLOC_B", DEFAULT_B))
        self.k = float(cfg.get("SMOOTH_ALLOC_K", DEFAULT_K))
        n_mc = int(cfg.get("SMOOTH_ALLOC_MC_SAMPLES", DEFAULT_MC_SAMPLES))
        mc_seed = int(cfg.get("SMOOTH_ALLOC_MC_SEED", DEFAULT_MC_SEED))
        # Inverse-Gamma prior on diag(Σ_0).
        self.prior_tau0_sq = float(cfg.get("EB_PRIOR_TAU0_SQ", DEFAULT_PRIOR_TAU0_SQ))
        self.prior_nu0 = float(cfg.get("EB_PRIOR_NU0", DEFAULT_PRIOR_NU0))
        rng = np.random.default_rng(mc_seed)
        # Fixed pre-sampled bank of 1-D N(0,1) draws, shared across all
        # decisions for the entire study. Stored on the algorithm and
        # never consumed/advanced.
        self.z_bank = rng.standard_normal(n_mc).astype(np.float64)

        self.logger.info(
            "EB-Gradient algorithm initialized "
            "(InvGamma prior τ₀²=%.2f, ν₀=%.1f; MC samples=%d, seed=%d, "
            "ρ=GenLogistic(Lmin=%.2f, Lmax=%.2f, c=%.2f, b=%.3f, k=%.2f); "
            "sampler: %d normals, %d uniforms; cursor=%s)",
            self.prior_tau0_sq, self.prior_nu0,
            n_mc, mc_seed,
            self.lmin, self.lmax, self.c, self.b, self.k,
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
                    "EBG warmup action=%d group_id=%s decision_type=%s "
                    "decision_idx=%d cursor=%s",
                    action, group_id, decision_type, decision_idx, cursor_end,
                )
                return action, 0.5, random_state

            posterior = self._load_latest_snapshot(
                "posterior", decision_type, group_id=group_id
            )
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

            phi0 = fb.expand_base_to_phi(state_vec, 0)
            phi1 = fb.expand_base_to_phi(state_vec, 1)
            dphi = phi1 - phi0
            m = float(dphi @ mean)
            v = float(dphi @ cov @ dphi)

            prob_action_1 = smooth_allocation_prob(
                m, v, self.z_bank,
                lmin=self.lmin, lmax=self.lmax,
                c=self.c, b=self.b, k=self.k,
            )

            cursor_start = self.sampler.cursor()
            action = int(self.sampler.draw_bernoulli(prob_action_1))
            cursor_end = self.sampler.cursor()
            prob = prob_action_1 if action == 1 else (1.0 - prob_action_1)

            random_state = {
                "mode": "smooth_logistic",
                "source": source,
                "m": m,
                "v": v,
                "sampler_cursor_start": cursor_start,
                "sampler_cursor_end": cursor_end,
            }

            self.logger.info(
                "EBG action=%d group_id=%s decision_type=%s decision_idx=%d "
                "prob=%.6f m=%.4f v=%.4f source=%s cursor=%s",
                action, group_id, decision_type, decision_idx,
                prob, m, v, source, cursor_end,
            )
            return action, float(prob), random_state
        except Exception as exc:
            self.logger.error("EB-Gradient action selection failed: %s", exc)
            raise

    # ------------------------------------------------------------------ update

    def update(self, old_params: dict, data: dict) -> tuple[bool, dict]:
        try:
            records = data.get("records", [])
            if not records:
                return True, {
                    "probability_of_action": old_params.get("probability_of_action", 0.5)
                }

            grouped: dict[str, dict[str, list[dict]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for record in records:
                grouped[record["decision_type"]][record["group_id"]].append(record)

            for decision_type, group_records in grouped.items():
                self._update_call_counts[decision_type] += 1
                local_fits: dict[str, dict] = {}

                for group_id, rows in group_records.items():
                    ordered_rows = sorted(
                        rows, key=lambda row: row["agent_decision_index"]
                    )
                    self._maybe_persist_baselines(group_id, decision_type, ordered_rows)
                    baselines = fetch_baselines(group_id, decision_type)

                    previous = self._load_latest_snapshot(
                        "local_fit", decision_type, group_id=group_id
                    )
                    fit_summary = self._fit_local_model(
                        decision_type, ordered_rows, previous, baselines
                    )
                    fit_summary["agent_decision_index"] = ordered_rows[-1][
                        "agent_decision_index"
                    ]
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

                max_agent_index = max(
                    fit["agent_decision_index"] for fit in local_fits.values()
                )
                if self._is_eb_refresh_point(decision_type):
                    eb_mean, eb_cov, opt_log = self._estimate_hyperparameters(
                        local_fits, decision_type
                    )
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
                            "map_loglik_final": opt_log["loglik_final"],
                            "map_iterations": opt_log["iterations"],
                            "tau_sq_diag": np.exp(opt_log["eta_final"]).tolist(),
                        },
                    )
                else:
                    prev_hyper = self._load_latest_snapshot("hyper", decision_type)
                    if prev_hyper is None:
                        eb_mean, eb_cov, _ = self._estimate_hyperparameters(
                            local_fits, decision_type
                        )
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

                # Per-dyad posterior is the Gaussian product of the Inf-LSVI
                # local fit (θ̂_i, Σ̂_i) and the EB hyperprior (θ̂_0, Σ̂_0) —
                # exactly the formula used by the MoM+anchor version, but with
                # (θ̂_0, Σ̂_0) coming from the gradient-descent MAP above.
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

            return True, {
                "probability_of_action": old_params.get("probability_of_action", 0.5)
            }
        except Exception as exc:
            self.logger.error("EB-Gradient update error: %s", exc)
            return False, old_params

    # ----------------------------------------------------------- delegation

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

    # ---------------------------------------------------------------- helpers

    _WARMUP_DECISIONS = {
        "aya_message": 14,
        "cp_message": 7,
        "dyad_game": 1,
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
        """Collect per-dyad sufficient statistics and the ridge-regularized
        flat-prior local estimator (\\hat\\theta_i, V_i). These are the
        quantities consumed by the marginal-likelihood objective."""
        fb = ProtocolRLFeatureBuilder(decision_type)
        feature_dim = fb.phi_dim
        base_dim = fb.base_dim
        state_dim = len(records[0].get("state", []))
        gamma = GAMMA_BY_AGENT.get(decision_type, 0.9)
        prev_theta = np.zeros(feature_dim, dtype=np.float64)
        if previous_snapshot is not None and previous_snapshot.feature_dim == feature_dim:
            prev_theta = np.asarray(previous_snapshot.theta, dtype=np.float64)

        x_rows: list[np.ndarray] = []
        y_vals: list[float] = []
        for idx, record in enumerate(records):
            raw = record["raw_context"]
            phi = fb.phi(raw, int(record["action"]), baselines=baselines)
            reward = float(record["reward"])
            if idx + 1 < len(records):
                nxt = records[idx + 1]["raw_context"]
                q0 = float(fb.phi(nxt, 0, baselines=baselines) @ prev_theta)
                q1 = float(fb.phi(nxt, 1, baselines=baselines) @ prev_theta)
                target = reward + (gamma * max(q0, q1))
            else:
                target = reward
            x_rows.append(phi)
            y_vals.append(target)

        x_mat = np.vstack(x_rows)
        y_vec = np.asarray(y_vals, dtype=np.float64)
        # S_i = σ⁻² Φᵀ Φ, b_i = σ⁻² Φᵀ y, both sums over the dyad's decisions.
        S = (x_mat.T @ x_mat) / (SIGMA_NOISE**2)
        b = (x_mat.T @ y_vec) / (SIGMA_NOISE**2)

        # Per-dyad Inf-LSVI fit (same as empirical_bayes.py): use the
        # structural Σ_0^anchor as a *regularizer* on the per-dyad fit so
        # cold-start (n_i < D) is well-posed. This anchor is *not* an EB
        # anchor — the EB hyper-parameters (θ_0, Σ_0) below are estimated
        # purely from data via marginal-likelihood maximization, with no
        # shrinkage toward this anchor.
        anchor = _prior_covariance(decision_type)
        prior_precision = np.linalg.inv(anchor)
        V_inv = S + prior_precision
        V = np.linalg.inv(V_inv)
        theta_hat = V @ b

        return {
            "sample_size": len(records),
            "state_dim": state_dim,
            "base_dim": base_dim,
            "feature_dim": feature_dim,
            "S": S,
            "b": b,
            "theta_hat": theta_hat,
            "covariance": V,  # = "V_i" in the note
            "decision_idx": records[-1]["decision_idx"],
            "sampler_cursor_start": self.sampler.cursor(),
            "sampler_cursor_end": self.sampler.cursor(),
        }

    # ---- MAP marginal-likelihood optimization -----------------------------

    def _estimate_hyperparameters(
        self,
        local_fits: dict[str, dict],
        decision_type: str,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """MAP on the marginal log-likelihood of the Inf-LSVI local fits
        under the hierarchical model
            θ_i ~ N(θ_0, Σ_0),   θ̂_{i,k} | θ_i ~ N(θ_i, Σ̂_{i,k}),
        plus an inverse-Gamma prior τ_d² ~ InvGamma(ν₀/2, ν₀·τ₀²/2)
        on each diagonal entry of Σ_0 = diag(τ²). Marginally
        θ̂_{i,k} ~ N(θ_0, M_i) with M_i = Σ̂_{i,k} + Σ_0, so

            ℓ_MAP = -½ Σ_i [ log|M_i| + (θ̂_i - θ_0)^T M_i^{-1} (θ̂_i - θ_0) ]
                    - Σ_d [ (ν₀/2 + 1) η_d  +  (ν₀·τ₀²/2) exp(-η_d) ].

        θ_0 has closed-form GLS given Σ_0; only η = log diag(Σ_0) is
        optimized by gradient (Adam). At N ≲ ν₀ the prior pulls each
        τ_d² toward τ₀² (large) so the EB posterior collapses to the
        per-dyad Inf-LSVI fit; at N ≫ ν₀ the data dominates and we
        recover the ML estimator.
        """
        thetas = [np.asarray(fit["theta_hat"], dtype=np.float64) for fit in local_fits.values()]
        Sigmas = [self._stabilize_covariance(fit["covariance"]) for fit in local_fits.values()]
        N = len(thetas)
        D = thetas[0].shape[0]
        eye = np.eye(D)

        tau0_sq = self.prior_tau0_sq
        nu0 = self.prior_nu0
        log_tau0_sq = np.log(tau0_sq)

        # Warm-start (θ_0, η). η starts at log τ₀² (uninformative cold pool);
        # θ_0 at the previous snapshot if available, else 0.
        prev_hyper = self._load_latest_snapshot("hyper", decision_type)
        if prev_hyper is not None and len(prev_hyper.theta) == D:
            theta0 = np.asarray(prev_hyper.theta, dtype=np.float64)
            prev_cov = np.asarray(prev_hyper.covariance, dtype=np.float64)
            eta = np.log(np.clip(np.diag(prev_cov), np.exp(ETA_FLOOR), np.exp(ETA_CEIL)))
        else:
            theta0 = np.zeros(D, dtype=np.float64)
            eta = _initial_eta(decision_type, tau0_sq)
        eta = np.clip(eta, ETA_FLOOR, ETA_CEIL)

        # Adam state.
        m_adam = np.zeros_like(eta)
        v_adam = np.zeros_like(eta)
        best_ll = -np.inf
        best_eta = eta.copy()
        best_theta0 = theta0.copy()

        # Constant prior contribution to log p(τ²) for monitoring; the
        # per-iteration -(ν₀/2+1) η - (ν₀ τ₀²/2) exp(-η) terms are added
        # to the data ℓ to form ℓ_MAP.
        prior_const = -((nu0 / 2.0 + 1.0))  # multiplies η_d in log-prior
        prior_pull = (nu0 * tau0_sq / 2.0)  # multiplies exp(-η_d) in log-prior

        for it in range(1, MAP_ITERATIONS + 1):
            Sigma0 = np.diag(np.exp(eta))

            # GLS closed-form for θ_0 given the current Σ_0 (prior is
            # independent of θ_0, so the GLS is unchanged).
            W_list: list[np.ndarray] = []
            sum_W = np.zeros((D, D), dtype=np.float64)
            sum_W_theta = np.zeros(D, dtype=np.float64)
            for Sigma_i, theta_i in zip(Sigmas, thetas):
                M_i = Sigma_i + Sigma0
                W_i = np.linalg.inv(M_i + MIN_COV_JITTER * eye)
                W_list.append(W_i)
                sum_W += W_i
                sum_W_theta += W_i @ theta_i
            theta0 = np.linalg.solve(sum_W + MIN_COV_JITTER * eye, sum_W_theta)

            # ℓ_MAP at (θ_0(Σ_0), Σ_0) for monitoring + best-found tracking.
            ll = 0.0
            r_list: list[np.ndarray] = []
            for Sigma_i, theta_i, W_i in zip(Sigmas, thetas, W_list):
                _, logdet = np.linalg.slogdet(Sigma_i + Sigma0 + MIN_COV_JITTER * eye)
                r_i = theta_i - theta0
                r_list.append(r_i)
                ll += -0.5 * (logdet + float(r_i @ (W_i @ r_i)))
            # Add log-prior on η (ignoring constants).
            ll += float(np.sum(prior_const * eta - prior_pull * np.exp(-eta)))

            if np.isfinite(ll) and ll > best_ll:
                best_ll = ll
                best_eta = eta.copy()
                best_theta0 = theta0.copy()

            # Gradient of the *profiled* ℓ_MAP w.r.t. η: data + prior.
            grad_eta = np.zeros_like(eta)
            for W_i, r_i in zip(W_list, r_list):
                Wr = W_i @ r_i
                grad_eta -= 0.5 * np.exp(eta) * (np.diag(W_i) - Wr * Wr)
            # Prior gradient: -(ν₀/2 + 1) + (ν₀ τ₀²/2) exp(-η_d).
            grad_eta += prior_const + prior_pull * np.exp(-eta)
            if not np.all(np.isfinite(grad_eta)):
                break

            # Adam descent on -ℓ_MAP.
            g = -grad_eta
            m_adam = ADAM_BETA1 * m_adam + (1 - ADAM_BETA1) * g
            v_adam = ADAM_BETA2 * v_adam + (1 - ADAM_BETA2) * (g * g)
            m_hat = m_adam / (1 - ADAM_BETA1 ** it)
            v_hat = v_adam / (1 - ADAM_BETA2 ** it)
            eta = eta - MAP_STEP * m_hat / (np.sqrt(v_hat) + ADAM_EPS)
            eta = np.clip(eta, ETA_FLOOR, ETA_CEIL)

        eb_mean = best_theta0
        eb_cov = np.diag(np.exp(best_eta))
        opt_log = {
            "iterations": MAP_ITERATIONS,
            "loglik_final": float(best_ll),
            "eta_final": best_eta.tolist(),
            "prior_tau0_sq": tau0_sq,
            "prior_nu0": nu0,
        }
        self.logger.info(
            "EBG MAP %s N=%d iters=%d loglik=%.3f tau2_mean=%.4f "
            "theta0[:2]=%s",
            decision_type, N, MAP_ITERATIONS,
            float(best_ll), float(np.mean(np.exp(best_eta))),
            np.array2string(eb_mean[:2], precision=3),
        )
        return eb_mean, eb_cov, opt_log

    def _shrink_to_hyperprior(
        self,
        local_theta: np.ndarray,
        local_cov: np.ndarray,
        eb_mean: np.ndarray,
        eb_cov: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gaussian product of the Inf-LSVI local posterior N(θ̂_i, Σ̂_i)
        with the EB-estimated hyperprior N(θ̂_0, Σ̂_0). Same formula as
        the MoM+anchor version — no anchor enters."""
        local_cov = self._stabilize_covariance(local_cov)
        eb_cov = self._stabilize_covariance(eb_cov)
        precision = np.linalg.inv(local_cov) + np.linalg.inv(eb_cov)
        post_cov = np.linalg.inv(precision)
        post_mean = post_cov @ (
            np.linalg.inv(local_cov) @ local_theta
            + np.linalg.inv(eb_cov) @ eb_mean
        )
        return post_mean, self._stabilize_covariance(post_cov)

    # ---- persistence wrappers (identical to empirical_bayes) -------------

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
