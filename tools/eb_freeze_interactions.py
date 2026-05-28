"""
Does freezing the action-interaction block let EB-Gradient saturate at
$L_{\\max}$?

Runs EB-Gradient twice on the same simulator:

  - Full features: $\\phi(s, a) = [1, a, (I_j, v_jI_j, aI_j, av_jI_j)_j]$,
    $D \\approx 50$ for AYA. This is the standard algorithm.

  - Main-effects only: drop the action-interaction block from the
    feature map: $\\phi(s, a) = [1, a, (I_j, v_jI_j)_j]$,
    $D \\approx 26$. Per-dyad Inf-LSVI fits only the main effects;
    EB-Gradient pools them; smooth allocation uses only the main
    action coefficient as the contrast direction.

The action contrast under the main-only map is just $m = \\theta[a]$,
$v = \\Sigma[a,a]$ — no 24-term interaction sum to add noise to $m$.
This isolates whether the interaction-coefficient noise is what's
keeping $\\pi$ short of $L_{\\max}$ in the full algorithm.

Plots cumulative cohort mean reward and cumulative cohort mean
$\\Pr(A{=}1)$ for both modes on the same axes.
"""

from __future__ import annotations

import datetime
import sys
import threading as _threading
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Sync /update background thread + suppress callback POST.
import app.routes.update as _update_module

class _SyncThread(_threading.Thread):
    def start(self):  # type: ignore[override]
        self.run()

_update_module.Thread = _SyncThread

import requests as _requests
_requests.post = lambda *a, **kw: type("R", (), {"status_code": 200, "text": ""})()

from config import TestingConfig
TestingConfig.SAMPLE_BUFFER_UNIFORMS = 50_000
TestingConfig.SAMPLE_BUFFER_NORMALS = 2_000_000
TestingConfig.RL_ALGORITHM = "eb_gradient"

FIG_DIR = REPO_ROOT.parent / "Monitoring_Algorithm" / "figures" / "eb_freeze_interactions"
FIG_DIR.mkdir(parents=True, exist_ok=True)

AGENT_ORDER = ("aya_message", "cp_message", "dyad_game")
AGENT_LABELS = {
    "aya_message": "AYA (twice/day)",
    "cp_message": "CP (daily)",
    "dyad_game": "REL (weekly)",
}


# ----- monkey-patches for the "main-effects only" feature map ---------------

import app.feature_builder as fb_module
import app.algorithms.empirical_bayes as eb_module
import app.algorithms.eb_gradient as eb_grad_module
import app.algorithms.inf_lsvi_local as il_module

_orig_phi_dim_descriptor = fb_module.ProtocolRLFeatureBuilder.__dict__["phi_dim"]
_orig_expand = fb_module.ProtocolRLFeatureBuilder.expand_base_to_phi
_orig_prior_eb = eb_module._prior_covariance
_orig_prior_grad = eb_grad_module._prior_covariance
_orig_prior_il = il_module._prior_covariance


def _phi_dim_main_only(self):
    return 2 + 2 * self.n_vars


def _expand_main_only(self, base, action):
    a = float(action)
    if base.shape[0] != self.base_dim:
        raise ValueError(
            f"base length {base.shape[0]} != expected {self.base_dim} for {self.decision_type}"
        )
    parts = [np.array([1.0, a], dtype=np.float64)]
    for j in range(self.n_vars):
        I_j = float(base[1 + 2 * j])
        vI_j = float(base[1 + 2 * j + 1])
        parts.append(np.array([I_j, vI_j], dtype=np.float64))
    return np.concatenate(parts)


def _prior_covariance_main_only(decision_type):
    fb = fb_module.ProtocolRLFeatureBuilder(decision_type)
    tau_m_sq = eb_module.TAU_M_SQ_BY_AGENT[decision_type]
    diag = [tau_m_sq] * (2 + 2 * fb.n_vars)
    return np.diag(np.asarray(diag, dtype=np.float64))


def _phi_dim_intercept_action(self):
    return 2


def _expand_intercept_action(self, base, action):
    return np.asarray([1.0, float(action)], dtype=np.float64)


def _prior_covariance_intercept_action(decision_type):
    tau_m_sq = eb_module.TAU_M_SQ_BY_AGENT[decision_type]
    return np.diag(np.asarray([tau_m_sq, tau_m_sq], dtype=np.float64))


def apply_main_only_patches():
    fb_module.ProtocolRLFeatureBuilder.phi_dim = property(_phi_dim_main_only)
    fb_module.ProtocolRLFeatureBuilder.expand_base_to_phi = _expand_main_only
    eb_module._prior_covariance = _prior_covariance_main_only
    eb_grad_module._prior_covariance = _prior_covariance_main_only
    il_module._prior_covariance = _prior_covariance_main_only


def apply_intercept_action_patches():
    fb_module.ProtocolRLFeatureBuilder.phi_dim = property(_phi_dim_intercept_action)
    fb_module.ProtocolRLFeatureBuilder.expand_base_to_phi = _expand_intercept_action
    eb_module._prior_covariance = _prior_covariance_intercept_action
    eb_grad_module._prior_covariance = _prior_covariance_intercept_action
    il_module._prior_covariance = _prior_covariance_intercept_action


def restore_full_features():
    fb_module.ProtocolRLFeatureBuilder.phi_dim = _orig_phi_dim_descriptor
    fb_module.ProtocolRLFeatureBuilder.expand_base_to_phi = _orig_expand
    eb_module._prior_covariance = _orig_prior_eb
    eb_grad_module._prior_covariance = _orig_prior_grad
    il_module._prior_covariance = _orig_prior_il


# ----- runner -----------------------------------------------------------------

def run_one(label: str) -> dict[str, dict[str, list[float]]]:
    from app import create_app, db
    from app.models import StudyData
    from tests.simulate_adapts_hct import run_simulation

    app = create_app("config.TestingConfig")
    with app.app_context():
        db.create_all()
        results = run_simulation(
            app.test_client(),
            base_date=datetime.date(2025, 1, 5),
            num_weeks=35,
            num_dyads=25,
            verbose=False,
        )
        print(
            f"[{label}] add_group={results['add_group']} "
            f"action={results['action']} upload_data={results['upload_data']} "
            f"update={results['update']} errors={len(results['errors'])}"
        )
        rows = (
            StudyData.query.filter(StudyData.reward.isnot(None))
            .order_by(StudyData.request_timestamp, StudyData.id)
            .all()
        )
        per_agent: dict[str, dict[str, list[float]]] = {
            a: {"reward": [], "action": []} for a in AGENT_ORDER
        }
        for r in rows:
            if r.decision_type in per_agent:
                per_agent[r.decision_type]["reward"].append(float(r.reward))
                per_agent[r.decision_type]["action"].append(float(r.action))
    return per_agent


def _cumulative(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = max(len(arr), 1)
    x = np.arange(1, n + 1)
    return x, np.cumsum(arr) / x


_MODE_COLORS = {
    "full": "#1f77b4",
    "main_only": "#2ca02c",
    "intercept_action": "#d62728",
}
_MODE_LABELS = {
    "full": "full features",
    "main_only": "main-only features",
    "intercept_action": "intercept + action",
}


def _plot_panel(ax, data_per_mode: dict[str, np.ndarray], ylabel: str, agent_label: str):
    for mode, arr in data_per_mode.items():
        x, c = _cumulative(arr)
        ax.plot(x, c, color=_MODE_COLORS[mode], lw=2.0,
                label=f"{_MODE_LABELS[mode]} (final {float(c[-1]):.3f})")
    ax.set_xlabel("decision count (chronological)")
    ax.set_ylabel(ylabel)
    ax.set_title(agent_label, fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)


def main():
    # Run 1 — full features.
    restore_full_features()
    data_full = run_one("full")
    fb = fb_module.ProtocolRLFeatureBuilder("aya_message")
    print(f"  [full] AYA phi_dim = {fb.phi_dim}")

    # Run 2 — main-only features.
    apply_main_only_patches()
    data_main = run_one("main_only")
    fb = fb_module.ProtocolRLFeatureBuilder("aya_message")
    print(f"  [main_only] AYA phi_dim = {fb.phi_dim}")
    restore_full_features()

    # Run 3 — intercept + action only (D=2).
    apply_intercept_action_patches()
    data_min = run_one("intercept_action")
    fb = fb_module.ProtocolRLFeatureBuilder("aya_message")
    print(f"  [intercept_action] AYA phi_dim = {fb.phi_dim}")
    restore_full_features()

    runs = {
        "full": data_full,
        "main_only": data_main,
        "intercept_action": data_min,
    }

    # ----- figure 1: cumulative reward -----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={"wspace": 0.22})
    for ax, agent in zip(axes, AGENT_ORDER):
        _plot_panel(
            ax,
            {mode: np.asarray(runs[mode][agent]["reward"], dtype=np.float64)
             for mode in runs},
            "cumulative cohort mean reward",
            AGENT_LABELS[agent],
        )
    fig.suptitle(
        "EB-Gradient cumulative reward: feature dimension sweep",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_reward = FIG_DIR / "cumulative_reward.png"
    fig.savefig(out_reward, dpi=180)
    plt.close(fig)
    print(f"wrote {out_reward}")

    # ----- figure 2: cumulative Pr(A=1) -----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={"wspace": 0.22})
    for ax, agent in zip(axes, AGENT_ORDER):
        _plot_panel(
            ax,
            {mode: np.asarray(runs[mode][agent]["action"], dtype=np.float64)
             for mode in runs},
            r"cumulative cohort mean $\Pr(A{=}1)$",
            AGENT_LABELS[agent],
        )
        ax.axhline(0.5, color="gray", ls=":", lw=0.9, alpha=0.6)
        ax.axhline(0.8, color="red", ls="--", lw=0.9, alpha=0.5)
        ax.set_ylim(-0.02, 1.02)
    fig.suptitle(
        "EB-Gradient cumulative $\\Pr(A{=}1)$: feature dimension sweep",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_action = FIG_DIR / "cumulative_action.png"
    fig.savefig(out_action, dpi=180)
    plt.close(fig)
    print(f"wrote {out_action}")


if __name__ == "__main__":
    main()
