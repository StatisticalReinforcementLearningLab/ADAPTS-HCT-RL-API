"""
Prior-variance validation on the *live* RL API (not the standalone
Prior_Construction/code/ implementation).

Boots the Flask app with RL_ALGORITHM=eb_gradient, runs the protocol-
faithful 25-dyad × 35-week simulator, then queries the persisted
ModelParameters + Action rows to plot, per agent:

  (1) Trace of the EB hyper-covariance  $\\hat\\Sigma_0$  over time
      (the per-coefficient variance pool).
  (2) Median trace of per-dyad posterior  $\\hat\\Sigma_i$  over time
      (with 5--95% band across dyads).
  (3) Rolling cumulative cohort $\\Pr(A{=}1)$ (the running policy).

Optional flag: ``--correlate AYA_FEATURE_A,AYA_FEATURE_B,RHO`` injects
pairwise correlation between two AYA continuous features by replacing
one with $\\rho \\cdot$ the other $+ \\sqrt{1-\\rho^2} \\cdot \\text{noise}$
at feature-build time. Used by ``tools/stress_test_correlation.py``.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import threading as _threading
from collections import defaultdict
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
TestingConfig.RL_ALGORITHM = os.environ.get("RL_ALGORITHM", "eb_gradient")

AGENT_ORDER = ("aya_message", "cp_message", "dyad_game")
AGENT_LABELS = {
    "aya_message": "AYA (twice/day)",
    "cp_message": "CP (daily)",
    "dyad_game": "REL (weekly)",
}


def run_simulation_and_collect(num_dyads: int = 25, num_weeks: int = 35) -> dict:
    """Boot the app, run the simulator, return raw snapshot + action rows."""
    from app import create_app, db
    from app.models import ModelParameters, Action
    from tests.simulate_adapts_hct import run_simulation

    app = create_app("config.TestingConfig")
    out: dict = {"hyper": {}, "posterior": {}, "actions": {}}
    with app.app_context():
        db.create_all()
        results = run_simulation(
            app.test_client(),
            base_date=datetime.date(2025, 1, 5),
            num_weeks=num_weeks,
            num_dyads=num_dyads,
            verbose=False,
        )
        print(
            f"[run] add_group={results['add_group']} action={results['action']} "
            f"upload_data={results['upload_data']} update={results['update']} "
            f"errors={len(results['errors'])}"
        )

        # Hyper snapshots: one per (agent, refresh).
        for agent in AGENT_ORDER:
            rows = (
                ModelParameters.query
                .filter_by(snapshot_type="hyper", decision_type=agent)
                .order_by(ModelParameters.agent_decision_index.asc())
                .all()
            )
            out["hyper"][agent] = [
                {
                    "agent_idx": int(r.agent_decision_index),
                    "sample_size": int(r.sample_size),  # N (active dyads)
                    "trace": float(np.sum(np.diag(np.asarray(r.covariance, dtype=np.float64)))),
                    "diag_mean": float(np.mean(np.diag(np.asarray(r.covariance, dtype=np.float64)))),
                    "theta_action": float(r.theta[1]),  # position 1 = action coef
                }
                for r in rows
            ]

        # Posterior snapshots: per-(agent, dyad, refresh).
        # Fully-pooled learners (e.g. REL under hybrid_rel_pool) write one
        # shared local_fit per refresh with group_id=None instead — surface
        # that single trajectory as a degenerate "all dyads share this trace".
        for agent in AGENT_ORDER:
            rows = (
                ModelParameters.query
                .filter_by(snapshot_type="posterior", decision_type=agent)
                .order_by(ModelParameters.agent_decision_index.asc())
                .all()
            )
            if rows:
                by_idx: dict[int, list[float]] = defaultdict(list)
                for r in rows:
                    tr = float(np.sum(np.diag(np.asarray(r.covariance, dtype=np.float64))))
                    by_idx[int(r.agent_decision_index)].append(tr)
                out["posterior"][agent] = sorted(by_idx.items())
            else:
                pooled = (
                    ModelParameters.query
                    .filter_by(snapshot_type="local_fit", decision_type=agent, group_id=None)
                    .order_by(ModelParameters.agent_decision_index.asc())
                    .all()
                )
                out["posterior"][agent] = [
                    (i + 1, [float(np.sum(np.diag(np.asarray(r.covariance, dtype=np.float64))))])
                    for i, r in enumerate(pooled)
                ]

        # Actions: chronological list of (request_timestamp, action, prob).
        for agent in AGENT_ORDER:
            rows = (
                Action.query.filter_by(decision_type=agent)
                .order_by(Action.request_timestamp.asc(), Action.id.asc())
                .all()
            )
            out["actions"][agent] = [
                # convert chosen-prob to π(a=1)
                (float(r.action),
                 float(r.action_prob) if int(r.action) == 1 else 1.0 - float(r.action_prob),
                 (r.random_state or {}).get("mode", "eb"))
                for r in rows
            ]
    return out


def _make_panel_grid(suptitle: str):
    fig, axes = plt.subplots(3, 3, figsize=(18, 13), gridspec_kw={"hspace": 0.42, "wspace": 0.25})
    fig.suptitle(suptitle, fontsize=14)
    return fig, axes


def plot_three_diagnostics(data: dict, out_path: Path, suptitle: str) -> None:
    fig, axes = _make_panel_grid(suptitle)

    # Row 0: hyper Σ_0 trace (and θ_0[action] overlay) vs refresh number.
    for col, agent in enumerate(AGENT_ORDER):
        ax = axes[0, col]
        hyp = data["hyper"][agent]
        if hyp:
            ns = [h["sample_size"] for h in hyp]
            traces = [h["trace"] for h in hyp]
            ax.plot(ns, traces, "o-", color="#1f77b4", lw=1.8,
                    label=r"$\mathrm{tr}\,\hat\Sigma_0$")
            ax2 = ax.twinx()
            ax2.plot(ns, [h["theta_action"] for h in hyp], "s--",
                     color="#d62728", lw=1.2, ms=4,
                     label=r"$\hat\theta_0[\mathrm{action}]$")
            ax2.set_ylabel(r"$\hat\theta_0[\mathrm{action}]$", color="#d62728")
            ax2.tick_params(axis="y", labelcolor="#d62728")
            ax.set_yscale("log")
            ax.set_xlabel("active dyads $N$")
            ax.set_ylabel(r"$\mathrm{tr}\,\hat\Sigma_0$ (log)", color="#1f77b4")
            ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax.set_title(f"{AGENT_LABELS[agent]} --- hyper $\\hat\\Sigma_0$", fontsize=11)
        ax.grid(True, alpha=0.25)

    # Row 1: per-dyad Σ_i^post trace (median + 5–95% band) vs refresh.
    for col, agent in enumerate(AGENT_ORDER):
        ax = axes[1, col]
        post = data["posterior"][agent]
        if post:
            xs = [agent_idx for agent_idx, _ in post]
            traces_per_idx = [traces for _, traces in post]
            meds = [float(np.median(t)) for t in traces_per_idx]
            lo = [float(np.quantile(t, 0.05)) for t in traces_per_idx]
            hi = [float(np.quantile(t, 0.95)) for t in traces_per_idx]
            ax.fill_between(xs, lo, hi, color="#1f77b4", alpha=0.18, label="5–95%")
            ax.plot(xs, meds, "o-", color="#1f77b4", lw=1.8, ms=4, label="median")
            ax.set_yscale("log")
            ax.set_xlabel("agent decision index")
            ax.set_ylabel(r"$\mathrm{tr}\,\hat\Sigma_i^{\mathrm{post}}$ (log)")
        ax.set_title(f"{AGENT_LABELS[agent]} --- per-dyad $\\hat\\Sigma_i^{{\\mathrm{{post}}}}$", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    # Row 2: rolling cumulative Pr(A=1) over decision count.
    for col, agent in enumerate(AGENT_ORDER):
        ax = axes[2, col]
        acts = data["actions"][agent]
        if acts:
            arr = np.asarray([a for a, _, _ in acts], dtype=np.float64)
            modes = np.asarray([m for _, _, m in acts])
            xs = np.arange(1, len(arr) + 1)
            cum = np.cumsum(arr) / xs
            ax.plot(xs, cum, color="#1f77b4", lw=1.8,
                    label=f"cumulative $\\Pr(A{{=}}1)$ (final {float(cum[-1]):.3f})")
            ax.axhline(0.5, color="gray", ls=":", lw=0.9, alpha=0.6)
            ax.axhline(0.8, color="red", ls="--", lw=0.9, alpha=0.5, label=r"$L_{\max}=0.8$")
            ax.axhline(0.2, color="red", ls="--", lw=0.9, alpha=0.5, label=r"$L_{\min}=0.2$")
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("decision count")
            ax.set_ylabel(r"cumulative $\Pr(A{=}1)$")
        ax.set_title(f"{AGENT_LABELS[agent]} --- running policy", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-dyads", type=int, default=25)
    ap.add_argument("--num-weeks", type=int, default=35)
    ap.add_argument(
        "--out",
        default=str(
            REPO_ROOT.parent
            / "Monitoring_Algorithm/figures/eb_gradient_prior_validation/prior_validation.png"
        ),
    )
    ap.add_argument("--title", default="EB-Gradient prior-variance validation in the live RL API")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = run_simulation_and_collect(num_dyads=args.num_dyads, num_weeks=args.num_weeks)
    plot_three_diagnostics(data, out_path, args.title)


if __name__ == "__main__":
    main()
