"""
Per-dyad sampling-probability trajectory.

Runs the live RL API under EB-Gradient and plots, per agent, one
rolling-mean $\\pi(a{=}1)$ line per dyad over the dyad's own decision
index. Each line starts at $0.5$ in the warm-up week and then transits
to whatever the EB-Gradient + smooth-allocation policy commits to.

Distinct from existing $\\pi$-over-time views (fig3 = chronologically
concatenated, fig4 = first-14-decision summary): here we keep dyads
separated so heterogeneity in commitment is visible.
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

FIG_DIR = REPO_ROOT.parent / "Monitoring_Algorithm/figures/eb_gradient_prior_validation"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) <= 1:
        return values
    window = max(2, min(window, len(values)))
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-dyads", type=int, default=25)
    ap.add_argument("--num-weeks", type=int, default=35)
    ap.add_argument("--out", default=str(FIG_DIR / "per_dyad_pi_trajectory.png"))
    args = ap.parse_args()

    from app import create_app, db
    from app.models import Action
    from tests.simulate_adapts_hct import run_simulation

    app = create_app("config.TestingConfig")
    with app.app_context():
        db.create_all()
        results = run_simulation(
            app.test_client(),
            base_date=datetime.date(2025, 1, 5),
            num_weeks=args.num_weeks,
            num_dyads=args.num_dyads,
            verbose=False,
        )
        print(
            f"[run] add_group={results['add_group']} action={results['action']} "
            f"upload_data={results['upload_data']} update={results['update']} "
            f"errors={len(results['errors'])}"
        )

        per_dyad_agent: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for r in (
            Action.query.order_by(
                Action.decision_type, Action.group_id, Action.decision_idx
            ).all()
        ):
            if r.decision_type in AGENT_LABELS:
                # action_prob stores Pr(chosen action). Convert to π(a=1).
                pi1 = float(r.action_prob) if int(r.action) == 1 else 1.0 - float(r.action_prob)
                per_dyad_agent[r.decision_type][r.group_id].append(pi1)

    # ------ figure ------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={"wspace": 0.22})

    for ax, agent in zip(axes, AGENT_ORDER):
        groups = sorted(per_dyad_agent[agent].keys())
        cmap = plt.get_cmap("viridis")
        colors = [cmap(i / max(len(groups) - 1, 1)) for i in range(len(groups))]
        for gid, color in zip(groups, colors):
            probs = np.asarray(per_dyad_agent[agent][gid], dtype=np.float64)
            if len(probs) < 2:
                continue
            window = max(4, int(len(probs) ** 0.5))
            rm = rolling_mean(probs, window)
            xs = np.arange(1, len(probs) + 1)
            ax.plot(xs, rm, color=color, lw=1.0, alpha=0.7)
        # Cohort median as a thick black line for reference.
        # Build a matrix dyads × max_T (NaN-padded) and take nanmedian.
        max_T = max((len(per_dyad_agent[agent][g]) for g in groups), default=0)
        if max_T > 1:
            mat = np.full((len(groups), max_T), np.nan)
            for i, g in enumerate(groups):
                v = np.asarray(per_dyad_agent[agent][g], dtype=np.float64)
                w = max(4, int(len(v) ** 0.5))
                rm = rolling_mean(v, w)
                mat[i, : len(rm)] = rm
            cohort_med = np.nanmedian(mat, axis=0)
            ax.plot(np.arange(1, max_T + 1), cohort_med, color="black", lw=2.4,
                    label="cohort median")
        ax.axhline(0.5, color="gray", ls=":", lw=0.9, alpha=0.6)
        ax.axhline(0.8, color="red", ls="--", lw=0.9, alpha=0.5)
        ax.axhline(0.2, color="red", ls="--", lw=0.9, alpha=0.5)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("within-dyad decision index")
        ax.set_ylabel(r"rolling-mean $\pi(a{=}1 \mid s)$")
        ax.set_title(AGENT_LABELS[agent], fontsize=12)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, loc="lower right", framealpha=0.92)

    fig.suptitle(
        "Per-dyad sampling-probability trajectory over within-dyad decisions",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = Path(args.out)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
