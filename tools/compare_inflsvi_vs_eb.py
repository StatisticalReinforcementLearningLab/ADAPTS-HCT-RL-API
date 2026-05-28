"""
Side-by-side comparison: per-dyad Inf-LSVI (no pooling) vs. EB-Gradient
(with pooling) on the same protocol-faithful simulator.

For each algorithm we boot a fresh in-memory Flask app (so DB and
sampler buffer are independent), run the 25-dyad × 35-week trial, and
collect every StudyData reward in chronological order. We then plot
cumulative cohort mean reward over decision count for both algorithms
on the same axes, per agent (AYA / CP / REL).

Both runs share the same simulator parameters (boosted action effects
from the verification setup) and the same sampler seed, so the only
thing that differs is the learner.
"""

from __future__ import annotations

import datetime
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

# Sync /update background thread + suppress callback POST (same pattern
# as rerun_sanity_check_eb_gradient.py). Patch BEFORE importing the app.
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

FIG_DIR = REPO_ROOT.parent / "Monitoring_Algorithm" / "figures" / "inf_lsvi_vs_eb"
FIG_DIR.mkdir(parents=True, exist_ok=True)

AGENT_ORDER = ("aya_message", "cp_message", "dyad_game")
AGENT_LABELS = {
    "aya_message": "AYA (twice/day)",
    "cp_message": "CP (daily)",
    "dyad_game": "REL (weekly)",
}


def run_one(algo_name: str) -> dict[str, dict[str, list[float]]]:
    """Boot a fresh app with `algo_name`, run the simulation, and return
    per-agent lists of (rewards, actions) in chronological order."""
    # Reset the TestingConfig.RL_ALGORITHM in-place so create_app picks it up.
    TestingConfig.RL_ALGORITHM = algo_name
    # Each algorithm gets its own deterministic buffer (same seed, so the
    # warm-up Bernoullis and any random consumption are reproducible).
    TestingConfig.SAMPLE_BUFFER_PATH = None  # in-memory, per-boot
    # Use the same MC seed for both algorithms (so the smooth-allocation
    # bank is identical — they share that piece of randomness).

    # Defer imports until *after* TestingConfig is mutated so create_app
    # sees the right RL_ALGORITHM.
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
            f"[{algo_name}] add_group={results['add_group']} "
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
    """Return (x, cumulative-mean) for a 1-D sequence."""
    n = max(len(arr), 1)
    x = np.arange(1, n + 1)
    return x, np.cumsum(arr) / x


_ALGO_STYLE = {
    "eb_gradient":   {"color": "#1f77b4", "label": "EB-Gradient",            "lw": 2.0, "ls": "-"},
    "inf_lsvi_pool": {"color": "#d95f02", "label": "Inf-LSVI (full pooling)", "lw": 2.0, "ls": "-"},
    "always_send":   {"color": "#2ca02c", "label": "Always send (a=1)",       "lw": 1.6, "ls": "--"},
    "always_none":   {"color": "#7f7f7f", "label": "Always none (a=0)",       "lw": 1.6, "ls": ":"},
}


def _plot_panel(ax, runs: dict[str, np.ndarray], ylabel: str, agent_label: str):
    for algo, arr in runs.items():
        style = _ALGO_STYLE[algo]
        x, cum = _cumulative(arr)
        ax.plot(x, cum, color=style["color"], lw=style["lw"], ls=style["ls"],
                label=f"{style['label']} (final {float(cum[-1]):.3f})")
    ax.set_xlabel("decision count (chronological)")
    ax.set_ylabel(ylabel)
    ax.set_title(agent_label, fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8.5, framealpha=0.92)


def main():
    # Run each algorithm once. Each gets its own in-memory Flask app + DB,
    # same SAMPLE_BUFFER_SEED, same simulator seed.
    runs = {
        algo: run_one(algo)
        for algo in ("eb_gradient", "inf_lsvi_pool", "always_send", "always_none")
    }

    # ---- Figure 1: cumulative cohort mean reward ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={"wspace": 0.22})
    for ax, agent in zip(axes, AGENT_ORDER):
        _plot_panel(
            ax,
            {algo: np.asarray(runs[algo][agent]["reward"], dtype=np.float64)
             for algo in runs},
            "cumulative cohort mean reward",
            AGENT_LABELS[agent],
        )
    fig.suptitle(
        "Cumulative cohort mean reward: EB-Gradient vs. Inf-LSVI (full pool) vs. always-send vs. always-none",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_reward = FIG_DIR / "cumulative_reward_comparison.png"
    fig.savefig(out_reward, dpi=180)
    plt.close(fig)
    print(f"wrote {out_reward}")

    # ---- Figure 2: cumulative cohort mean Pr(A = 1) ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={"wspace": 0.22})
    for ax, agent in zip(axes, AGENT_ORDER):
        _plot_panel(
            ax,
            {algo: np.asarray(runs[algo][agent]["action"], dtype=np.float64)
             for algo in runs},
            r"cumulative cohort mean $\Pr(A{=}1)$",
            AGENT_LABELS[agent],
        )
        ax.axhline(0.5, color="gray", ls=":", lw=0.9, alpha=0.6)
        ax.axhline(0.8, color="red", ls="--", lw=0.9, alpha=0.5)
        ax.set_ylim(-0.02, 1.02)
    fig.suptitle(
        "Cumulative cohort mean $\\Pr(A{=}1)$: EB-Gradient vs. Inf-LSVI (full pool) vs. always-send vs. always-none",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_action = FIG_DIR / "cumulative_action_comparison.png"
    fig.savefig(out_action, dpi=180)
    plt.close(fig)
    print(f"wrote {out_action}")


if __name__ == "__main__":
    main()
