"""Regenerate the monitoring sanity-check figures under the per-dyad week-1
warm-up protocol.

Drives the protocol-faithful simulator against an in-memory Flask test app
(empirical-Bayes learner). Pulls Action and StudyData rows directly from the
DB and writes two PNGs to ../Monitoring_Algorithm/figures/sanity_check/:

  - fig3_action_prob.png  --- per-decision action_prob trajectory, grey =
    per-dyad week-1 warm-up (action_prob = 0.5 by construction), blue =
    learned-policy EB decisions (probit-TS marginal probability).
  - fig2_per_dyad_avg.png --- per-dyad mean reward as cohort grows; one point
    per dyad, ordered by recruitment.

The simulator already calls /add_group with warmup=False; the EB algorithm's
_is_warmup() applies a per-decision threshold (14 AYA / 7 CP / 1 REL = first
calendar week of every dyad).
"""

from __future__ import annotations

import datetime
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Patch app.routes.update.Thread *before* the app boots, so the /update
# background worker runs synchronously in the main test-client thread. That
# avoids the SQLAlchemy "session is in prepared state" race between the
# background EB-update thread and the main thread that queries the DB at the
# end, and also suppresses the callback-URL connection errors (the simulator
# points at a non-existent localhost:5000 callback).
import app.routes.update as _update_module
import threading as _threading

class _SyncThread(_threading.Thread):
    def start(self):  # type: ignore[override]
        self.run()

_update_module.Thread = _SyncThread

# Make the failed callback POST a no-op too (we don't want any HTTP error spam).
import requests as _requests
_requests.post = lambda *a, **kw: type("R", (), {"status_code": 200, "text": ""})()

from config import TestingConfig
# Default TestingConfig allocates only 5 000 uniform primitives; the full
# 25-dyad × 35-week run consumes ~7 500 (one per /action Bernoulli draw).
# Bump both buffers *before* create_app reads them.
TestingConfig.SAMPLE_BUFFER_UNIFORMS = 50_000
TestingConfig.SAMPLE_BUFFER_NORMALS = 2_000_000

from app import create_app, db
from app.models import Action, StudyData
from tests.simulate_adapts_hct import run_simulation

FIG_DIR = REPO_ROOT.parent / "Monitoring_Algorithm" / "figures" / "sanity_check"
FIG_DIR.mkdir(parents=True, exist_ok=True)

AGENT_ORDER = ("aya_message", "cp_message", "dyad_game")
AGENT_LABELS = {
    "aya_message": "AYA (twice/day)",
    "cp_message": "CP (daily)",
    "dyad_game": "REL (weekly)",
}
CLIP_BOUNDS = (0.1, 0.9)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < 2:
        return values
    window = max(2, min(window, len(values)))
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode="valid")


def main():
    app = create_app("config.TestingConfig")
    with app.app_context():
        db.create_all()

        # Full 25-dyad, 35-week run.
        results = run_simulation(
            app.test_client(),
            base_date=datetime.date(2025, 1, 5),
            num_weeks=35,
            num_dyads=25,
            verbose=False,
        )
        print(
            f"add_group={results['add_group']} action={results['action']} "
            f"upload_data={results['upload_data']} update={results['update']} "
            f"errors={len(results['errors'])}"
        )

        actions = (
            Action.query.order_by(
                Action.decision_type, Action.group_id, Action.decision_idx
            ).all()
        )
        study = (
            StudyData.query.filter(StudyData.reward.isnot(None))
            .order_by(StudyData.group_id, StudyData.decision_type, StudyData.decision_idx)
            .all()
        )

        # Bucket actions by agent, then split warmup vs EB by random_state.mode.
        by_agent: dict[str, list[Action]] = {a: [] for a in AGENT_ORDER}
        for a in actions:
            if a.decision_type in by_agent:
                by_agent[a.decision_type].append(a)

        # ---------- Figure 3: action_prob trajectory ----------
        fig, axes = plt.subplots(
            1, 3, figsize=(18, 5), sharey=True, gridspec_kw={"wspace": 0.12}
        )
        for ax, agent in zip(axes, AGENT_ORDER):
            rows = by_agent[agent]
            xs = np.arange(1, len(rows) + 1)
            probs = np.array([float(a.action_prob) for a in rows])
            modes = np.array([
                (a.random_state or {}).get("mode", "eb_policy") for a in rows
            ])
            warm_mask = modes == "warmup"
            eb_mask = ~warm_mask

            # EB-policy decisions in blue (drawn first / behind).
            ax.scatter(
                xs[eb_mask], probs[eb_mask],
                s=10, color="#1f77b4", alpha=0.5, zorder=2,
                label="EB-policy decisions",
            )

            # Warm-up decisions in orange, drawn on top so they stay visible
            # at y=0.5 where many EB-prior probit-TS draws also cluster. A
            # tiny vertical jitter is added purely so a 700-point stack at
            # exactly 0.5 is visible as a band rather than a single pixel row.
            if warm_mask.any():
                jitter = np.random.default_rng(0).uniform(-0.012, 0.012, warm_mask.sum())
                ax.scatter(
                    xs[warm_mask], probs[warm_mask] + jitter,
                    s=14, color="#d95f02", alpha=0.7, zorder=3, edgecolor="none",
                    label="warm-up decisions ($\\pi{=}0.5$, jittered)",
                )

            if eb_mask.any():
                window = max(8, int(eb_mask.sum() ** 0.5))
                eb_probs = probs[eb_mask]
                eb_x = xs[eb_mask]
                rm = rolling_mean(eb_probs, window)
                ax.plot(eb_x, rm, color="#0b3d91", lw=2.0, zorder=4,
                        label=f"rolling mean (EB, w={window})")

            for b in CLIP_BOUNDS:
                ax.axhline(b, color="red", lw=0.9, ls="--", alpha=0.6, zorder=1)
            ax.axhline(0.5, color="gray", lw=0.9, ls=":", alpha=0.6, zorder=1)

            ax.set_title(AGENT_LABELS[agent], fontsize=12)
            ax.set_xlabel("decision count")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.25)
            if agent == AGENT_ORDER[0]:
                ax.set_ylabel("probit-TS $\\pi(a{=}1\\mid s)$")
            ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

        fig.suptitle(
            "EB action_prob trajectory --- per-dyad week-1 warm-up vs learned policy",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = FIG_DIR / "fig3_action_prob.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"wrote {out_path}")

        # ---------- Figure 2: per-dyad average reward ----------
        per_dyad: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {a: [] for a in AGENT_ORDER}
        )
        recruit_order: dict[str, int] = {}
        for sd in study:
            if sd.decision_type not in AGENT_ORDER:
                continue
            per_dyad[sd.group_id][sd.decision_type].append(float(sd.reward))
            recruit_order.setdefault(sd.group_id, len(recruit_order) + 1)

        # Order groups by recruitment.
        ordered_groups = sorted(recruit_order, key=lambda g: recruit_order[g])

        fig, axes = plt.subplots(
            1, 3, figsize=(18, 5), gridspec_kw={"wspace": 0.22}
        )
        for ax, agent in zip(axes, AGENT_ORDER):
            xs, ys = [], []
            for gid in ordered_groups:
                rewards = per_dyad[gid][agent]
                if rewards:
                    xs.append(recruit_order[gid])
                    ys.append(np.mean(rewards))
            xs = np.array(xs); ys = np.array(ys)

            if len(xs):
                ax.scatter(xs, ys, s=36, color="#1f77b4", alpha=0.85,
                           label="per-dyad mean reward")
                if len(xs) >= 3:
                    cum = np.cumsum(ys) / np.arange(1, len(ys) + 1)
                    ax.plot(xs, cum, color="#0b3d91", lw=2.0,
                            label="cumulative cohort mean")

            ax.set_title(AGENT_LABELS[agent], fontsize=12)
            ax.set_xlabel("dyad index (recruitment order)")
            ax.set_ylabel("mean reward over dyad")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, loc="best", framealpha=0.9)

        fig.suptitle(
            "Per-dyad mean reward as the cohort accumulates", fontsize=13
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = FIG_DIR / "fig2_per_dyad_avg.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
