"""Sanity check for EB-Gradient (MAP marginal-likelihood + generalized-logistic
smooth allocation). Mirrors `rerun_sanity_check.py` but boots the app with
RL_ALGORITHM=eb_gradient and writes figures to
Monitoring_Algorithm/figures/sanity_check_eb_gradient/.

See Prior_Construction_Note.tex §EB-Gradient and §Generalized-logistic smooth
allocation for the algorithm.
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

# Force EB-Gradient before create_app reads config.RL_ALGORITHM.
import os
os.environ["RL_ALGORITHM"] = "eb_gradient"

# Sync the /update background thread (same pattern as the MoM sanity check).
import app.routes.update as _update_module
import threading as _threading

class _SyncThread(_threading.Thread):
    def start(self):  # type: ignore[override]
        self.run()

_update_module.Thread = _SyncThread

# Suppress the callback POST.
import requests as _requests
_requests.post = lambda *a, **kw: type("R", (), {"status_code": 200, "text": ""})()

from config import TestingConfig
TestingConfig.SAMPLE_BUFFER_UNIFORMS = 50_000
TestingConfig.SAMPLE_BUFFER_NORMALS = 2_000_000
TestingConfig.RL_ALGORITHM = "eb_gradient"

from app import create_app, db
from app.models import Action, StudyData
from tests.simulate_adapts_hct import run_simulation

FIG_DIR = REPO_ROOT.parent / "Monitoring_Algorithm" / "figures" / "sanity_check_eb_gradient"
FIG_DIR.mkdir(parents=True, exist_ok=True)

AGENT_ORDER = ("aya_message", "cp_message", "dyad_game")
AGENT_LABELS = {
    "aya_message": "AYA (twice/day)",
    "cp_message": "CP (daily)",
    "dyad_game": "REL (weekly)",
}
CLIP_BOUNDS = (0.2, 0.8)  # L_min, L_max for generalized-logistic smooth allocation


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

        results = run_simulation(
            app.test_client(),
            base_date=datetime.date(2025, 1, 5),
            num_weeks=35,
            num_dyads=25,
            verbose=False,
        )
        print(
            f"algo={app.config.get('RL_ALGORITHM')} "
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
                (a.random_state or {}).get("mode", "smooth_logistic") for a in rows
            ])
            warm_mask = modes == "warmup"
            eb_mask = ~warm_mask

            ax.scatter(
                xs[eb_mask], probs[eb_mask],
                s=10, color="#1f77b4", alpha=0.5, zorder=2,
                label="EB-Gradient decisions",
            )

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
                        label=f"rolling mean (EBG, w={window})")

            for b in CLIP_BOUNDS:
                ax.axhline(b, color="red", lw=0.9, ls="--", alpha=0.6, zorder=1)
            ax.axhline(0.5, color="gray", lw=0.9, ls=":", alpha=0.6, zorder=1)

            ax.set_title(AGENT_LABELS[agent], fontsize=12)
            ax.set_xlabel("decision count")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.25)
            if agent == AGENT_ORDER[0]:
                ax.set_ylabel("smooth-logistic $\\pi(a{=}1\\mid s)$")
            ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

        fig.suptitle(
            "EB-Gradient action_prob trajectory --- "
            "per-dyad week-1 warm-up vs MAP-EB + generalized-logistic policy",
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
            "EB-Gradient: per-dyad mean reward as the cohort accumulates",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = FIG_DIR / "fig2_per_dyad_avg.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"wrote {out_path}")

        # ---------- Figure 4: post-warmup commitment per dyad ----------
        # For each dyad, look at the FIRST N_EARLY post-warmup decisions
        # (which is where "well warmed up given a pre-pooled hyperprior"
        # should be visible). If new dyads inherit the cohort's committed
        # policy, these π values should cluster near L_min / L_max and not
        # bounce near 0.5. We compare against the dyad-level mean π over
        # the same first window — a horizontal line in dyad-order.
        N_EARLY = 14   # one full week of EB decisions for AYA; ~2 weeks CP; all of REL post-warmup
        by_dyad: dict[tuple[str, str], list[Action]] = defaultdict(list)
        for a in actions:
            by_dyad[(a.decision_type, a.group_id)].append(a)

        fig, axes = plt.subplots(
            1, 3, figsize=(18, 5), sharey=True, gridspec_kw={"wspace": 0.12}
        )
        for ax, agent in zip(axes, AGENT_ORDER):
            dyads = sorted({gid for (dt, gid), _ in by_dyad.items() if dt == agent})
            dyad_order = {gid: i + 1 for i, gid in enumerate(dyads)}

            xs_all: list[float] = []
            ys_all: list[float] = []
            dyad_means_x: list[float] = []
            dyad_means_y: list[float] = []
            for gid in dyads:
                rows = sorted(by_dyad[(agent, gid)], key=lambda r: r.decision_idx)
                # Split warmup vs EB by `mode`
                eb_rows = [r for r in rows if (r.random_state or {}).get("mode") != "warmup"]
                if not eb_rows:
                    continue
                early = eb_rows[:N_EARLY]
                early_probs = [float(r.action_prob) for r in early]
                jitter = np.random.default_rng(dyad_order[gid]).uniform(-0.15, 0.15, len(early_probs))
                xs_all.extend(dyad_order[gid] + jitter)
                ys_all.extend(early_probs)
                if early_probs:
                    dyad_means_x.append(dyad_order[gid])
                    dyad_means_y.append(float(np.mean(early_probs)))

            ax.scatter(xs_all, ys_all, s=14, color="#1f77b4", alpha=0.5, zorder=2,
                       label=f"first {N_EARLY} post-warmup decisions")
            ax.plot(dyad_means_x, dyad_means_y, "o-", color="#d95f02", lw=1.5,
                    markersize=6, zorder=3, label="dyad mean π over the window")
            for bnd in CLIP_BOUNDS:
                ax.axhline(bnd, color="red", lw=0.9, ls="--", alpha=0.6, zorder=1)
            ax.axhline(0.5, color="gray", lw=0.9, ls=":", alpha=0.6, zorder=1)

            ax.set_title(AGENT_LABELS[agent], fontsize=12)
            ax.set_xlabel("dyad index (recruitment order)")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.25)
            if agent == AGENT_ORDER[0]:
                ax.set_ylabel("smooth-logistic $\\pi(a{=}1\\mid s)$")
            ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

        fig.suptitle(
            "Post-warmup commitment: are new dyads using the cohort policy from day 8?",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = FIG_DIR / "fig4_post_warmup.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
