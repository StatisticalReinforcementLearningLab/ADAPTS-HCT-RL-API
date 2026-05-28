"""
Within-dyad longitudinal view: pick one representative dyad per agent
and plot its EB posterior variance and rolling-mean sampling probability
over its own decision index.

For each agent (AYA / CP / REL):

  - Top row: $\\|\\Sigma_i^{\\mathrm{post}}\\|_F$ on log y, one point per
    /update (the dyad's posterior covariance from the EmpiricalBayes
    snapshot table). Cohort median across all dyads at the same
    within-dyad index is plotted as a faint grey reference.

  - Bottom row: rolling-mean $\\pi(a{=}1 \\mid s_k)$ over the dyad's
    decisions. Cohort median again as faint grey reference. Warm-up
    decisions plotted in orange to mark the $\\pi = 0.5$ window.

The chosen dyad is configurable via ``--dyad-index`` (default: 12, the
middle of the 25-dyad recruitment order).
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
    ap.add_argument("--dyad-index", type=int, default=12,
                    help="1-based recruitment index of the focal dyad.")
    ap.add_argument("--out", default=str(FIG_DIR / "within_dyad_trajectory.png"))
    ap.add_argument("--bottom-only", action="store_true",
                    help="Plot only the sampling-probability row (drop the "
                         "posterior-variance row), wider per panel.")
    args = ap.parse_args()

    from app import create_app, db
    from app.models import EmpiricalBayesSnapshot, Action, Group
    from tests.simulate_adapts_hct import run_simulation

    focal_gid = f"dyad_{args.dyad_index:03d}"
    print(f"focal dyad: {focal_gid}")

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
            f"update={results['update']} errors={len(results['errors'])}"
        )
        # Sanity check that the focal dyad exists.
        focal = Group.query.filter_by(group_id=focal_gid).first()
        if focal is None:
            raise SystemExit(
                f"Focal dyad {focal_gid} not found. Available dyads: "
                f"{sorted({g.group_id for g in Group.query.all()})}"
            )

        # Posterior variance trajectories: per-dyad, indexed by agent_decision_index.
        # Fully-pooled learners (e.g. REL under hybrid_rel_pool) instead write
        # one shared "local_fit" snapshot per refresh with group_id=None; we
        # surface that as `pooled_posterior[agent] = [(refresh_idx, ||Σ||_F), ...]`.
        posterior: dict[str, dict[str, list[tuple[int, float]]]] = {
            a: defaultdict(list) for a in AGENT_ORDER
        }
        pooled_posterior: dict[str, list[tuple[int, float]]] = {a: [] for a in AGENT_ORDER}
        per_dyad_rows = (
            EmpiricalBayesSnapshot.query
            .filter_by(snapshot_type="posterior")
            .filter(EmpiricalBayesSnapshot.group_id.isnot(None))
            .all()
        )
        agents_with_per_dyad = set()
        for r in per_dyad_rows:
            if r.decision_type in posterior:
                cov = np.asarray(r.covariance, dtype=np.float64)
                posterior[r.decision_type][r.group_id].append(
                    (int(r.agent_decision_index), float(np.linalg.norm(cov, ord="fro")))
                )
                agents_with_per_dyad.add(r.decision_type)

        for agent in AGENT_ORDER:
            if agent in agents_with_per_dyad:
                continue
            pooled = (
                EmpiricalBayesSnapshot.query
                .filter_by(snapshot_type="local_fit", decision_type=agent, group_id=None)
                .order_by(EmpiricalBayesSnapshot.agent_decision_index.asc())
                .all()
            )
            pooled_posterior[agent] = [
                (i + 1, float(np.linalg.norm(np.asarray(r.covariance, dtype=np.float64), ord="fro")))
                for i, r in enumerate(pooled)
            ]

        # Action trajectories.
        actions: dict[str, dict[str, list[tuple[int, float, str]]]] = {
            a: defaultdict(list) for a in AGENT_ORDER
        }
        for a in (
            Action.query.order_by(
                Action.decision_type, Action.group_id, Action.decision_idx
            ).all()
        ):
            if a.decision_type in actions:
                # action_prob stores Pr(chosen action). Convert to π(a=1).
                pi1 = float(a.action_prob) if int(a.action) == 1 else 1.0 - float(a.action_prob)
                actions[a.decision_type][a.group_id].append(
                    (int(a.decision_idx), pi1,
                     (a.random_state or {}).get("mode", "eb"))
                )

    # ---------- figure ----------
    if args.bottom_only:
        fig, axes_bottom = plt.subplots(
            1, 3, figsize=(18, 5.0), gridspec_kw={"wspace": 0.22}
        )
        axes_top = [None] * 3
        axes_bot = list(axes_bottom)
    else:
        fig, axes = plt.subplots(
            2, 3, figsize=(18, 8.5), gridspec_kw={"hspace": 0.32, "wspace": 0.22}
        )
        axes_top = list(axes[0])
        axes_bot = list(axes[1])
    fig.suptitle(
        rf"Within-dyad longitudinal view: focal dyad = \texttt{{{focal_gid}}}",
        fontsize=13,
    )

    for col, agent in enumerate(AGENT_ORDER):
        # ----- top: posterior variance (skipped if --bottom-only) -----
        if not args.bottom_only:
            ax = axes_top[col]
            pooled_pairs = pooled_posterior.get(agent, [])
            if pooled_pairs:
                xs, ys = zip(*pooled_pairs)
                ax.plot(xs, ys, "o-", color="#1f77b4", lw=1.8, ms=4,
                        label=rf"$\|\Sigma^{{\mathrm{{pool}}}}\|_F$ (shared)")
                ax.set_xlabel("pool refresh index")
            else:
                per_idx: dict[int, list[float]] = defaultdict(list)
                for gid, pairs in posterior[agent].items():
                    for k, v in pairs:
                        per_idx[k].append(v)
                idxs = sorted(per_idx)
                if idxs:
                    ax.plot(idxs, [float(np.median(per_idx[k])) for k in idxs],
                            color="#888888", lw=1.4, ls=":", label="cohort median")
                focal_pairs = sorted(posterior[agent].get(focal_gid, []))
                if focal_pairs:
                    xs, ys = zip(*focal_pairs)
                    ax.plot(xs, ys, "o-", color="#1f77b4", lw=1.8, ms=4,
                            label=rf"$\|\Sigma_i^{{\mathrm{{post}}}}\|_F$ for {focal_gid}")
                ax.set_xlabel("within-dyad decision index")
            ax.set_yscale("log")
            ax.set_ylabel(r"$\|\Sigma^{\mathrm{post}}\|_F$ (log)")
            ax.set_title(f"{AGENT_LABELS[agent]} --- posterior variance", fontsize=11)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=9, loc="best", framealpha=0.92)

        # ----- bottom: rolling-mean sampling probability -----
        ax = axes_bot[col]
        # cohort rolling-mean median
        all_dyad_curves: list[np.ndarray] = []
        max_T = max((len(actions[agent][g]) for g in actions[agent]), default=0)
        for gid in actions[agent]:
            probs = np.asarray([p for _, p, _ in actions[agent][gid]], dtype=np.float64)
            if len(probs) < 2:
                continue
            w = max(4, int(len(probs) ** 0.5))
            rm = rolling_mean(probs, w)
            padded = np.full(max_T, np.nan)
            padded[: len(rm)] = rm
            all_dyad_curves.append(padded)
        if all_dyad_curves:
            mat = np.vstack(all_dyad_curves)
            ax.plot(np.arange(1, max_T + 1), np.nanmedian(mat, axis=0),
                    color="#888888", lw=1.4, ls=":", label="cohort median (rolling)")
        focal_acts = actions[agent].get(focal_gid, [])
        if focal_acts:
            probs = np.asarray([p for _, p, _ in focal_acts], dtype=np.float64)
            modes = np.asarray([m for _, _, m in focal_acts])
            xs = np.arange(1, len(probs) + 1)
            # Mark warm-up decisions in orange, EB decisions in blue.
            warm = modes == "warmup"
            eb = ~warm
            if warm.any():
                ax.scatter(xs[warm], probs[warm], color="#d95f02", s=18,
                           alpha=0.7, zorder=3, label="warm-up ($\\pi{=}0.5$)")
            if eb.any():
                ax.scatter(xs[eb], probs[eb], color="#1f77b4", s=10, alpha=0.45,
                           zorder=2, label="EB decisions")
                w = max(4, int(eb.sum() ** 0.5))
                rm = rolling_mean(probs[eb], w)
                ax.plot(xs[eb], rm, color="#1f77b4", lw=2.0, zorder=4,
                        label=rf"rolling mean (w={w})")
        ax.axhline(0.5, color="gray", ls=":", lw=0.9, alpha=0.6)
        ax.axhline(0.8, color="red", ls="--", lw=0.9, alpha=0.5)
        ax.axhline(0.2, color="red", ls="--", lw=0.9, alpha=0.5)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("within-dyad decision index")
        ax.set_ylabel(r"$\pi(a{=}1 \mid s)$")
        ax.set_title(f"{AGENT_LABELS[agent]} --- sampling probability", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8.5, loc="lower right", framealpha=0.92)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = Path(args.out)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
