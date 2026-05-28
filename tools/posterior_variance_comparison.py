"""
Posterior-variance shrinkage diagnostic (replaces the standalone
$\\kappa_b$ validation simulator).

For each agent, plot the Frobenius norm of three covariance objects over
the per-dyad agent decision index:

  (1) $\\|\\hat\\Sigma_i\\|_F$    --- per-dyad Inf-LSVI local fit
                                    (no pooling, EB-Gradient algorithm).
  (2) $\\|\\Sigma_i^{\\mathrm{post}}\\|_F$ --- per-dyad EB posterior
                                    (Gaussian product with the pool).
  (3) $\\|\\Sigma_{\\mathrm{pool}}\\|_F$   --- fully-pooled Inf-LSVI fit
                                    (one shared posterior across all dyads).

(1) and (2) come from the same EB-Gradient run; (3) comes from a
separate Inf-LSVI-with-full-pooling run on the same simulator seed.

Per-dyad (1) and (2) are shown as median $+$ 5--95\\% band across the
active dyads at each refresh; (3) is a single line.
"""

from __future__ import annotations

import argparse
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

# Sync /update background thread + suppress callback.
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

AGENT_ORDER = ("aya_message", "cp_message", "dyad_game")
AGENT_LABELS = {
    "aya_message": "AYA (twice/day)",
    "cp_message": "CP (daily)",
    "dyad_game": "REL (weekly)",
}

FIG_DIR = REPO_ROOT.parent / "Monitoring_Algorithm/figures/eb_gradient_prior_validation"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _frobenius(cov_json) -> float:
    """Frobenius norm of a JSON-serialised covariance matrix."""
    C = np.asarray(cov_json, dtype=np.float64)
    return float(np.linalg.norm(C, ord="fro"))


def run_one(algo_name: str, num_dyads: int, num_weeks: int) -> dict:
    """Boot the app under `algo_name`, run the simulator, return raw
    snapshot rows for the three snapshot_types we care about."""
    TestingConfig.RL_ALGORITHM = algo_name
    from app import create_app, db
    from app.models import ModelParameters
    from tests.simulate_adapts_hct import run_simulation

    out: dict = {a: {"local_fit": [], "posterior": [], "pooled": []} for a in AGENT_ORDER}
    app = create_app("config.TestingConfig")
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
            f"[{algo_name}] add_group={results['add_group']} "
            f"action={results['action']} upload_data={results['upload_data']} "
            f"update={results['update']} errors={len(results['errors'])}"
        )
        for agent in AGENT_ORDER:
            # Per-dyad local fit (group_id != None) and EB posterior (same).
            rows = (
                ModelParameters.query
                .filter(ModelParameters.decision_type == agent)
                .filter(ModelParameters.group_id.isnot(None))
                .all()
            )
            for r in rows:
                if r.snapshot_type == "local_fit":
                    out[agent]["local_fit"].append(
                        (int(r.agent_decision_index), _frobenius(r.covariance))
                    )
                elif r.snapshot_type == "posterior":
                    out[agent]["posterior"].append(
                        (int(r.agent_decision_index), _frobenius(r.covariance))
                    )
            # Pooled local fit (group_id is None) — only present for inf_lsvi_pool.
            rows = (
                ModelParameters.query
                .filter(ModelParameters.decision_type == agent)
                .filter(ModelParameters.group_id.is_(None))
                .filter(ModelParameters.snapshot_type == "local_fit")
                .order_by(ModelParameters.agent_decision_index.asc())
                .all()
            )
            for r in rows:
                out[agent]["pooled"].append(
                    (int(r.agent_decision_index), _frobenius(r.covariance))
                )
    return out


def _aggregate(pairs: list[tuple[int, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bucket (agent_idx, value) pairs by agent_idx, return (xs, median, q05, q95)."""
    by_idx: dict[int, list[float]] = defaultdict(list)
    for k, v in pairs:
        by_idx[k].append(float(v))
    xs = np.asarray(sorted(by_idx.keys()), dtype=np.float64)
    if xs.size == 0:
        return xs, np.array([]), np.array([]), np.array([])
    med = np.asarray([float(np.median(by_idx[int(x)])) for x in xs])
    lo = np.asarray([float(np.quantile(by_idx[int(x)], 0.05)) for x in xs])
    hi = np.asarray([float(np.quantile(by_idx[int(x)], 0.95)) for x in xs])
    return xs, med, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-dyads", type=int, default=25)
    ap.add_argument("--num-weeks", type=int, default=35)
    ap.add_argument(
        "--out",
        default=str(FIG_DIR / "posterior_variance_comparison.png"),
    )
    args = ap.parse_args()

    data_eb = run_one("eb_gradient", args.num_dyads, args.num_weeks)
    data_pool = run_one("inf_lsvi_pool", args.num_dyads, args.num_weeks)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), gridspec_kw={"wspace": 0.22})
    for ax, agent in zip(axes, AGENT_ORDER):
        # (1) Per-dyad Inf-LSVI local fit ||Σ̂_i||_F  — from EB-Gradient run.
        xs, med, lo, hi = _aggregate(data_eb[agent]["local_fit"])
        if xs.size:
            ax.fill_between(xs, lo, hi, color="#1f77b4", alpha=0.16)
            ax.plot(xs, med, "-", color="#1f77b4", lw=1.8,
                    label=r"per-dyad $\|\hat\Sigma_i\|_F$ (Inf-LSVI, no pool)")

        # (2) Per-dyad EB posterior ||Σ_i^post||_F  — from EB-Gradient run.
        xs, med, lo, hi = _aggregate(data_eb[agent]["posterior"])
        if xs.size:
            ax.fill_between(xs, lo, hi, color="#d95f02", alpha=0.16)
            ax.plot(xs, med, "-", color="#d95f02", lw=1.8,
                    label=r"per-dyad $\|\Sigma_i^{\mathrm{post}}\|_F$ (EB-Gradient)")

        # (3) Fully-pooled Inf-LSVI ||Σ_pool||_F  — single line.
        xs, med, _, _ = _aggregate(data_pool[agent]["pooled"])
        if xs.size:
            ax.plot(xs, med, "-", color="#2ca02c", lw=2.2,
                    label=r"pooled $\|\Sigma_{\mathrm{pool}}\|_F$ (Inf-LSVI, full pool)")

        ax.set_yscale("log")
        ax.set_xlabel("agent decision index")
        ax.set_ylabel(r"Frobenius norm (log)")
        ax.set_title(AGENT_LABELS[agent], fontsize=12)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8.5, loc="upper right", framealpha=0.92)

    fig.suptitle(
        "Posterior-variance shrinkage: per-dyad Inf-LSVI vs.\\ EB posterior vs.\\ fully-pooled Inf-LSVI",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = Path(args.out)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
