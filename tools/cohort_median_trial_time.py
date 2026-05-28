"""
Cohort-median π(a=1) over trial-calendar decision time, averaged across
several independent simulator seeds to dampen run-level noise.
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
TestingConfig.RL_ALGORITHM = os.environ.get("RL_ALGORITHM", "hybrid_rel_pool")

AGENT_ORDER = ("aya_message", "cp_message", "dyad_game")
AGENT_LABELS = {
    "aya_message": "AYA (twice/day)",
    "cp_message": "CP (daily)",
    "dyad_game": "REL (weekly)",
}

FIG_DIR = REPO_ROOT.parent / "Monitoring_Algorithm/figures/eb_gradient_prior_validation"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_one(seed: int, num_dyads: int, num_weeks: int) -> dict[str, dict]:
    """Run one sim with the given simulator seed, return per-agent curves."""
    # Patch ProtocolTrialSimulator's default seed for this run via monkey-patch.
    from tests import simulate_adapts_hct as sm
    _orig_init = sm.ProtocolTrialSimulator.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs["seed"] = seed
        _orig_init(self, *args, **kwargs)

    sm.ProtocolTrialSimulator.__init__ = _patched_init
    try:
        from app import create_app, db
        from app.models import Action
        from tests.simulate_adapts_hct import run_simulation

        app = create_app("config.TestingConfig")
        out: dict[str, dict] = {}
        with app.app_context():
            db.create_all()
            res = run_simulation(
                app.test_client(),
                base_date=datetime.date(2025, 1, 5),
                num_weeks=num_weeks,
                num_dyads=num_dyads,
                verbose=False,
                group_prefix=f"s{seed}_",
            )

            for agent in AGENT_ORDER:
                events = []
                for a in (
                    Action.query.filter_by(decision_type=agent)
                    .order_by(Action.request_timestamp.asc(), Action.id.asc())
                    .all()
                ):
                    pi1 = float(a.action_prob) if int(a.action) == 1 else 1.0 - float(a.action_prob)
                    events.append((a.group_id, pi1))

                last_pi: dict[str, float] = {}
                xs, medians, n_active = [], [], []
                for k, (gid, pi) in enumerate(events, start=1):
                    last_pi[gid] = pi
                    vals = np.array(list(last_pi.values()))
                    xs.append(k)
                    medians.append(float(np.median(vals)))
                    n_active.append(len(last_pi))
                out[agent] = {
                    "xs": np.asarray(xs, dtype=np.int64),
                    "medians": np.asarray(medians, dtype=np.float64),
                    "n_active": np.asarray(n_active, dtype=np.int64),
                }
        return out
    finally:
        sm.ProtocolTrialSimulator.__init__ = _orig_init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-dyads", type=int, default=25)
    ap.add_argument("--num-weeks", type=int, default=35)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--out", default=str(FIG_DIR / "cohort_median_trial_time.png"))
    args = ap.parse_args()

    seeds = [42 + i for i in range(args.n_runs)]
    print(f"running {args.n_runs} sims with seeds {seeds}")

    per_run = []
    for s in seeds:
        out = run_one(s, args.num_dyads, args.num_weeks)
        for agent in AGENT_ORDER:
            n = len(out[agent]["xs"])
            fm = out[agent]["medians"][-1] if n else float("nan")
            print(f"  seed={s} {agent}: n={n} final-median={fm:.3f}")
        per_run.append(out)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6), gridspec_kw={"wspace": 0.22})
    for ax, agent in zip(axes, AGENT_ORDER):
        L = min(len(r[agent]["medians"]) for r in per_run)
        mat = np.vstack([r[agent]["medians"][:L] for r in per_run])
        xs = per_run[0][agent]["xs"][:L]
        mean = mat.mean(axis=0)
        lo = np.percentile(mat, 5, axis=0)
        hi = np.percentile(mat, 95, axis=0)

        for r in per_run:
            ax.plot(r[agent]["xs"], r[agent]["medians"], color="#1f77b4", lw=0.6, alpha=0.25)
        ax.fill_between(xs, lo, hi, color="#1f77b4", alpha=0.22, label=f"5--95\\% over {args.n_runs} seeds")
        ax.plot(xs, mean, color="#1f77b4", lw=2.2, label="mean median")

        ax.axhline(0.5, color="gray", ls=":", lw=0.9, alpha=0.6)
        ax.axhline(0.8, color="red", ls="--", lw=0.9, alpha=0.5, label="$L_{\\max}=0.8$")
        ax.axhline(0.2, color="red", ls="--", lw=0.9, alpha=0.5)

        ax2 = ax.twinx()
        ax2.plot(per_run[0][agent]["xs"], per_run[0][agent]["n_active"],
                 color="#888888", ls=":", lw=1.2)
        ax2.set_ylabel("active dyads", color="#666666")
        ax2.tick_params(axis="y", labelcolor="#666666")

        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("trial-wide decision index (chronological)")
        ax.set_ylabel(r"cohort median $\pi(a{=}1)$")
        ax.set_title(AGENT_LABELS[agent], fontsize=12)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8.5, loc="lower right", framealpha=0.92)

    fig.suptitle(
        f"Cohort-median $\\pi(a{{=}}1)$ over trial-calendar time "
        f"(mean across {args.n_runs} independent sim seeds)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = Path(args.out)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
