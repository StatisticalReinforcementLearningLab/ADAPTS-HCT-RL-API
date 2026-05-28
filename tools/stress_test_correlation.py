"""
Stress test: inject pairwise correlation between two AYA continuous
features in the RL API feature map and rerun the EB-Gradient
prior-validation diagnostics under $\\rho \\in \\{0, 0.5, 0.95\\}$.

Mirrors the standalone Phase-6 stress test in
``Prior_Construction/code/phase6_stress.py``, but now executed inside
the live RL API so the diagnostics use the actual EB-Gradient learner,
the actual Bellman targets, and the actual smooth allocation.

The correlation is injected post-feature-build by overwriting one column
of $\\phi$ with $\\rho \\cdot$ (the other column) $+ \\sqrt{1-\\rho^2}
\\cdot$ (its own residual). This keeps the feature dimension fixed and
matches the linear-model assumption of the stress test.

The two paired features are chosen from the AYA continuous variables;
the user can swap them via ``--pair COL_A,COL_B`` (default: action·indicator
columns of ``aya_app_burden`` and ``aya_app_engagement``).
"""

from __future__ import annotations

import argparse
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

# Patch /update Thread + suppress callback.
import app.routes.update as _update_module

class _SyncThread(_threading.Thread):
    def start(self):  # type: ignore[override]
        self.run()

_update_module.Thread = _SyncThread

import requests as _requests
_requests.post = lambda *a, **kw: type("R", (), {"status_code": 200, "text": ""})()

from config import TestingConfig
import os
TestingConfig.SAMPLE_BUFFER_UNIFORMS = 50_000
TestingConfig.SAMPLE_BUFFER_NORMALS = 2_000_000
TestingConfig.RL_ALGORITHM = os.environ.get("RL_ALGORITHM", "hybrid_rel_pool")

import app.feature_builder as fb_module
from tools.validate_prior_in_rl_api import run_simulation_and_collect, AGENT_ORDER, AGENT_LABELS

FIG_DIR = REPO_ROOT.parent / "Monitoring_Algorithm/figures/eb_gradient_prior_validation"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ----- correlation injection ------------------------------------------------

_ORIG_EXPAND = fb_module.ProtocolRLFeatureBuilder.expand_base_to_phi

# Globals set by apply_correlation_patch():
_RHO: float = 0.0
_MODE: str = "pair"  # "pair" or "all"
_PAIR: tuple[str, str] = ("aya_app_burden", "aya_app_engagement")
_LATENT_RNG: np.random.Generator | None = None  # deterministic per-call draws


def _value_indices(agent: str, var_name: str) -> tuple[int, int]:
    """Return (main_value_idx, interaction_value_idx) for var_name in φ(s, a)
    under the shared-I layout: φ = [1, a, I, (v_j I)_j, a I, (a v_j I)_j].
    """
    fb = fb_module.ProtocolRLFeatureBuilder(agent)
    names = fb.variable_names
    if var_name not in names:
        raise ValueError(f"{var_name} not in {agent} variables: {names}")
    j = names.index(var_name)
    J = fb.n_vars
    main_idx = 3 + j               # [1, a, I, v_0·I, v_1·I, ...]; v_j at 3+j
    inter_idx = 3 + J + 1 + j      # [..., aI, a·v_0·I, ...]; a·v_j·I at 3+J+1+j
    return main_idx, inter_idx


def _expand_with_correlation(self, base, action):
    phi = _ORIG_EXPAND(self, base, action)
    if _RHO <= 0.0:
        return phi

    fb = fb_module.ProtocolRLFeatureBuilder(self.decision_type)
    J = fb.n_vars
    rho = float(_RHO)

    if _MODE == "pair":
        if self.decision_type != "aya_message":
            return phi
        try:
            main_a, inter_a = _value_indices("aya_message", _PAIR[0])
            main_b, inter_b = _value_indices("aya_message", _PAIR[1])
        except ValueError:
            return phi
        s = np.sqrt(1.0 - rho * rho)
        phi[main_b]  = rho * float(phi[main_a])  + s * float(phi[main_b])
        phi[inter_b] = rho * float(phi[inter_a]) + s * float(phi[inter_b])
        return phi

    # _MODE == "all": equicorrelated mix via shared latent u.
    # Replace each value column v_j with sqrt(ρ)·u + sqrt(1-ρ)·v_j, where
    # u ~ N(0,1) is drawn ONCE per decision and shared across all J vars.
    # Doing so on both main (v_j*I) and interaction (a*v_j*I) keeps the
    # action-contrast structure consistent.
    rng = _LATENT_RNG if _LATENT_RNG is not None else np.random
    u = float(rng.standard_normal())
    sqrt_rho = np.sqrt(rho)
    sqrt_1mr = np.sqrt(1.0 - rho)
    main_start = 3
    inter_start = 3 + J + 1
    for j in range(J):
        main_idx = main_start + j
        inter_idx = inter_start + j
        I = float(phi[2])  # shared indicator
        # Only mix when observed (I=1). When I=0 the values are already 0.
        if I == 0.0:
            continue
        a_eff = 1.0 if action else 0.0
        phi[main_idx]  = sqrt_rho * u * I            + sqrt_1mr * float(phi[main_idx])
        phi[inter_idx] = sqrt_rho * u * I * a_eff    + sqrt_1mr * float(phi[inter_idx])
    return phi


def apply_correlation_patch(rho: float, pair: tuple[str, str], mode: str = "pair",
                            latent_seed: int = 12345) -> None:
    global _RHO, _PAIR, _MODE, _LATENT_RNG
    _RHO = float(rho)
    _PAIR = pair
    _MODE = mode
    _LATENT_RNG = np.random.default_rng(latent_seed)
    fb_module.ProtocolRLFeatureBuilder.expand_base_to_phi = _expand_with_correlation


def restore_feature_builder() -> None:
    fb_module.ProtocolRLFeatureBuilder.expand_base_to_phi = _ORIG_EXPAND


# ----- per-ρ run + summary plot --------------------------------------------

def _summarize_run(data: dict) -> dict[str, dict]:
    """Compress the raw snapshot dump into the per-ρ summary plotted below:
    hyper trace trajectory, posterior trace trajectory, cumulative π."""
    out: dict[str, dict] = {a: {} for a in AGENT_ORDER}
    for agent in AGENT_ORDER:
        hyp = data["hyper"][agent]
        out[agent]["hyper_n"] = [h["sample_size"] for h in hyp]
        out[agent]["hyper_tr"] = [h["trace"] for h in hyp]
        out[agent]["hyper_theta_a"] = [h["theta_action"] for h in hyp]

        post = data["posterior"][agent]
        out[agent]["post_x"] = [agent_idx for agent_idx, _ in post]
        out[agent]["post_med"] = [float(np.median(t)) for _, t in post]
        out[agent]["post_lo"] = [float(np.quantile(t, 0.05)) for _, t in post]
        out[agent]["post_hi"] = [float(np.quantile(t, 0.95)) for _, t in post]

        acts = data["actions"][agent]
        arr = np.asarray([a for a, _, _ in acts], dtype=np.float64)
        xs = np.arange(1, len(arr) + 1) if len(arr) > 0 else np.array([1])
        cum = np.cumsum(arr) / xs if len(arr) > 0 else np.array([0.0])
        out[agent]["pi_x"] = xs
        out[agent]["pi_cum"] = cum
    return out


def plot_stress_grid(summaries: dict[float, dict], out_path: Path,
                     include_hyper: bool = False) -> None:
    rhos = sorted(summaries.keys())
    colors = {0.0: "#1f77b4", 0.5: "#ff7f0e", 0.95: "#d62728"}

    n_rows = 3 if include_hyper else 2
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 9 if n_rows == 2 else 13),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.25})
    if not include_hyper:
        # Pad with a None row so the index arithmetic below stays the same.
        import numpy as _np
        axes = _np.vstack([_np.array([None, None, None], dtype=object), axes])
    if _MODE == "pair":
        title = (
            rf"Stress test: pairwise feature correlation on AYA "
            rf"(\verb|aya_app_burden| $\leftrightarrow$ \verb|aya_app_engagement|, value cols), "
            rf"$\rho \in \{{{', '.join(f'{r:.2f}' for r in rhos)}\}}$"
        )
    else:
        title = (
            rf"Stress test (all-pairwise): equicorrelation across every value column "
            rf"(AYA, CP, REL), shared latent per decision, "
            rf"$\rho \in \{{{', '.join(f'{r:.2f}' for r in rhos)}\}}$"
        )
    fig.suptitle(title, fontsize=13)

    for col, agent in enumerate(AGENT_ORDER):
        # Row 0: hyper Σ_0 trace vs N, one line per ρ. (Optional)
        if include_hyper:
            ax = axes[0, col]
            for rho in rhos:
                s = summaries[rho][agent]
                if s["hyper_n"]:
                    ax.plot(s["hyper_n"], s["hyper_tr"], "o-", color=colors.get(rho, "k"),
                            lw=1.6, ms=3.5, label=rf"$\rho = {rho:.2f}$")
            ax.set_yscale("log")
            ax.set_title(f"{AGENT_LABELS[agent]} --- hyper $\\mathrm{{tr}}\\,\\hat\\Sigma_0$",
                         fontsize=11)
            ax.set_xlabel("active dyads $N$")
            ax.set_ylabel("trace (log)")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, loc="best")

        # Row 1: per-dyad Σ_i^post median trace vs refresh index.
        ax = axes[1, col]
        for rho in rhos:
            s = summaries[rho][agent]
            if s["post_x"]:
                ax.plot(s["post_x"], s["post_med"], "-", color=colors.get(rho, "k"),
                        lw=1.6, label=rf"$\rho = {rho:.2f}$")
                ax.fill_between(s["post_x"], s["post_lo"], s["post_hi"],
                                color=colors.get(rho, "k"), alpha=0.10)
        ax.set_yscale("log")
        ax.set_title(f"{AGENT_LABELS[agent]} --- per-dyad $\\mathrm{{tr}}\\,\\hat\\Sigma_i^{{\\mathrm{{post}}}}$ (med, 5–95%)",
                     fontsize=11)
        ax.set_xlabel("agent decision index")
        ax.set_ylabel("trace (log)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")

        # Row 2: cumulative Pr(A=1) vs decision count.
        ax = axes[2, col]
        for rho in rhos:
            s = summaries[rho][agent]
            if len(s["pi_x"]) > 1:
                final = float(s["pi_cum"][-1])
                ax.plot(s["pi_x"], s["pi_cum"], color=colors.get(rho, "k"), lw=1.6,
                        label=rf"$\rho = {rho:.2f}$ (final {final:.3f})")
        ax.axhline(0.5, color="gray", ls=":", lw=0.9, alpha=0.6)
        ax.axhline(0.8, color="red", ls="--", lw=0.9, alpha=0.5)
        ax.axhline(0.2, color="red", ls="--", lw=0.9, alpha=0.5)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{AGENT_LABELS[agent]} --- cumulative $\\Pr(A{{=}}1)$",
                     fontsize=11)
        ax.set_xlabel("decision count")
        ax.set_ylabel("cum. mean")
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
    ap.add_argument("--rhos", type=str, default="0.0,0.5,0.95",
                    help="Comma-separated list of correlation values.")
    ap.add_argument("--pair", type=str,
                    default="aya_app_burden,aya_app_engagement",
                    help="Two AYA variables to correlate (comma-separated).")
    ap.add_argument("--mode", choices=("pair", "all"), default="pair",
                    help="pair = correlate one AYA variable pair only; "
                         "all = equicorrelate every value column across all agents via a shared latent.")
    ap.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default depends on --mode).",
    )
    args = ap.parse_args()

    rhos = [float(x) for x in args.rhos.split(",")]
    pair = tuple(args.pair.split(","))
    if len(pair) != 2:
        raise SystemExit(f"--pair must be 'A,B'; got {args.pair!r}")

    out_path = Path(args.out) if args.out else (
        FIG_DIR / ("stress_test_correlation.png" if args.mode == "pair"
                   else "stress_test_correlation_all.png")
    )

    summaries: dict[float, dict] = {}
    for rho in rhos:
        if args.mode == "pair":
            print(f"\n=== running mode=pair rho={rho:.2f} (pair = {pair[0]}, {pair[1]}) ===")
        else:
            print(f"\n=== running mode=all rho={rho:.2f} (equicorrelated across all agents/vars) ===")
        if rho > 0.0:
            apply_correlation_patch(rho, pair, mode=args.mode)
        else:
            restore_feature_builder()
        try:
            data = run_simulation_and_collect(
                num_dyads=args.num_dyads, num_weeks=args.num_weeks
            )
            summaries[rho] = _summarize_run(data)
        finally:
            restore_feature_builder()

    plot_stress_grid(summaries, out_path)


if __name__ == "__main__":
    main()
