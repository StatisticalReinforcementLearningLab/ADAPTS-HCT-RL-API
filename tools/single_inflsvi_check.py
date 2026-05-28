"""
Sanity check: a single Inf-LSVI (Bayesian linear regression) on a
low-dimensional synthetic problem with a known optimal policy.

Question: does the per-dyad learner + smooth allocation pick up the
optimal policy when given enough data and a clean signal? This isolates
the basic learning machinery from the EB-Gradient pooling + the
high-dimensional interaction-coefficient noise that's damping the
full-stack sanity check.

Setup:
  - Feature map: phi(s, a) = [1, a, s, a*s]  (D = 4)
  - Truth: y = beta_0 + beta_a * a + beta_s * s + beta_as * a * s + epsilon
  - Default truth: (0, 1.5, 0.5, 0.3) — action coefficient +1.5 dominates
    the state contribution (+0.5), so optimal action is a=1 across all s.
  - State distribution: s ~ Uniform[0, 1]; behavior policy a ~ Bernoulli(0.5).
  - Bayesian regression with prior N(0, tau_prior^2 * I) on theta and
    observation noise variance sigma_noise^2. Posterior at n obs:
      Sigma_n = (sigma_noise^-2 Phi^T Phi + tau_prior^-2 I)^{-1}
      theta_n = sigma_noise^-2 Sigma_n Phi^T y
  - Smooth allocation: pi(s) = E_{theta ~ N(theta_n, Sigma_n)}[rho(Delta_phi^T theta)]
    where rho is the generalized logistic from the live algorithm.

We plot for increasing n:
  - Posterior mean of each coefficient against the true value
  - pi(s=0.5) (a canonical "middle" state) — should rise from 0.5 to ~L_max
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Reuse the live smooth-allocation function so the test exercises the
# *same* MC integrator the production policy uses.
from app.algorithms.eb_gradient import smooth_allocation_prob

FIG_DIR = REPO_ROOT.parent / "Monitoring_Algorithm" / "figures" / "single_inflsvi_check"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- helpers

def phi(s: float, a: int) -> np.ndarray:
    """phi(s, a) = [1, a, s, a*s]."""
    return np.asarray([1.0, float(a), float(s), float(a) * float(s)], dtype=np.float64)


def delta_phi(s: float) -> np.ndarray:
    return phi(s, 1) - phi(s, 0)


def bayesian_lsvi_posterior(
    X: np.ndarray,
    y: np.ndarray,
    sigma_noise_sq: float,
    tau_prior_sq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form Bayesian linear regression posterior."""
    D = X.shape[1]
    precision = (X.T @ X) / sigma_noise_sq + (1.0 / tau_prior_sq) * np.eye(D)
    cov = np.linalg.inv(precision)
    mean = cov @ (X.T @ y) / sigma_noise_sq
    return mean, cov


# ----------------------------------------------------------------- run

def main():
    # Truth.
    beta_star = np.asarray([0.0, 1.5, 0.5, 0.3], dtype=np.float64)
    sigma_noise = 0.5

    # Algorithm hyperparameters (same defaults as live algorithm).
    tau_prior_sq = 1.0
    sigma_noise_sq = sigma_noise ** 2

    # Pre-sampled standard-normal bank for smooth allocation MC.
    rng_mc = np.random.default_rng(12345)
    z_bank = rng_mc.standard_normal(500).astype(np.float64)

    # Generate synthetic data.
    rng = np.random.default_rng(0)
    N_max = 400
    states = rng.uniform(0.0, 1.0, N_max)
    actions = rng.integers(0, 2, N_max).astype(np.float64)
    noise = rng.normal(0.0, sigma_noise, N_max)
    X = np.vstack([phi(s, int(a)) for s, a in zip(states, actions)])
    y = X @ beta_star + noise

    # Sweep posterior over growing data sizes.
    ns = sorted(set(list(range(5, 50, 5)) + list(range(50, N_max + 1, 25))))
    coef_means = []  # list of D-vectors
    coef_stds = []
    pis_mid = []     # pi at s = 0.5
    pis_lo = []      # pi at s = 0.0
    pis_hi = []      # pi at s = 1.0
    contrast_m = []  # action contrast m at s=0.5
    contrast_v = []  # action contrast v at s=0.5

    for n in ns:
        mean_n, cov_n = bayesian_lsvi_posterior(
            X[:n], y[:n], sigma_noise_sq, tau_prior_sq
        )
        coef_means.append(mean_n)
        coef_stds.append(np.sqrt(np.diag(cov_n)))

        for s_target, store in ((0.0, pis_lo), (0.5, pis_mid), (1.0, pis_hi)):
            d = delta_phi(s_target)
            m = float(d @ mean_n)
            v = float(d @ cov_n @ d)
            store.append(smooth_allocation_prob(m, v, z_bank))
            if s_target == 0.5:
                contrast_m.append(m)
                contrast_v.append(v)

    coef_means = np.vstack(coef_means)  # (n_grid, D)
    coef_stds = np.vstack(coef_stds)

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={"wspace": 0.25})

    # Panel 1: posterior mean ± std of each coefficient vs. truth
    ax = axes[0]
    coef_names = [r"$\beta_0$ (intercept)", r"$\beta_a$ (action)",
                  r"$\beta_s$ (state)", r"$\beta_{as}$ (action·state)"]
    colors = ["#444444", "#d62728", "#1f77b4", "#9467bd"]
    for d, (name, col) in enumerate(zip(coef_names, colors)):
        ax.plot(ns, coef_means[:, d], color=col, label=name, lw=1.8)
        ax.fill_between(ns, coef_means[:, d] - coef_stds[:, d],
                        coef_means[:, d] + coef_stds[:, d],
                        color=col, alpha=0.18)
        ax.axhline(beta_star[d], color=col, ls=":", lw=1.2, alpha=0.8)
    ax.set_xlabel("n (observations)")
    ax.set_ylabel("coefficient value")
    ax.set_title("Inf-LSVI posterior mean ± 1 SD vs. truth (dotted)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    # Panel 2: smooth-allocation pi at three canonical states
    ax = axes[1]
    ax.plot(ns, pis_lo, "o-", color="#1f77b4", label=r"$\pi(s{=}0)$")
    ax.plot(ns, pis_mid, "s-", color="#d62728", label=r"$\pi(s{=}0.5)$")
    ax.plot(ns, pis_hi, "^-", color="#2ca02c", label=r"$\pi(s{=}1)$")
    ax.axhline(0.5, color="gray", ls=":", lw=0.9, alpha=0.6)
    ax.axhline(0.8, color="red", ls="--", lw=0.9, alpha=0.6, label=r"$L_{\max}=0.8$")
    ax.axhline(0.2, color="red", ls="--", lw=0.9, alpha=0.6)
    ax.set_xlabel("n (observations)")
    ax.set_ylabel(r"smooth allocation $\pi(a{=}1\mid s)$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Policy commitment vs. data accumulation")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)

    # Panel 3: action contrast m and posterior variance v at s=0.5
    ax = axes[2]
    m_arr = np.asarray(contrast_m)
    v_arr = np.asarray(contrast_v)
    ax.plot(ns, m_arr, "s-", color="#d62728", label=r"$m = \Delta\phi^\top \theta$")
    ax.fill_between(ns, m_arr - np.sqrt(v_arr), m_arr + np.sqrt(v_arr),
                    color="#d62728", alpha=0.18, label=r"$m \pm \sqrt{v}$")
    # True m at s=0.5: beta_a + beta_as * 0.5 = 1.5 + 0.15 = 1.65
    true_m = float(beta_star[1] + beta_star[3] * 0.5)
    ax.axhline(true_m, color="black", ls=":", lw=1.2, label=f"true m = {true_m:.2f}")
    ax.axhline(0.0, color="gray", ls=":", lw=0.6, alpha=0.5)
    ax.set_xlabel("n (observations)")
    ax.set_ylabel("action contrast at $s = 0.5$")
    ax.set_title("Posterior shrinks; mean approaches truth")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        rf"Single Inf-LSVI sanity check — truth $\beta = ({beta_star[0]}, {beta_star[1]}, {beta_star[2]}, {beta_star[3]})$, "
        rf"$\sigma_\epsilon = {sigma_noise}$, prior $\tau^2 = {tau_prior_sq}$",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = FIG_DIR / "fig_single_inflsvi.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path}")

    # Print the final posterior for the record.
    print("\nFinal posterior at n =", ns[-1])
    for d, name in enumerate(coef_names):
        print(f"  {name:30s}: {coef_means[-1, d]:+7.3f}  ± {coef_stds[-1, d]:.3f}    "
              f"(truth {beta_star[d]:+.2f})")
    print(f"  pi(s=0)   = {pis_lo[-1]:.3f}")
    print(f"  pi(s=0.5) = {pis_mid[-1]:.3f}")
    print(f"  pi(s=1)   = {pis_hi[-1]:.3f}")


if __name__ == "__main__":
    main()
