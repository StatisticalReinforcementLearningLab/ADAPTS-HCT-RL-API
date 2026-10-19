"""
Allocation rule of the ``inf_lsvi_eb`` learner (implementation contract D10;
main_new.tex §3.6 eqs. (rho), (contrast-post), (alloc), (alloc-mc); §5.9).

    rho(x) = L_min + (L_max - L_min) / (1 + c exp(-b x))^kappa
           = L_min + (L_max - L_min) exp(-kappa logaddexp(0, log c - b x)),

evaluated in log space so it is stable for any slope. The randomization
probability is the Monte Carlo average over a fixed bank
``Z = np.random.default_rng(mc_seed).standard_normal(M)``:

    pi(mu, v) = (1/M) sum_j rho(mu + sqrt(max(v, 0)) Z_j),

with ``mu = f' alpha_EB`` and ``v = f' Sigma_EB f``. The slope is per agent:
``b = b_rho * r_max`` on the normalized reward scale (40 / 100 / 100 + 40 w_adh
for AYA / CP / REL; eq. (rho-params)). The arithmetic matches the testbed's
``Allocator`` operation for operation, so the same bank gives the same
``pi`` bit for bit.
"""

from __future__ import annotations

import threading

import numpy as np

_BANK_CACHE: dict[tuple[int, int], np.ndarray] = {}
_BANK_LOCK = threading.Lock()


def rho(x, l_min: float, l_max: float, c: float, b: float, kappa: float):
    """Generalized logistic of eq. (rho), in log space."""
    return l_min + (l_max - l_min) * np.exp(-kappa * np.logaddexp(0.0, np.log(c) - b * x))


def default_bank(seed: int, M: int) -> np.ndarray:
    """``np.random.default_rng(seed).standard_normal(M)`` (read-only, cached)."""
    key = (int(seed), int(M))
    with _BANK_LOCK:
        bank = _BANK_CACHE.get(key)
        if bank is None:
            bank = np.random.default_rng(key[0]).standard_normal(key[1]).astype(np.float64)
            bank.setflags(write=False)
            _BANK_CACHE[key] = bank
        return bank


def mc_bank(hp=None) -> np.ndarray:
    """
    The Monte Carlo bank for ``allocation.mc_seed`` / ``allocation.mc_samples``:
    the running app's sampler bank (``sampler.get_mc_bank``) when there is one,
    else the identical array from :func:`default_bank`. Never consumes the
    sample-buffer streams.
    """
    if hp is None:
        from app.hyperparams import get_hyperparams

        hp = get_hyperparams()
    try:
        from flask import current_app, has_app_context

        if has_app_context():
            sampler = getattr(current_app, "sampler", None)
            if sampler is not None and hasattr(sampler, "get_mc_bank"):
                return sampler.get_mc_bank(hp.mc_seed, hp.mc_samples)
    except RuntimeError:  # pragma: no cover - no app
        pass
    return default_bank(hp.mc_seed, hp.mc_samples)


def allocation_probability(
    mu: float,
    v: float,
    *,
    bank: np.ndarray,
    l_min: float,
    l_max: float,
    c: float,
    b: float,
    kappa: float,
) -> float:
    """eq. (alloc-mc): ``mean_j rho(mu + sqrt(max(v, 0)) Z_j)``."""
    sd = np.sqrt(max(float(v), 0.0))
    return float(np.mean(rho(float(mu) + sd * bank, l_min, l_max, c, b, kappa)))


def agent_probability(decision_type: str, mu: float, v: float, *, hp=None, bank: np.ndarray | None = None) -> float:
    """``pi(mu, v)`` for an agent, with its slope ``b_rho * r_max`` and the configured shape."""
    if hp is None:
        from app.hyperparams import get_hyperparams

        hp = get_hyperparams()
    if bank is None:
        bank = mc_bank(hp)
    return allocation_probability(
        mu,
        v,
        bank=bank,
        l_min=hp.l_min,
        l_max=hp.l_max,
        c=hp.c_rho,
        b=hp.agent(decision_type).b_rho,
        kappa=hp.kappa_rho,
    )


def contrast_moments(f: np.ndarray, alpha: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    """eq. (contrast-post): ``(mu, v) = (f' alpha, f' Sigma f)``."""
    f = np.asarray(f, dtype=float)
    return float(f @ np.asarray(alpha, dtype=float)), float(f @ np.asarray(cov, dtype=float) @ f)


__all__ = ["rho", "default_bank", "mc_bank", "allocation_probability", "agent_probability", "contrast_moments"]
