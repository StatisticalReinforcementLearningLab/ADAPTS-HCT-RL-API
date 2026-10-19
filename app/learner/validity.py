"""
C3 validity gate of a new policy (Monitoring-Spec §5.C C3; §2.1 "Before an
update is published"; implementation contract §3.1, D14).

For one agent, before its new policy is published, all of the following must
hold, else the agent keeps last week's policy (F-U2):

1. Every fitted number is finite: theta_hat_i, Sigma_hat_i, Theta_hat, tau_hat^2,
   alpha_EB_i, Sigma_EB_i.
2. Sigma_hat_i and Sigma_EB_i are valid variance matrices: symmetric, with no
   eigenvalue below ``-1e-10 x`` their trace.
3. Each between-dyad variance is at or above its floor:
   ``tau_b >= (1 - 1e-6) floor_b``.
4. Pooling did not increase any variance: the smallest eigenvalue of
   ``Sigma_hat^aa_i - Sigma_EB_i`` and of ``Sigma0 - Sigma_EB_i`` is at least
   ``-(eps + 1e-10 x trace)``, with ``eps`` the numerical ridge
   (``numerics.eps_ridge``) and the trace that of the minuend
   (``Sigma_hat^aa_i`` resp. ``Sigma0``).
5. The number of parameters matches the design: AYA 7 + 4, CP 6 + 4, REL 7 + 2.

The C3 alarm itself is raised by monitoring from the returned result
(``on_update_finished``); this module only decides.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from app.hyperparams import P_F, P_H, agent_key

CHECK_FINITE = "1_finite"
CHECK_VARIANCE = "2_valid_variance"
CHECK_FLOOR = "3_floor"
CHECK_POOLING = "4_pooling_no_increase"
CHECK_DIMS = "5_dimensions"
CHECKS = (CHECK_FINITE, CHECK_VARIANCE, CHECK_FLOOR, CHECK_POOLING, CHECK_DIMS)

EIG_REL_TOL = 1e-10
FLOOR_REL_TOL = 1e-6
# Symmetry is checked relative to the largest entry (the learner symmetrizes
# every stored covariance, so this is exact in practice).
SYM_REL_TOL = 1e-12


def _finite(x) -> bool:
    arr = np.asarray(x, dtype=float)
    return bool(arr.size == 0 or np.all(np.isfinite(arr)))


def _valid_variance(M: np.ndarray) -> tuple[bool, str | None]:
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return False, f"not square (shape {M.shape})"
    if not _finite(M):
        return False, "not finite"
    scale = max(1.0, float(np.max(np.abs(M)))) if M.size else 1.0
    if float(np.max(np.abs(M - M.T))) > SYM_REL_TOL * scale:
        return False, "not symmetric"
    min_eig = float(np.min(np.linalg.eigvalsh(0.5 * (M + M.T))))
    tr = float(np.trace(M))
    if min_eig < -EIG_REL_TOL * abs(tr):
        return False, f"eigenvalue {min_eig:.3e} below -1e-10 x trace ({tr:.3e})"
    return True, None


def _min_eig_ok(minuend: np.ndarray, subtrahend: np.ndarray, eps: float) -> tuple[bool, float]:
    D = np.asarray(minuend, dtype=float) - np.asarray(subtrahend, dtype=float)
    min_eig = float(np.min(np.linalg.eigvalsh(0.5 * (D + D.T))))
    bound = -(eps + EIG_REL_TOL * abs(float(np.trace(minuend))))
    return min_eig >= bound, min_eig


def check_policy(
    decision_type: str,
    local_fits: Mapping[str, tuple[np.ndarray, np.ndarray]],
    theta: np.ndarray,
    tau: np.ndarray,
    floor: np.ndarray,
    sigma0: np.ndarray,
    posteriors: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    eps: float,
) -> dict:
    """
    Run C3 checks 1-5 for one agent. ``local_fits`` maps dyad -> ``(theta_hat,
    Sigma_hat)``; ``posteriors`` maps dyad -> ``(alpha_EB, Sigma_EB)``.
    Returns ``{"passed": bool, "checks": {name: bool}, "failures": [str]}``.
    """
    agent = agent_key(decision_type)
    p_h, p_f = P_H[agent], P_F[agent]
    d = p_h + p_f
    failures: list[str] = []
    ok = {name: True for name in CHECKS}

    def fail(check: str, message: str) -> None:
        ok[check] = False
        if len(failures) < 50:
            failures.append(f"{check}: {message}")

    # 5. dimensions (checked first: the other checks index by p_h / p_f).
    dims_ok = True
    if np.asarray(theta).shape != (p_f,):
        fail(CHECK_DIMS, f"Theta has shape {np.asarray(theta).shape}, expected ({p_f},)")
        dims_ok = False
    if np.asarray(sigma0).shape != (p_f, p_f):
        fail(CHECK_DIMS, f"Sigma0 has shape {np.asarray(sigma0).shape}, expected ({p_f}, {p_f})")
        dims_ok = False
    for gid, (th, cov) in local_fits.items():
        if np.asarray(th).shape != (d,) or np.asarray(cov).shape != (d, d):
            fail(CHECK_DIMS, f"local fit of {gid} has {np.asarray(th).shape[0] if np.ndim(th) else '?'} coefficients, expected {p_h} + {p_f}")
            dims_ok = False
    for gid, (a, cov) in posteriors.items():
        if np.asarray(a).shape != (p_f,) or np.asarray(cov).shape != (p_f, p_f):
            fail(CHECK_DIMS, f"posterior of {gid} has shape {np.asarray(a).shape}, expected ({p_f},)")
            dims_ok = False

    # 1. finite
    for name, value in (("Theta", theta), ("tau", tau)):
        if not _finite(value):
            fail(CHECK_FINITE, f"{name} is not finite")
    for gid, (th, cov) in local_fits.items():
        if not _finite(th) or not _finite(cov):
            fail(CHECK_FINITE, f"local fit of {gid} is not finite")
    for gid, (a, cov) in posteriors.items():
        if not _finite(a) or not _finite(cov):
            fail(CHECK_FINITE, f"posterior of {gid} is not finite")

    # 2. valid variance matrices
    for gid, (_, cov) in local_fits.items():
        good, why = _valid_variance(cov)
        if not good:
            fail(CHECK_VARIANCE, f"Sigma_hat of {gid} {why}")
    for gid, (_, cov) in posteriors.items():
        good, why = _valid_variance(cov)
        if not good:
            fail(CHECK_VARIANCE, f"Sigma_EB of {gid} {why}")

    # 3. floor
    tau_arr = np.asarray(tau, dtype=float)
    floor_arr = np.asarray(floor, dtype=float)
    for b, name in enumerate(("act", "mod")):
        if not (np.isfinite(tau_arr[b]) and tau_arr[b] >= (1.0 - FLOOR_REL_TOL) * floor_arr[b]):
            fail(CHECK_FLOOR, f"tau_{name} = {tau_arr[b]!r} below floor {floor_arr[b]!r}")

    # 4. pooling does not increase variance
    if dims_ok and _finite(sigma0):
        for gid, (a_eb, cov_eb) in posteriors.items():
            if gid not in local_fits or not _finite(cov_eb):
                continue
            _, cov = local_fits[gid]
            cov_aa = np.asarray(cov, dtype=float)[p_h:, p_h:]
            if not _finite(cov_aa):
                continue
            good, m = _min_eig_ok(cov_aa, cov_eb, eps)
            if not good:
                fail(CHECK_POOLING, f"Sigma_hat^aa - Sigma_EB of {gid} has eigenvalue {m:.3e}")
            good, m = _min_eig_ok(sigma0, cov_eb, eps)
            if not good:
                fail(CHECK_POOLING, f"Sigma0 - Sigma_EB of {gid} has eigenvalue {m:.3e}")

    return {"passed": all(ok.values()), "checks": ok, "failures": failures}


__all__ = ["CHECKS", "check_policy"]
