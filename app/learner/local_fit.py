"""
Inf-LSVI local fit (implementation contract D8; main_new.tex §3.3 eqs. (Xy),
(boundary-game-average), (ridge), (posterior), (fitblocks); Algorithm 1;
Appendix A Algorithm 4 and its contraction paragraph).

For one dyad and one agent, with rows ``k = 1..n`` (state ``S_k``, action
``A_k``, recorded ``pi_k = Pr(A_k = 1)``, normalized reward ``R_k``):

    X_k = [h(S_k)', (A_k - pi_k) f(S_k)'],
    Y_k = R_k + gamma V(theta_check; S_{k+1}, pi_{k+1})          (k < n),
    Y_n = R_n + gamma sum_c w_c V(theta_check; S^c, pi^c)         (successor candidates),
    V(theta; s, pi) = h(s)'beta + [f(s)'alpha]_+ - pi f(s)'alpha  (eq. (value)),

    theta_hat = (X'X + Lambda + eps I)^{-1} X'Y,
    Sigma_hat = sigma^2 (X'X + Lambda + eps I)^{-1},

with ``J`` Bellman backups warm-started at ``theta_check`` (the dyad's
previous fit, else 0). The last row's successor is given as weighted
candidates: two (game off / on, weights ``(1 - q^REL, q^REL)``) at a week
boundary, one otherwise, none when no successor upload exists (target
``R_n`` alone). With ``gamma = 0`` (REL) the targets are the rewards.

The L3 diagnostic (Monitoring-Spec §5.L) is the spectral norm of
``G X' X_+`` with ``G = (X'X + Lambda + eps I)^{-1}`` and ``X_+`` the gradient
of each target's successor value with respect to theta at ``theta_check``:
row ``k`` is ``[h(S_{k+1})', (1{f(S_{k+1})'alpha_check > 0} - pi_{k+1}) f(S_{k+1})']``
(the candidate-weighted average for the last row, zero without a successor).

Pure numpy; every inverse carries ``+ eps I`` and the same operation order is
used on every call, so reruns are bitwise identical.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

# Early-stop tolerance between Bellman backups when J > 1 (Algorithm 4's
# epsilon_fix; the testbed's value). Irrelevant at the deployed J = 1.
BACKUP_TOL = 1e-6


@dataclasses.dataclass(frozen=True)
class SuccessorCandidate:
    """One candidate successor state of the last row: weight, h, f and the probability pi at it."""

    weight: float
    h: np.ndarray
    f: np.ndarray
    pi: float


@dataclasses.dataclass
class LocalFitResult:
    """Output of :func:`fit_local` (eq. (posterior) partitioned by eq. (fitblocks))."""

    theta: np.ndarray
    cov: np.ndarray
    p_h: int
    p_f: int
    n: int
    X: np.ndarray
    Y: np.ndarray
    spectral_norm: float | None
    theta_change_inf: float
    no_successor: bool
    backups: int

    @property
    def alpha(self) -> np.ndarray:
        return self.theta[self.p_h:]

    @property
    def cov_aa(self) -> np.ndarray:
        return self.cov[self.p_h:, self.p_h:]


def value(h: np.ndarray, f: np.ndarray, pi, beta: np.ndarray, alpha: np.ndarray):
    """eq. (value): ``h'beta + [f'alpha]_+ - pi f'alpha`` (row-wise for matrices)."""
    fa = f @ alpha
    return h @ beta + np.maximum(fa, 0.0) - pi * fa


def design_matrix(H: np.ndarray, Fm: np.ndarray, A: np.ndarray, PI: np.ndarray) -> np.ndarray:
    """eq. (Xy): rows ``[h(S_k)', (A_k - pi_k) f(S_k)']``."""
    return np.hstack([H, (A - PI)[:, None] * Fm])


def penalty_matrix(p_h: int, p_f: int, lambda_main: float, lambda_mod: float) -> np.ndarray:
    """eq. (localprior): ``diag(lambda_main I_{p_h}, lambda_main, lambda_mod I_{p_f - 1})``."""
    return np.diag([lambda_main] * p_h + [lambda_main] + [lambda_mod] * (p_f - 1))


def targets(
    H: np.ndarray,
    Fm: np.ndarray,
    PI: np.ndarray,
    R: np.ndarray,
    successors: Sequence[SuccessorCandidate],
    theta: np.ndarray,
    gamma: float,
    p_h: int,
) -> np.ndarray:
    """eqs. (Xy) and (boundary-game-average) at ``theta``."""
    y = np.asarray(R, dtype=float).copy()
    if gamma == 0.0:
        return y
    beta, alpha = theta[:p_h], theta[p_h:]
    if len(y) > 1:
        y[:-1] += gamma * value(H[1:], Fm[1:], PI[1:], beta, alpha)
    if successors:
        tail = 0.0
        for c in successors:
            tail += float(c.weight) * float(value(c.h[None, :], c.f[None, :], np.array([c.pi]), beta, alpha)[0])
        y[-1] += gamma * tail
    return y


def successor_gradients(
    H: np.ndarray,
    Fm: np.ndarray,
    PI: np.ndarray,
    successors: Sequence[SuccessorCandidate],
    theta: np.ndarray,
    p_h: int,
) -> np.ndarray:
    """``X_+`` of the L3 diagnostic: d V(S_{k+1}) / d theta at ``theta`` per row."""
    n = H.shape[0]
    p_f = Fm.shape[1]
    alpha = theta[p_h:]
    Xp = np.zeros((n, p_h + p_f))
    if n > 1:
        ind = (Fm[1:] @ alpha > 0.0).astype(float)
        Xp[:-1, :p_h] = H[1:]
        Xp[:-1, p_h:] = (ind - PI[1:])[:, None] * Fm[1:]
    for c in successors:
        ind = 1.0 if float(c.f @ alpha) > 0.0 else 0.0
        Xp[-1, :p_h] += float(c.weight) * c.h
        Xp[-1, p_h:] += float(c.weight) * (ind - float(c.pi)) * c.f
    return Xp


def fit_local(
    H: np.ndarray,
    Fm: np.ndarray,
    A: np.ndarray,
    PI: np.ndarray,
    R: np.ndarray,
    successors: Sequence[SuccessorCandidate],
    *,
    theta_check: np.ndarray | None,
    gamma: float,
    lambda_main: float,
    lambda_mod: float,
    sigma_sq: float,
    eps: float,
    j_backups: int = 1,
) -> LocalFitResult:
    """
    Algorithm 1 / Algorithm 4 for one dyad: ``J`` backups from ``theta_check``
    (zeros when ``None``). ``successors`` describe the last row's successor
    (empty: no successor, target ``R_n``).
    """
    H = np.asarray(H, dtype=float)
    Fm = np.asarray(Fm, dtype=float)
    A = np.asarray(A, dtype=float)
    PI = np.asarray(PI, dtype=float)
    R = np.asarray(R, dtype=float)
    n, p_h = H.shape
    p_f = Fm.shape[1]
    if n == 0:
        raise ValueError("a local fit needs at least one row.")
    if not (Fm.shape[0] == len(A) == len(PI) == len(R) == n):
        raise ValueError("row counts of H, F, A, pi and R differ.")
    d = p_h + p_f
    theta0 = np.zeros(d) if theta_check is None else np.asarray(theta_check, dtype=float).copy()
    if theta0.shape != (d,):
        raise ValueError(f"warm start has shape {theta0.shape}, expected ({d},).")

    X = design_matrix(H, Fm, A, PI)
    Amat = X.T @ X + penalty_matrix(p_h, p_f, lambda_main, lambda_mod) + eps * np.eye(d)
    Ainv = np.linalg.inv(Amat)

    theta = theta0
    y = targets(H, Fm, PI, R, successors, theta, gamma, p_h)
    backups = 0
    for j in range(max(1, int(j_backups))):
        if j > 0:
            y = targets(H, Fm, PI, R, successors, theta, gamma, p_h)
        new = Ainv @ (X.T @ y)
        backups += 1
        done = float(np.max(np.abs(new - theta))) <= BACKUP_TOL
        theta = new
        if done or gamma == 0.0:
            break

    cov = sigma_sq * Ainv
    cov = 0.5 * (cov + cov.T)

    spectral = None
    if gamma > 0.0:
        Xp = successor_gradients(H, Fm, PI, successors, theta0, p_h)
        spectral = float(np.linalg.norm(Ainv @ (X.T @ Xp), 2))

    return LocalFitResult(
        theta=theta,
        cov=cov,
        p_h=p_h,
        p_f=p_f,
        n=n,
        X=X,
        Y=y,
        spectral_norm=spectral,
        theta_change_inf=float(np.max(np.abs(theta - theta0))),
        # Only meaningful when the target looks ahead (REL, gamma = 0, never does).
        no_successor=gamma > 0.0 and len(successors) == 0,
        backups=backups,
    )


__all__ = [
    "BACKUP_TOL",
    "SuccessorCandidate",
    "LocalFitResult",
    "value",
    "design_matrix",
    "penalty_matrix",
    "targets",
    "successor_gradients",
    "fit_local",
]
