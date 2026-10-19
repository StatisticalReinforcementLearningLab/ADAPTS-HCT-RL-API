"""
Empirical-Bayes pooling of the advantage block (implementation contract D9;
main_new.tex §3.5 eqs. (hier), (diagparam), (ml), (hyperprior), (map),
(maptarget), (floor), (shrink), (shrinkmatrix); Algorithm 2; Monitoring-Spec
§7 F-U5).

Inputs per dyad ``i`` in ``E_w^+`` (``N_w`` dyads): the local advantage fit
``alpha_i`` and its covariance ``V_i = Sigma_hat^{aa}_i``. With
``Sigma0 = diag(tau_act, tau_mod I_{p_f - 1})``, ``M_i = V_i + eps I + Sigma0``,
``P_i = M_i^{-1}`` and ``z_i = 1``, the objective minimized is the negative of
eq. (map):

    f(Theta, tau) = 1/2 sum_i [log det M_i + r_i' P_i r_i] + lambda0/2 |Theta|^2
                    + sum_b |b| [(nu0/2 + 1) log tau_b + nu0 tau0_b / (2 tau_b)],
    r_i = alpha_i - Theta.

``Theta`` is profiled out exactly, ``Theta(tau) = (sum P_i + lambda0 I)^{-1} sum P_i alpha_i``,
so (envelope theorem) the gradient in ``tau`` is

    df/dtau_b = 1/2 sum_i sum_{j in b} ([P_i]_jj - [P_i r_i]_j^2)
                + |b| [(nu0/2 + 1)/tau_b - nu0 tau0_b / (2 tau_b^2)].

The profiled objective is minimized over ``tau`` directly with scipy's
L-BFGS-B, bounds ``[floor_b, inf)`` with ``floor_b = nu0 tau0_b / (N_w + nu0 + 2)``
(eq. (floor)), ``maxiter = k_max``, ``gtol = eps_tol``, ``ftol = 0`` (so it
stops on the gradient or the cap) and the analytic gradient, warm-started at
last week's ``tau`` clipped to the floor (``tau0`` at the first update).

Convergence (Algorithm 2 line 6): every value finite and the projected
sup-norm gradient with respect to ``(Theta, tau)`` at most ``eps_tol``. The
``Theta`` part is ``sum P_i r_i - lambda0 Theta`` (zero up to rounding at the
profiled optimum). L-BFGS-B's line search works on function values, which near
the optimum change by less than their rounding error; when it stops there (an
abnormal line-search exit) with the gradient still above ``eps_tol``,
projected Newton steps on the analytic gradient (no function values involved)
finish the job inside the same ``k_max`` iteration budget. Otherwise the
fallback F-U5 applies: last week's ``(Theta, Sigma0)``, or the prior
``(0, diag(tau0))`` at the first update. Posteriors are always recomputed from
this week's local fits (eq. (shrink)), with ``eps I`` added before every
inverse:

    S0inv = (Sigma0 + eps I)^{-1},  Vinv = (V_i + eps I)^{-1},
    Sigma_EB = (S0inv + Vinv)^{-1},  alpha_EB = Sigma_EB (S0inv Theta + Vinv alpha_i).

Pure numpy / scipy; deterministic given its inputs.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize

TERMINATION_NEWTON = "projected Newton polish on the analytic gradient converged"


@dataclasses.dataclass
class EBResult:
    """Hyper-update of one agent (Monitoring-Spec §9.5 ``eb_hyper_updates``)."""

    theta: np.ndarray            # selected Theta (solver optimum, or fallback)
    tau: np.ndarray              # selected (tau_act, tau_mod)
    floor: np.ndarray            # (floor_act, floor_mod)
    n_w: int
    converged: bool
    iterations: int
    grad_inf: float              # projected sup-norm gradient at the solver's point
    objective: float             # f at the solver's point (negative penalized log likelihood)
    fallback_used: str | None    # None | "previous" | "prior"
    termination: str
    solver_theta: np.ndarray     # the solver's (Theta, tau), kept even when rejected
    solver_tau: np.ndarray
    warm_start_tau: np.ndarray
    tau0: np.ndarray
    newton_steps: int = 0

    @property
    def sigma0(self) -> np.ndarray:
        """``Sigma0 = diag(tau_act, tau_mod I_{p_f - 1})`` (eq. (diagparam))."""
        p_f = len(self.theta)
        return np.diag(block_values(self.tau, p_f))

    def at_floor(self, rel: float = 0.01) -> list[bool]:
        """L1 (Monitoring-Spec §5.L): ``tau_b`` within ``rel`` of its floor, per block."""
        return [bool(self.tau[b] <= (1.0 + rel) * self.floor[b]) for b in range(2)]


def block_index(p_f: int) -> np.ndarray:
    """Block of each advantage coordinate: 0 = act (the constant of f), 1 = mod."""
    return np.concatenate([[0], np.ones(max(p_f - 1, 0), dtype=int)]).astype(int)


def block_values(tau: Sequence[float], p_f: int) -> np.ndarray:
    """The diagonal ``(tau_act, tau_mod, ..., tau_mod)`` of length ``p_f``."""
    return np.asarray(tau, dtype=float)[block_index(p_f)]


def floors(tau0: Sequence[float], nu0: float, n_w: int) -> np.ndarray:
    """eq. (floor): ``nu0 tau0_b / (N_w + nu0 + 2)``."""
    return nu0 * np.asarray(tau0, dtype=float) / (float(n_w) + nu0 + 2.0)


class _Problem:
    """The profiled objective of one hyper-update and its analytic gradient."""

    def __init__(self, alphas, covs, tau0, nu0, lambda0, eps):
        self.alphas = [np.asarray(a, dtype=float) for a in alphas]
        p_f = len(self.alphas[0])
        self.p_f = p_f
        self.eye = np.eye(p_f)
        # V_i + eps I enters every inverse (contract D9).
        self.covs = [np.asarray(V, dtype=float) + eps * self.eye for V in covs]
        self.tau0 = np.asarray(tau0, dtype=float)
        self.nu0 = float(nu0)
        self.lambda0 = float(lambda0)
        self.blk = block_index(p_f)
        self.blk_size = (int((self.blk == 0).sum()), int((self.blk == 1).sum()))

    def evaluate(self, tau: np.ndarray):
        """``(f, df/dtau, Theta, dl/dTheta)`` at ``tau`` (Theta profiled); ``f = inf`` if a matrix is singular."""
        tau = np.asarray(tau, dtype=float)
        sigma0 = np.diag(tau[self.blk])
        Ps = []
        logdets = 0.0
        Psum = np.zeros((self.p_f, self.p_f))
        q = np.zeros(self.p_f)
        for a_i, V_i in zip(self.alphas, self.covs):
            M = V_i + sigma0
            sign, ld = np.linalg.slogdet(M)
            if sign <= 0 or not math.isfinite(ld):
                return math.inf, np.full(2, np.nan), np.full(self.p_f, np.nan), np.full(self.p_f, np.nan)
            P = np.linalg.inv(M)
            Ps.append(P)
            logdets += ld
            Psum += P
            q += P @ a_i
        theta = np.linalg.solve(Psum + self.lambda0 * self.eye, q)
        quad = 0.0
        grad = np.zeros(2)
        grad_theta = -self.lambda0 * theta
        for P, a_i in zip(Ps, self.alphas):
            r = a_i - theta
            Pr = P @ r
            quad += float(r @ Pr)
            grad_theta = grad_theta + Pr
            dg = np.diag(P) - Pr**2
            for b in (0, 1):
                if self.blk_size[b]:
                    grad[b] += 0.5 * float(dg[self.blk == b].sum())
        value = 0.5 * (logdets + quad) + 0.5 * self.lambda0 * float(theta @ theta)
        for b, size in enumerate(self.blk_size):
            if size == 0:
                continue
            value += size * ((self.nu0 / 2.0 + 1.0) * math.log(tau[b]) + self.nu0 * self.tau0[b] / (2.0 * tau[b]))
            grad[b] += size * ((self.nu0 / 2.0 + 1.0) / tau[b] - self.nu0 * self.tau0[b] / (2.0 * tau[b] ** 2))
        return float(value), grad, theta, grad_theta

    def fun(self, tau):
        value, grad, _, _ = self.evaluate(tau)
        if not math.isfinite(value) or not np.all(np.isfinite(grad)):
            # Keep L-BFGS-B away from an invalid point; the post-check rejects it.
            return 1e300, np.zeros(2)
        return value, grad


def projected_grad_inf(tau: np.ndarray, grad_tau: np.ndarray, grad_theta: np.ndarray, floor: np.ndarray) -> float:
    """
    Projected sup-norm gradient over ``(Theta, tau)`` for the box ``tau >= floor``:
    a ``tau_b`` at its floor whose gradient (of the minimized objective) points
    below the floor contributes 0.
    """
    proj = np.array(grad_tau, dtype=float)
    for b in range(2):
        if tau[b] <= floor[b] and proj[b] > 0.0:
            proj[b] = 0.0
    parts = [np.abs(proj)]
    if grad_theta.size:
        parts.append(np.abs(grad_theta))
    return float(np.max(np.concatenate(parts)))


def _newton_polish(problem: _Problem, tau: np.ndarray, floor: np.ndarray, eps_tol: float, budget: int):
    """
    Projected Newton iterations on the analytic gradient (finite-difference
    Jacobian of the gradient, no function values) from ``tau``. Returns
    ``(tau, steps)``; stops when the projected gradient is at most ``eps_tol``
    or after ``budget`` steps.
    """
    tau = np.maximum(np.asarray(tau, dtype=float), floor)
    steps = 0
    while steps < budget:
        _, g, theta, g_theta = problem.evaluate(tau)
        if not np.all(np.isfinite(g)):
            break
        if projected_grad_inf(tau, g, g_theta, floor) <= eps_tol:
            break
        free = [b for b in range(2) if not (tau[b] <= floor[b] and g[b] > 0.0)]
        if not free:
            break
        J = np.zeros((2, 2))
        for b in range(2):
            h = 1e-6 * tau[b]
            tp = tau.copy()
            tp[b] += h
            tm = tau.copy()
            tm[b] = max(tau[b] - h, 0.5 * tau[b])
            _, gp, _, _ = problem.evaluate(tp)
            _, gm, _, _ = problem.evaluate(tm)
            J[:, b] = (gp - gm) / (tp[b] - tm[b])
        Jf = J[np.ix_(free, free)]
        try:
            step = np.linalg.solve(Jf, g[free])
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        new = tau.copy()
        new[free] = tau[free] - step
        tau = np.maximum(new, floor)
        steps += 1
    return tau, steps


def hyper_update(
    alphas: Sequence[np.ndarray],
    covs: Sequence[np.ndarray],
    *,
    tau0: Sequence[float],
    nu0: float,
    lambda0: float,
    eps: float,
    k_max: int,
    eps_tol: float,
    warm_tau: Sequence[float] | None = None,
    previous: tuple[np.ndarray, np.ndarray] | None = None,
    n_w: int | None = None,
) -> EBResult:
    """
    Algorithm 2 lines 4-9 for one agent. ``alphas``/``covs`` are the local
    advantage fits of ``E_w^+`` (``N_w = len(alphas)`` unless ``n_w`` is given);
    ``warm_tau`` last week's ``tau`` (``None`` at the first update);
    ``previous`` last week's ``(Theta, tau)`` for the F-U5 fallback (``None``:
    fall back to the prior ``(0, tau0)``).
    """
    if not alphas:
        raise ValueError("the hyper-update needs at least one local fit.")
    tau0 = np.asarray(tau0, dtype=float)
    n_w = len(alphas) if n_w is None else int(n_w)
    floor = floors(tau0, nu0, n_w)
    problem = _Problem(alphas, covs, tau0, nu0, lambda0, eps)
    p_f = problem.p_f
    start = tau0.copy() if warm_tau is None else np.asarray(warm_tau, dtype=float).copy()
    x0 = np.maximum(start, floor)

    solver_tau = np.full(2, np.nan)
    solver_theta = np.full(p_f, np.nan)
    objective = math.nan
    grad_inf = math.inf
    iterations = 0
    newton_steps = 0
    termination = ""
    converged = False
    try:
        res = minimize(
            problem.fun,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=[(float(floor[0]), None), (float(floor[1]), None)],
            options={"maxiter": int(k_max), "gtol": float(eps_tol), "ftol": 0.0},
        )
        iterations = int(res.nit)
        termination = str(res.message)
        solver_tau = np.asarray(res.x, dtype=float)
        if np.all(np.isfinite(solver_tau)):
            solver_tau = np.maximum(solver_tau, floor)
            objective, g, solver_theta, g_theta = problem.evaluate(solver_tau)
            grad_inf = (
                projected_grad_inf(solver_tau, g, g_theta, floor)
                if np.all(np.isfinite(g)) and np.all(np.isfinite(g_theta))
                else math.inf
            )
            hit_cap = iterations >= int(k_max)
            if math.isfinite(objective) and grad_inf > eps_tol and not hit_cap and math.isfinite(grad_inf):
                polished, newton_steps = _newton_polish(
                    problem, solver_tau, floor, eps_tol, budget=int(k_max) - iterations
                )
                if newton_steps:
                    obj2, g2, th2, gt2 = problem.evaluate(polished)
                    if math.isfinite(obj2) and np.all(np.isfinite(g2)) and np.all(np.isfinite(gt2)):
                        gi2 = projected_grad_inf(polished, g2, gt2, floor)
                        iterations += newton_steps
                        solver_tau, objective, solver_theta, grad_inf = polished, obj2, th2, gi2
                        if gi2 <= eps_tol:
                            termination = f"{termination}; {TERMINATION_NEWTON}"
            converged = bool(
                math.isfinite(objective)
                and np.all(np.isfinite(solver_theta))
                and np.all(np.isfinite(solver_tau))
                and grad_inf <= eps_tol
            )
    except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
        termination = f"error: {type(exc).__name__}: {exc}"
        converged = False

    if converged:
        theta, tau, fallback = solver_theta.copy(), solver_tau.copy(), None
    elif previous is not None:
        theta = np.asarray(previous[0], dtype=float).copy()
        tau = np.asarray(previous[1], dtype=float).copy()
        fallback = "previous"
    else:
        theta, tau, fallback = np.zeros(p_f), tau0.copy(), "prior"

    return EBResult(
        theta=theta,
        tau=tau,
        floor=floor,
        n_w=n_w,
        converged=converged,
        iterations=iterations,
        grad_inf=float(grad_inf),
        objective=float(objective),
        fallback_used=fallback,
        termination=termination,
        solver_theta=solver_theta,
        solver_tau=solver_tau,
        warm_start_tau=x0,
        tau0=tau0,
        newton_steps=newton_steps,
    )


def posterior(
    alpha: np.ndarray,
    cov_aa: np.ndarray,
    theta: np.ndarray,
    tau: Sequence[float],
    eps: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    eq. (shrink) for one dyad, ``eps I`` on every inverse. Returns
    ``(alpha_EB, Sigma_EB, own_data_weight)`` with the L2 weight
    ``tr(Sigma_EB (V + eps I)^{-1}) / p_f`` (eq. (shrinkmatrix)).
    """
    alpha = np.asarray(alpha, dtype=float)
    p_f = len(alpha)
    eye = np.eye(p_f)
    sigma0 = np.diag(block_values(tau, p_f))
    s0inv = np.linalg.inv(sigma0 + eps * eye)
    vinv = np.linalg.inv(np.asarray(cov_aa, dtype=float) + eps * eye)
    cov = np.linalg.inv(s0inv + vinv)
    mean = cov @ (s0inv @ np.asarray(theta, dtype=float) + vinv @ alpha)
    cov = 0.5 * (cov + cov.T)
    weight = float(np.trace(cov @ vinv) / p_f)
    return mean, cov, weight


__all__ = [
    "EBResult",
    "block_index",
    "block_values",
    "floors",
    "projected_grad_inf",
    "hyper_update",
    "posterior",
]
