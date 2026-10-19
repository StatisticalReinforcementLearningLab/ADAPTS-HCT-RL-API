"""
WS-L tests: the learner mathematics (implementation contract D8-D10;
main_new.tex eqs. (Xy), (boundary-game-average), (posterior), (ml),
(hyperprior), (floor), (shrink), (rho), (alloc-mc); Algorithms 1, 2, 4;
Monitoring-Spec C3, F-U5).

The empirical-Bayes and allocation results are compared with the reference
implementation of the Simulation_Testbed (``algos/deploy.py``); those tests
skip when the testbed directory is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.hyperparams import default_hyperparams
from app.learner import allocation as AL
from app.learner import eb as EB
from app.learner import validity as V
from app.learner.local_fit import (
    SuccessorCandidate,
    design_matrix,
    fit_local,
    penalty_matrix,
    value,
)
from tests.test_learner_support import learner_app, load_testbed  # noqa: F401 - fixture

HP = default_hyperparams()
EPS = HP.eps_ridge


def _dataset(rng, n, p_h, p_f):
    H = rng.uniform(-1, 1, (n, p_h))
    H[:, 0] = 1.0
    Fm = rng.uniform(-1, 1, (n, p_f))
    Fm[:, 0] = 1.0
    A = rng.integers(0, 2, n).astype(float)
    PI = rng.uniform(0.2, 0.8, n)
    R = rng.uniform(0, 1, n)
    return H, Fm, A, PI, R


def _v(h, f, pi, theta, p_h):
    beta, alpha = theta[:p_h], theta[p_h:]
    fa = float(f @ alpha)
    return float(h @ beta) + max(fa, 0.0) - pi * fa


# --------------------------------------------------------------- local fit


@pytest.mark.parametrize("agent", ["aya", "cp"])
def test_local_fit_matches_direct_evaluation(agent):
    a = HP.agent(agent)
    rng = np.random.default_rng(3)
    n = 40
    H, Fm, A, PI, R = _dataset(rng, n, a.p_h, a.p_f)
    theta_check = rng.normal(0, 0.3, a.p_h + a.p_f)
    s_h = rng.uniform(-1, 1, a.p_h)
    s_f = rng.uniform(-1, 1, a.p_f)
    succ = [SuccessorCandidate(1.0, s_h, s_f, 0.4)]
    res = fit_local(
        H, Fm, A, PI, R, succ, theta_check=theta_check, gamma=a.gamma,
        lambda_main=a.lambda_main, lambda_mod=a.lambda_mod, sigma_sq=a.sigma_sq, eps=EPS,
    )
    # Direct evaluation of eqs. (Xy) and (posterior).
    X = np.array([np.r_[H[k], (A[k] - PI[k]) * Fm[k]] for k in range(n)])
    Y = np.empty(n)
    for k in range(n - 1):
        Y[k] = R[k] + a.gamma * _v(H[k + 1], Fm[k + 1], PI[k + 1], theta_check, a.p_h)
    Y[-1] = R[-1] + a.gamma * _v(s_h, s_f, 0.4, theta_check, a.p_h)
    Lam = np.diag([a.lambda_main] * a.p_h + [a.lambda_main] + [a.lambda_mod] * (a.p_f - 1))
    M = X.T @ X + Lam + EPS * np.eye(a.p_h + a.p_f)
    np.testing.assert_allclose(res.X, X, rtol=0, atol=0)
    np.testing.assert_allclose(res.Y, Y, rtol=0, atol=1e-14)
    np.testing.assert_allclose(res.theta, np.linalg.solve(M, X.T @ Y), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(res.cov, a.sigma_sq * np.linalg.inv(M), rtol=1e-10, atol=1e-14)
    assert np.array_equal(res.cov, res.cov.T)
    assert res.alpha.shape == (a.p_f,) and res.cov_aa.shape == (a.p_f, a.p_f)
    np.testing.assert_array_equal(res.alpha, res.theta[a.p_h:])
    assert not res.no_successor


def test_penalty_matrix_blocks():
    a = HP.agent("aya")
    lam = np.diag(penalty_matrix(a.p_h, a.p_f, a.lambda_main, a.lambda_mod))
    assert list(lam) == [16.0] * 7 + [16.0] + [6.0] * 3
    assert list(lam) == a.local_penalty_diag()
    r = HP.agent("rel")
    assert list(np.diag(penalty_matrix(r.p_h, r.p_f, r.lambda_main, r.lambda_mod))) == [16.0] * 8 + [2.0]


@pytest.mark.parametrize("q", [0.0, 0.3, 1.0])
def test_boundary_game_average(q):
    a = HP.agent("aya")
    rng = np.random.default_rng(11)
    H, Fm, A, PI, R = _dataset(rng, 5, a.p_h, a.p_f)
    theta_check = rng.normal(0, 0.5, a.p_h + a.p_f)
    h0, h1 = rng.uniform(-1, 1, a.p_h), rng.uniform(-1, 1, a.p_h)
    f0, f1 = rng.uniform(-1, 1, a.p_f), rng.uniform(-1, 1, a.p_f)
    succ = [SuccessorCandidate(1 - q, h0, f0, 0.3), SuccessorCandidate(q, h1, f1, 0.7)]
    res = fit_local(
        H, Fm, A, PI, R, succ, theta_check=theta_check, gamma=a.gamma,
        lambda_main=a.lambda_main, lambda_mod=a.lambda_mod, sigma_sq=a.sigma_sq, eps=EPS,
    )
    expected = R[-1] + a.gamma * (
        (1 - q) * _v(h0, f0, 0.3, theta_check, a.p_h) + q * _v(h1, f1, 0.7, theta_check, a.p_h)
    )
    assert res.Y[-1] == pytest.approx(expected, abs=1e-14)


def test_no_successor_and_gamma_zero_targets():
    rng = np.random.default_rng(5)
    a = HP.agent("cp")
    H, Fm, A, PI, R = _dataset(rng, 6, a.p_h, a.p_f)
    theta_check = rng.normal(0, 0.5, a.p_h + a.p_f)
    res = fit_local(
        H, Fm, A, PI, R, [], theta_check=theta_check, gamma=a.gamma,
        lambda_main=a.lambda_main, lambda_mod=a.lambda_mod, sigma_sq=a.sigma_sq, eps=EPS,
    )
    assert res.no_successor
    assert res.Y[-1] == R[-1]  # target R_n alone
    r = HP.agent("rel")
    H, Fm, A, PI, R = _dataset(rng, 9, r.p_h, r.p_f)
    res = fit_local(
        H, Fm, A, PI, R, [SuccessorCandidate(1.0, H[0], Fm[0], 0.5)],
        theta_check=rng.normal(0, 1, r.p_h + r.p_f), gamma=0.0,
        lambda_main=r.lambda_main, lambda_mod=r.lambda_mod, sigma_sq=r.sigma_sq, eps=EPS,
    )
    np.testing.assert_array_equal(res.Y, R)  # gamma = 0: Y = R
    assert res.spectral_norm is None and not res.no_successor


def test_backups_iterate_and_warm_start_default():
    rng = np.random.default_rng(8)
    a = HP.agent("aya")
    H, Fm, A, PI, R = _dataset(rng, 30, a.p_h, a.p_f)
    kw = dict(gamma=a.gamma, lambda_main=a.lambda_main, lambda_mod=a.lambda_mod, sigma_sq=a.sigma_sq, eps=EPS)
    one = fit_local(H, Fm, A, PI, R, [], theta_check=None, **kw)
    zero = fit_local(H, Fm, A, PI, R, [], theta_check=np.zeros(a.p_h + a.p_f), **kw)
    np.testing.assert_array_equal(one.theta, zero.theta)
    two = fit_local(H, Fm, A, PI, R, [], theta_check=None, j_backups=2, **kw)
    again = fit_local(H, Fm, A, PI, R, [], theta_check=one.theta, **kw)
    np.testing.assert_allclose(two.theta, again.theta, rtol=0, atol=1e-14)
    assert two.backups == 2


def test_spectral_norm_diagnostic():
    rng = np.random.default_rng(9)
    a = HP.agent("aya")
    H, Fm, A, PI, R = _dataset(rng, 12, a.p_h, a.p_f)
    theta_check = rng.normal(0, 0.5, a.p_h + a.p_f)
    sh, sf = rng.uniform(-1, 1, a.p_h), rng.uniform(-1, 1, a.p_f)
    res = fit_local(
        H, Fm, A, PI, R, [SuccessorCandidate(1.0, sh, sf, 0.6)], theta_check=theta_check, gamma=a.gamma,
        lambda_main=a.lambda_main, lambda_mod=a.lambda_mod, sigma_sq=a.sigma_sq, eps=EPS,
    )
    alpha = theta_check[a.p_h:]
    Xp = np.zeros((12, a.p_h + a.p_f))
    for k in range(11):
        ind = 1.0 if Fm[k + 1] @ alpha > 0 else 0.0
        Xp[k] = np.r_[H[k + 1], (ind - PI[k + 1]) * Fm[k + 1]]
    ind = 1.0 if sf @ alpha > 0 else 0.0
    Xp[-1] = np.r_[sh, (ind - 0.6) * sf]
    X = design_matrix(H, Fm, A, PI)
    G = np.linalg.inv(X.T @ X + penalty_matrix(a.p_h, a.p_f, a.lambda_main, a.lambda_mod) + EPS * np.eye(11))
    assert res.spectral_norm == pytest.approx(np.linalg.norm(G @ X.T @ Xp, 2), rel=1e-12)


def test_value_function_matches_equation():
    h = np.array([1.0, 0.5])
    f = np.array([1.0, -0.5])
    beta = np.array([0.2, 0.4])
    for alpha in (np.array([0.3, 0.1]), np.array([-0.3, 0.1])):
        fa = float(f @ alpha)
        assert value(h, f, 0.3, beta, alpha) == pytest.approx(0.4 + max(fa, 0) - 0.3 * fa)


# --------------------------------------------------------------------- EB


def _eb_inputs(agent, N, seed, spread=1.0):
    a = HP.agent(agent)
    rng = np.random.default_rng(seed)
    alphas, covs = [], []
    for _ in range(N):
        n = int(rng.integers(5, 120))
        F = rng.uniform(-1, 1, (n, a.p_f))
        F[:, 0] = 1.0
        X = 0.5 * F
        lam = np.diag([a.lambda_main] + [a.lambda_mod] * (a.p_f - 1))
        cov = a.sigma_sq * np.linalg.inv(X.T @ X + lam + EPS * np.eye(a.p_f))
        alpha = rng.normal(0, spread * np.sqrt([a.tau0_act] + [a.tau0_mod] * (a.p_f - 1)))
        alpha = alpha + rng.multivariate_normal(np.zeros(a.p_f), cov)
        alphas.append(alpha)
        covs.append(0.5 * (cov + cov.T))
    return a, alphas, covs


def _hyper(a, alphas, covs, **kw):
    params = dict(
        tau0=(a.tau0_act, a.tau0_mod), nu0=HP.nu0, lambda0=HP.lambda0, eps=EPS, k_max=HP.k_max, eps_tol=HP.eps_tol
    )
    params.update(kw)
    return EB.hyper_update(alphas, covs, **params)


@pytest.mark.parametrize("agent,N,seed", [("aya", 5, 1), ("aya", 12, 2), ("cp", 8, 3), ("rel", 6, 4), ("rel", 25, 5)])
def test_eb_agrees_with_testbed(agent, N, seed):
    deploy = load_testbed()
    a, alphas, covs = _eb_inputs(agent, N, seed)
    ours = _hyper(a, alphas, covs)
    assert ours.converged and ours.fallback_used is None

    cfg = deploy.AgentConfig(
        agent=agent,
        gamma=a.gamma,
        sigma_sq=a.sigma_sq,
        n_runin_main=a.lambda_main / 1.0,
        n_runin_mod=a.lambda_mod / 0.25,
        layout="doc",
    )
    tb = deploy.AgentLearner(cfg)
    assert tb.pf == a.p_f
    np.testing.assert_allclose(tb.tau0, [a.tau0_act, a.tau0_mod], rtol=1e-15)
    tb.K_MAX = 10_000
    tb.EPS_TOL = 1e-12
    for i, (al, cv) in enumerate(zip(alphas, covs)):
        buf = tb.buffer(i)
        buf.alpha = np.asarray(al, dtype=float)
        buf.cov_aa = np.asarray(cv, dtype=float)
    dyads = list(range(N))
    assert tb.update_hyper(dyads)
    tb.shrink(dyads, n_w=N)

    np.testing.assert_allclose(ours.tau, tb.tau_sq, rtol=0, atol=1e-6)
    np.testing.assert_allclose(ours.theta, tb.Theta, rtol=0, atol=1e-6)
    for i in dyads:
        mean, cov, _ = EB.posterior(alphas[i], covs[i], ours.theta, ours.tau, EPS)
        tb_mean, tb_cov = tb.posterior[i]
        np.testing.assert_allclose(mean, tb_mean, rtol=0, atol=1e-6)
        np.testing.assert_allclose(cov, tb_cov, rtol=0, atol=1e-6)


def test_eb_respects_floor_and_reports_l1():
    a = HP.agent("cp")
    rng = np.random.default_rng(0)
    # Identical, nearly exact local fits: no between-dyad spread, so the
    # maximizer approaches the closed-form floor of eq. (floor) from above
    # (main_new.tex Proposition `floor`; equality in the limit V_i -> 0).
    alpha = rng.normal(0, 0.05, a.p_f)
    cov = np.diag([1e-9] * a.p_f)
    res = _hyper(a, [alpha.copy() for _ in range(10)], [cov.copy() for _ in range(10)])
    assert res.converged
    floor = EB.floors((a.tau0_act, a.tau0_mod), HP.nu0, 10)
    np.testing.assert_allclose(res.floor, floor)
    assert np.all(res.tau >= res.floor)
    assert res.at_floor() == [True, True]
    # Noisier local fits keep tau strictly above the floor.
    noisy = _hyper(a, [alpha.copy() for _ in range(10)], [np.diag([0.002] * a.p_f) for _ in range(10)])
    assert noisy.converged and noisy.at_floor() == [False, False]
    assert np.all(noisy.tau > noisy.floor)
    for seed in range(20):
        _, alphas, covs = _eb_inputs("aya", 7, 100 + seed, spread=float(seed % 4))
        r = _hyper(HP.agent("aya"), alphas, covs)
        assert np.all(r.tau >= r.floor)
        assert r.converged


def test_eb_floor_uses_n_w():
    tau0 = (0.01, 0.03)
    np.testing.assert_allclose(EB.floors(tau0, 4.0, 5), [4 * 0.01 / 11, 4 * 0.03 / 11])
    assert HP.eb_floor("aya", 5) == pytest.approx(tuple(EB.floors((HP.agent("aya").tau0_act, HP.agent("aya").tau0_mod), 4.0, 5)))


def test_eb_kmax_one_falls_back_to_prior_then_previous():
    a, alphas, covs = _eb_inputs("aya", 8, 21, spread=2.0)
    res = _hyper(a, alphas, covs, k_max=1)
    assert not res.converged
    assert res.fallback_used == "prior"
    np.testing.assert_array_equal(res.theta, np.zeros(a.p_f))
    np.testing.assert_array_equal(res.tau, [a.tau0_act, a.tau0_mod])
    assert res.iterations == 1 and res.grad_inf > HP.eps_tol
    # The solver's own (rejected) point is kept for the record.
    assert np.all(np.isfinite(res.solver_tau))
    prev_theta = np.full(a.p_f, 0.01)
    prev_tau = np.array([0.02, 0.05])
    res2 = _hyper(a, alphas, covs, k_max=1, warm_tau=prev_tau, previous=(prev_theta, prev_tau))
    assert res2.fallback_used == "previous"
    np.testing.assert_array_equal(res2.theta, prev_theta)
    np.testing.assert_array_equal(res2.tau, prev_tau)
    np.testing.assert_array_equal(res2.warm_start_tau, np.maximum(prev_tau, res2.floor))
    # Posteriors are recomputed from this week's fits under the fallback values.
    mean, cov, w = EB.posterior(alphas[0], covs[0], res2.theta, res2.tau, EPS)
    s0inv = np.linalg.inv(np.diag([0.02] + [0.05] * (a.p_f - 1)) + EPS * np.eye(a.p_f))
    vinv = np.linalg.inv(covs[0] + EPS * np.eye(a.p_f))
    exp_cov = np.linalg.inv(s0inv + vinv)
    np.testing.assert_allclose(cov, exp_cov, rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(mean, exp_cov @ (s0inv @ prev_theta + vinv @ alphas[0]), rtol=1e-10, atol=1e-14)
    assert 0.0 < w < 1.0


def test_eb_nonfinite_input_falls_back():
    a, alphas, covs = _eb_inputs("cp", 6, 30)
    alphas[2] = np.full(a.p_f, np.nan)
    res = _hyper(a, alphas, covs)
    assert not res.converged and res.fallback_used == "prior"


def test_eb_is_deterministic():
    a, alphas, covs = _eb_inputs("aya", 15, 44)
    r1 = _hyper(a, alphas, covs, warm_tau=np.array([0.02, 0.01]))
    r2 = _hyper(a, alphas, covs, warm_tau=np.array([0.02, 0.01]))
    assert np.array_equal(r1.tau, r2.tau) and np.array_equal(r1.theta, r2.theta)
    assert r1.iterations == r2.iterations and r1.objective == r2.objective


def test_eb_gradient_matches_finite_differences():
    a, alphas, covs = _eb_inputs("aya", 9, 50)
    prob = EB._Problem(alphas, covs, (a.tau0_act, a.tau0_mod), HP.nu0, HP.lambda0, EPS)
    tau = np.array([0.012, 0.02])
    f0, g, theta, g_theta = prob.evaluate(tau)
    for b in range(2):
        h = 1e-7
        tp, tm = tau.copy(), tau.copy()
        tp[b] += h
        tm[b] -= h
        fd = (prob.evaluate(tp)[0] - prob.evaluate(tm)[0]) / (2 * h)
        assert g[b] == pytest.approx(fd, rel=1e-5)
    # Theta is profiled exactly: its gradient vanishes.
    assert np.max(np.abs(g_theta)) < 1e-9


def test_projected_gradient_ignores_outward_gradient_at_floor():
    floor = np.array([0.01, 0.02])
    assert EB.projected_grad_inf(floor.copy(), np.array([5.0, -1e-6]), np.zeros(3), floor) == pytest.approx(1e-6)
    assert EB.projected_grad_inf(floor * 2, np.array([5.0, 0.0]), np.zeros(3), floor) == 5.0


# -------------------------------------------------------------- allocation


def test_rho_at_zero_and_limits():
    assert float(AL.rho(0.0, 0.2, 0.8, 3.5, 40.0, 1.0)) == pytest.approx(1 / 3, abs=1e-15)
    assert HP.rho_at_zero == pytest.approx(1 / 3)
    assert float(AL.rho(0.0, 0.2, 0.8, 1.0, 40.0, 1.0)) == pytest.approx(0.5)
    assert float(AL.rho(10.0, 0.2, 0.8, 3.5, 40.0, 1.0)) == pytest.approx(0.8)
    assert float(AL.rho(-10.0, 0.2, 0.8, 3.5, 40.0, 1.0)) == pytest.approx(0.2)
    # Per-agent slope on the normalized scale (eq. (rho-params)).
    assert [HP.agent(x).b_rho for x in ("aya", "cp", "rel")] == [40.0, 100.0, 100.0]


def test_allocation_in_bounds_and_symmetric_limit():
    bank = AL.default_bank(HP.mc_seed, HP.mc_samples)
    rng = np.random.default_rng(2)
    for _ in range(200):
        mu, v = rng.normal(0, 0.3), rng.exponential(0.05)
        for dt in ("aya_message", "cp_message", "dyad_game"):
            pi = AL.agent_probability(dt, mu, v, hp=HP, bank=bank)
            assert 0.2 <= pi <= 0.8
    # Large posterior variance pulls pi toward (L_min + L_max) / 2.
    assert AL.agent_probability("cp_message", 0.0, 100.0, hp=HP, bank=bank) == pytest.approx(0.5, abs=0.03)
    # Zero variance: pi = rho(mu).
    assert AL.agent_probability("aya_message", 0.0, 0.0, hp=HP, bank=bank) == pytest.approx(1 / 3, abs=1e-15)


def test_allocation_equals_testbed_allocator():
    deploy = load_testbed()
    bank = AL.default_bank(HP.mc_seed, HP.mc_samples)
    for agent in ("aya", "cp", "rel"):
        a = HP.agent(agent)
        cfg = deploy.AgentConfig(
            agent=agent, gamma=a.gamma, sigma_sq=a.sigma_sq, n_runin_main=16, n_runin_mod=24, b_rho=a.b_rho
        )
        alloc = deploy.Allocator(cfg, seed=HP.mc_seed)
        np.testing.assert_array_equal(alloc.bank, bank)
        rng = np.random.default_rng(7)
        for _ in range(50):
            mu, v = float(rng.normal(0, 0.1)), float(rng.exponential(0.01))
            assert AL.agent_probability(a.decision_type, mu, v, hp=HP, bank=bank) == alloc(mu, v)


def test_mc_bank_from_app_sampler(learner_app):
    bank = AL.mc_bank()
    np.testing.assert_array_equal(bank, np.random.default_rng(HP.mc_seed).standard_normal(HP.mc_samples))
    cursor = learner_app.sampler.cursor()
    AL.agent_probability("aya_message", 0.1, 0.01)
    assert learner_app.sampler.cursor() == cursor  # the bank never consumes the streams


# ---------------------------------------------------------------------- C3


def _valid_policy(agent="aya", N=4, seed=60):
    a, alphas, covs = _eb_inputs(agent, N, seed)
    res = _hyper(a, alphas, covs)
    d = a.p_h + a.p_f
    fits, posts = {}, {}
    rng = np.random.default_rng(seed)
    for i, (al, cv) in enumerate(zip(alphas, covs)):
        cov = np.eye(d) * 0.01
        cov[a.p_h:, a.p_h:] = cv
        fits[f"g{i}"] = (np.r_[rng.normal(0, 0.1, a.p_h), al], cov)
        m, c, _ = EB.posterior(al, cv, res.theta, res.tau, EPS)
        posts[f"g{i}"] = (m, c)
    return a, res, fits, posts


def _check(a, res, fits, posts, **kw):
    params = dict(theta=res.theta, tau=res.tau, floor=res.floor, sigma0=res.sigma0)
    params.update(kw)
    return V.check_policy(
        a.decision_type, fits, params["theta"], params["tau"], params["floor"], params["sigma0"], posts, eps=EPS
    )


def test_c3_passes_on_a_valid_policy():
    for agent in ("aya", "cp", "rel"):
        a, res, fits, posts = _valid_policy(agent)
        out = _check(a, res, fits, posts)
        assert out["passed"], out["failures"]
        assert set(out["checks"]) == set(V.CHECKS)


def test_c3_detects_each_failure():
    a, res, fits, posts = _valid_policy()
    bad = dict(fits)
    th, cv = bad["g0"]
    bad["g0"] = (np.r_[th[:-1], np.nan], cv)
    out = _check(a, res, bad, posts)
    assert not out["passed"] and not out["checks"][V.CHECK_FINITE]

    bad = dict(fits)
    th, cv = bad["g1"]
    cv = cv.copy()
    cv[0, 1] += 1e-3
    bad["g1"] = (th, cv)
    out = _check(a, res, bad, posts)
    assert not out["checks"][V.CHECK_VARIANCE]

    bad_post = dict(posts)
    m, c = bad_post["g2"]
    c = c.copy()
    c[0, 0] = -1.0
    bad_post["g2"] = (m, c)
    out = _check(a, res, fits, bad_post)
    assert not out["checks"][V.CHECK_VARIANCE]

    out = _check(a, res, fits, posts, tau=res.floor * 0.5)
    assert not out["checks"][V.CHECK_FLOOR]
    out = _check(a, res, fits, posts, tau=res.floor * (1 - 1e-7))
    assert out["checks"][V.CHECK_FLOOR]  # within the (1 - 1e-6) tolerance

    inflated = {g: (m, c + 0.5 * np.eye(len(m))) for g, (m, c) in posts.items()}
    out = _check(a, res, fits, inflated)
    assert not out["checks"][V.CHECK_POOLING]

    short = {g: (th[:-1], cv[:-1, :-1]) for g, (th, cv) in fits.items()}
    out = _check(a, res, short, posts)
    assert not out["checks"][V.CHECK_DIMS]
    out = _check(a, res, fits, posts, theta=np.zeros(a.p_f + 1))
    assert not out["checks"][V.CHECK_DIMS]
