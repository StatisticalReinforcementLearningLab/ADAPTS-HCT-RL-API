"""Unit tests for app.deterministic_sampler.DeterministicSampleStream."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from app.deterministic_sampler import (
    DeterministicSampleStream,
    SampleBufferExhausted,
    closed_form_action_prob,
)


class TestBasicDraws:
    def test_fresh_has_expected_sizes(self):
        s = DeterministicSampleStream.fresh(n_normals=100, n_uniforms=50, seed=1)
        assert s.n_normals == 100
        assert s.n_uniforms == 50
        assert s.cursor() == {"normal": 0, "uniform": 0}
        assert s.seed == 1

    def test_draw_normal_advances_cursor(self):
        s = DeterministicSampleStream.fresh(100, 50, seed=1)
        a = s.draw_normal(3)
        assert a.shape == (3,)
        assert s.cursor()["normal"] == 3
        b = s.draw_normal(2)
        assert s.cursor()["normal"] == 5
        # Independent draws — no overlap
        a2 = s.draw_normal(3)
        assert not np.allclose(a, a2)

    def test_draw_uniform_is_in_range_and_advances(self):
        s = DeterministicSampleStream.fresh(100, 50, seed=1)
        u = s.draw_uniform()
        assert 0.0 <= u < 1.0
        assert s.cursor()["uniform"] == 1

    def test_draw_bernoulli_consumes_one_uniform(self):
        s = DeterministicSampleStream.fresh(100, 50, seed=1)
        a = s.draw_bernoulli(0.5)
        assert a in (0, 1)
        assert s.cursor() == {"normal": 0, "uniform": 1}

    def test_two_streams_with_same_seed_are_identical(self):
        a = DeterministicSampleStream.fresh(100, 50, seed=1)
        b = DeterministicSampleStream.fresh(100, 50, seed=1)
        assert np.array_equal(a.draw_normal(10), b.draw_normal(10))
        assert a.draw_uniform() == b.draw_uniform()


class TestMultivariate:
    def test_multivariate_matches_explicit_transform(self):
        s = DeterministicSampleStream.fresh(1000, 10, seed=42)
        mean = np.array([1.0, 2.0, 3.0])
        cov = np.array([[1.0, 0.3, 0.0], [0.3, 2.0, 0.1], [0.0, 0.1, 0.5]])
        # Capture the z we're about to consume for the explicit formula
        cursor_before = s.cursor()["normal"]
        # Use a sibling stream to peek at the primitives (doesn't advance ours)
        peek = DeterministicSampleStream.fresh(1000, 10, seed=42)
        peek._normal_cursor = cursor_before
        z = peek.draw_normal(3)
        sample = s.multivariate_normal(mean, cov)
        # Reproduce L @ z + mean
        cov_sym = (cov + cov.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(cov_sym)
        eigvals = np.maximum(eigvals, 1e-12)
        L = eigvecs * np.sqrt(eigvals)
        expected = mean + L @ z
        assert np.allclose(sample, expected)

    def test_multivariate_consumes_d_normals(self):
        s = DeterministicSampleStream.fresh(1000, 10, seed=42)
        s.multivariate_normal(np.zeros(5), np.eye(5))
        assert s.cursor()["normal"] == 5

    def test_multivariate_approx_recovers_moments(self):
        s = DeterministicSampleStream.fresh(10_000, 100, seed=7)
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        # Draw 2000 samples, check mean + cov are approximately recovered.
        samples = np.stack([s.multivariate_normal(mean, cov) for _ in range(2000)])
        emp_mean = samples.mean(axis=0)
        emp_cov = np.cov(samples.T)
        assert np.allclose(emp_mean, mean, atol=0.1)
        assert np.allclose(emp_cov, cov, atol=0.2)


class TestExhaustion:
    def test_normal_exhaustion_raises(self):
        s = DeterministicSampleStream.fresh(5, 2, seed=1)
        s.draw_normal(3)
        with pytest.raises(SampleBufferExhausted):
            s.draw_normal(3)

    def test_uniform_exhaustion_raises(self):
        s = DeterministicSampleStream.fresh(5, 2, seed=1)
        s.draw_uniform()
        s.draw_uniform()
        with pytest.raises(SampleBufferExhausted):
            s.draw_uniform()


class TestPersistence:
    def test_save_load_round_trips_primitives_and_cursor(self):
        s = DeterministicSampleStream.fresh(100, 50, seed=9)
        s.draw_normal(7)
        s.draw_uniform()
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        tmp.close()
        try:
            s.save(tmp.name)
            loaded = DeterministicSampleStream.load(tmp.name)
            assert loaded.n_normals == 100
            assert loaded.n_uniforms == 50
            assert loaded.cursor() == {"normal": 7, "uniform": 1}
            assert loaded.seed == 9
            # Same primitives
            rest_a = s.draw_normal(3)
            rest_b = loaded.draw_normal(3)
            assert np.array_equal(rest_a, rest_b)
        finally:
            os.unlink(tmp.name)

    def test_restore_cursor_to_arbitrary_position(self):
        s = DeterministicSampleStream.fresh(100, 50, seed=1)
        s.restore({"normal": 7, "uniform": 2})
        assert s.cursor() == {"normal": 7, "uniform": 2}

    def test_restore_cursor_out_of_range_raises(self):
        s = DeterministicSampleStream.fresh(5, 3, seed=1)
        with pytest.raises(SampleBufferExhausted):
            s.restore({"normal": 6, "uniform": 0})


class TestClosedFormActionProb:
    def test_zero_variance_handled(self):
        state = np.array([1.0, 0.5])

        def expand(s, a):
            return np.array([s[0], s[1], a])

        mean = np.zeros(3)
        cov = np.zeros((3, 3))  # zero variance
        p = closed_form_action_prob(state, mean, cov, expand)
        assert p == 0.5  # mu_diff = 0

    def test_dominant_positive_coefficient_saturates_to_one(self):
        """With the action coefficient at +inf effectively, P(a=1) -> 1."""
        state = np.array([1.0])

        def expand(s, a):
            return np.array([s[0], a])

        # Q(s, a) = state + 100 * a; so P(a=1) should be ~1
        mean = np.array([0.0, 100.0])
        cov = np.eye(2) * 1e-6
        p = closed_form_action_prob(state, mean, cov, expand)
        assert p > 0.99

    def test_symmetric_zero_mean_gives_half(self):
        state = np.array([1.0])

        def expand(s, a):
            return np.array([s[0], a])

        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        p = closed_form_action_prob(state, mean, cov, expand)
        assert p == pytest.approx(0.5)
