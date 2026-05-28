"""
Deterministic pre-sampled random stream for ADAPTS-HCT.

Goal: bit-for-bit reproducibility of every algorithm decision (action,
posterior sample, perturbation) given (a) the same pre-sampled buffer of
primitives and (b) the same chronological sequence of events.

Design:
- Before the study starts, a single ``DeterministicSampleStream`` is generated
  by drawing a long sequence of standard Gaussian floats and a long sequence
  of uniform [0, 1) floats from a *named* numpy.Generator seeded once.
  These two sequences and their cursors are stored to a single ``.npz`` file.
- At runtime the algorithm holds this stream and, whenever it would call
  ``rng.standard_normal`` / ``rng.multivariate_normal`` / ``rng.uniform`` /
  ``rng.integers``, it consumes the next primitive(s) from the stream
  instead. Consumption is serialized by an internal lock.
- The stream's cursor position is captured before and after each consumption
  and stamped onto the Action / EB snapshot rows that triggered it. Together
  with the original buffer file, this is sufficient to deterministically
  replay the algorithm.

Multivariate sampling uses
``y = mean + L @ z`` where ``L = U sqrt(D)`` from the eigendecomposition of
``cov`` (``eigh``). We choose eigh over Cholesky to (a) match the existing
covariance stabilization pipeline used elsewhere in the algorithm and (b)
remain well-defined when ``cov`` is positive semi-definite but not strictly
positive definite. The transform ``L`` is otherwise deterministic, so any
two callers given the same ``mean``, ``cov``, and the same buffer cursor
position will produce the same ``y``.
"""
from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np


class SampleBufferExhausted(RuntimeError):
    """Raised when the pre-sampled stream runs out of primitives."""


class DeterministicSampleStream:
    NORMAL_KEY = "normals"
    UNIFORM_KEY = "uniforms"
    META_NORMAL_CURSOR = "normal_cursor"
    META_UNIFORM_CURSOR = "uniform_cursor"
    META_SEED = "seed"

    def __init__(
        self,
        normals: np.ndarray,
        uniforms: np.ndarray,
        normal_cursor: int = 0,
        uniform_cursor: int = 0,
        seed: int | None = None,
    ):
        self._normals = np.asarray(normals, dtype=np.float64)
        self._uniforms = np.asarray(uniforms, dtype=np.float64)
        if self._normals.ndim != 1 or self._uniforms.ndim != 1:
            raise ValueError("normals and uniforms must be 1-D arrays")
        if normal_cursor < 0 or normal_cursor > len(self._normals):
            raise ValueError(
                f"normal_cursor {normal_cursor} out of range [0, {len(self._normals)}]"
            )
        if uniform_cursor < 0 or uniform_cursor > len(self._uniforms):
            raise ValueError(
                f"uniform_cursor {uniform_cursor} out of range [0, {len(self._uniforms)}]"
            )
        self._normal_cursor = int(normal_cursor)
        self._uniform_cursor = int(uniform_cursor)
        self._seed = seed
        self._lock = threading.Lock()

    # ------------------------------------------------------------ properties

    @property
    def n_normals(self) -> int:
        return len(self._normals)

    @property
    def n_uniforms(self) -> int:
        return len(self._uniforms)

    @property
    def seed(self) -> int | None:
        return self._seed

    def cursor(self) -> dict[str, int]:
        """Snapshot of the current consumption position."""
        with self._lock:
            return {"normal": self._normal_cursor, "uniform": self._uniform_cursor}

    def restore(self, cursor: dict[str, int]) -> None:
        """Reset cursors. Used on server restart to resume mid-study."""
        with self._lock:
            n = int(cursor.get("normal", 0))
            u = int(cursor.get("uniform", 0))
            if n > len(self._normals):
                raise SampleBufferExhausted(
                    f"normal cursor {n} exceeds buffer size {len(self._normals)}"
                )
            if u > len(self._uniforms):
                raise SampleBufferExhausted(
                    f"uniform cursor {u} exceeds buffer size {len(self._uniforms)}"
                )
            self._normal_cursor = n
            self._uniform_cursor = u

    # ----------------------------------------------------------------- draws

    def draw_normal(self, dim: int = 1) -> np.ndarray:
        """Pull the next `dim` standard normal primitives (returns a 1-D array)."""
        if dim < 0:
            raise ValueError("dim must be non-negative")
        with self._lock:
            end = self._normal_cursor + dim
            if end > self.n_normals:
                raise SampleBufferExhausted(
                    f"normal buffer exhausted: cursor={self._normal_cursor} "
                    f"+ dim={dim} > size={self.n_normals}"
                )
            out = self._normals[self._normal_cursor:end].copy()
            self._normal_cursor = end
            return out

    def draw_uniform(self) -> float:
        """Pull the next uniform [0, 1) primitive."""
        with self._lock:
            if self._uniform_cursor >= self.n_uniforms:
                raise SampleBufferExhausted(
                    f"uniform buffer exhausted: cursor={self._uniform_cursor}"
                )
            u = float(self._uniforms[self._uniform_cursor])
            self._uniform_cursor += 1
            return u

    def draw_bernoulli(self, p: float = 0.5) -> int:
        """Pull a Bernoulli(p) by thresholding the next uniform primitive."""
        return 1 if self.draw_uniform() < float(p) else 0

    # -------------------------------------------------------------- helpers

    def multivariate_normal(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Deterministically sample y = mean + L @ z, with z drawn from the
        normal buffer and L = U sqrt(D) from eigh(cov). Any two callers given
        the same (mean, cov) and the same cursor will produce the same y.
        """
        mean = np.asarray(mean, dtype=np.float64)
        cov = np.asarray(cov, dtype=np.float64)
        if mean.ndim != 1:
            raise ValueError("mean must be 1-D")
        d = mean.shape[0]
        if cov.shape != (d, d):
            raise ValueError(f"cov shape {cov.shape} != ({d}, {d})")
        z = self.draw_normal(d)
        L = _eigh_sqrt(cov)
        return mean + L @ z

    # -------------------------------------------------------- persistence

    def save(self, path: str) -> str:
        """Write the buffer + cursors to a single ``.npz`` file. Returns the path."""
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        with self._lock:
            payload = {
                self.NORMAL_KEY: self._normals,
                self.UNIFORM_KEY: self._uniforms,
                self.META_NORMAL_CURSOR: np.int64(self._normal_cursor),
                self.META_UNIFORM_CURSOR: np.int64(self._uniform_cursor),
            }
            if self._seed is not None:
                payload[self.META_SEED] = np.int64(self._seed)
        np.savez(path, **payload)
        # numpy auto-appends .npz if missing
        if not path.endswith(".npz"):
            path = path + ".npz"
        return path

    @classmethod
    def load(cls, path: str) -> "DeterministicSampleStream":
        if not path.endswith(".npz") and not os.path.exists(path):
            path = path + ".npz"
        data = np.load(path)
        seed = int(data[cls.META_SEED]) if cls.META_SEED in data.files else None
        return cls(
            normals=data[cls.NORMAL_KEY],
            uniforms=data[cls.UNIFORM_KEY],
            normal_cursor=int(data[cls.META_NORMAL_CURSOR])
            if cls.META_NORMAL_CURSOR in data.files
            else 0,
            uniform_cursor=int(data[cls.META_UNIFORM_CURSOR])
            if cls.META_UNIFORM_CURSOR in data.files
            else 0,
            seed=seed,
        )

    @classmethod
    def fresh(
        cls,
        n_normals: int,
        n_uniforms: int,
        seed: int,
    ) -> "DeterministicSampleStream":
        """Generate a fresh buffer from a single seed."""
        if n_normals <= 0 or n_uniforms <= 0:
            raise ValueError("n_normals and n_uniforms must be positive")
        rng = np.random.default_rng(seed)
        normals = rng.standard_normal(n_normals).astype(np.float64)
        uniforms = rng.random(n_uniforms).astype(np.float64)
        return cls(normals=normals, uniforms=uniforms, seed=int(seed))


def _eigh_sqrt(cov: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
    """
    Symmetric square root of a PSD matrix: L = U sqrt(D) such that L @ L.T = cov
    (after PSD stabilization). Eigh is deterministic for the same input.
    """
    cov = (cov + cov.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, jitter)
    return eigvecs * np.sqrt(eigvals)


def closed_form_action_prob(
    state: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    expand_to_phi,
    eta: float = 1.0,
) -> float:
    """
    Exact marginal P(action = 1) under the probit Thompson-sampling allocation
    with inverse temperature `eta`, marginalized over theta ~ N(mean, cov).

    Per main.tex Eq. (probit-ts-closed):
        P(A=1 | s) = Phi(eta * m / sqrt(1 + eta^2 * v))
    where m = (phi_1 - phi_0)^T mean and v = (phi_1 - phi_0)^T cov (phi_1 - phi_0).

    The eta -> infinity limit recovers the hard-argmax TS formula
        Phi(m / sqrt(v)).

    No sampling — does not consume from the buffer.
    """
    from math import erf, sqrt

    phi0 = expand_to_phi(state, 0)
    phi1 = expand_to_phi(state, 1)
    d = phi1 - phi0
    m = float(d @ mean)
    v = float(d @ cov @ d)
    denom_sq = 1.0 + (eta * eta) * max(v, 0.0)
    denom = float(np.sqrt(denom_sq))
    z = (eta * m) / denom
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))
