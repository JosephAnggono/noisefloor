"""Marchenko-Pastur law: the eigenvalue distribution of a pure-noise matrix.

Convention used across the package
----------------------------------
Take a matrix ``W`` with i.i.d. entries of variance ``sigma2``. Let ``n`` be its
larger dimension, ``p`` its smaller one, and ``q = p / n`` (so ``0 < q <= 1``).
The ``p`` eigenvalues of ``(1/n) W W^T`` (equivalently, ``s_i**2 / n`` for the
singular values ``s_i`` of ``W``) then follow the Marchenko-Pastur density

    rho(x) = sqrt((hi - x)(x - lo)) / (2 pi sigma2 q x),   lo <= x <= hi,

with edges ``lo, hi = sigma2 * (1 -/+ sqrt(q))**2``.

Reference: Marchenko & Pastur (1967); Bai & Silverstein (2010), ch. 3.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

_GRID = 4_000  # midpoint-rule resolution for the CDF table (error ~1e-7)


def mp_edges(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    """Lower and upper edge of the Marchenko-Pastur bulk."""
    _check_q(q)
    root = np.sqrt(q)
    return sigma2 * (1.0 - root) ** 2, sigma2 * (1.0 + root) ** 2


def mp_pdf(x, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Marchenko-Pastur density evaluated at ``x`` (zero outside the bulk)."""
    lo, hi = mp_edges(q, sigma2)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    inside = (x > lo) & (x < hi)
    xi = x[inside]
    out[inside] = np.sqrt((hi - xi) * (xi - lo)) / (2.0 * np.pi * sigma2 * q * xi)
    return out


@lru_cache(maxsize=128)
def _cdf_table(q: float) -> tuple[np.ndarray, np.ndarray]:
    """Tabulate the unit-variance MP CDF on a grid that is dense near both edges.

    Substituting ``x = lo + (hi - lo) * (1 - cos t) / 2`` removes the square-root
    behaviour at the edges (and the 1/sqrt(x) singularity at zero when q = 1), so a
    plain midpoint rule in ``t`` converges quickly.
    """
    lo, hi = mp_edges(q)
    half_width = (hi - lo) / 2.0
    t_edges = np.linspace(0.0, np.pi, _GRID + 1)
    t_mid = 0.5 * (t_edges[:-1] + t_edges[1:])
    x_mid = lo + half_width * (1.0 - np.cos(t_mid))
    integrand = (half_width * np.sin(t_mid)) ** 2 / (2.0 * np.pi * q * x_mid)
    cdf = np.concatenate([[0.0], np.cumsum(integrand) * (np.pi / _GRID)])
    cdf /= cdf[-1]  # absorb the tiny quadrature error so the table ends at exactly 1
    x_edges = lo + half_width * (1.0 - np.cos(t_edges))
    return x_edges, cdf


def mp_cdf(x, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Marchenko-Pastur cumulative distribution function."""
    _check_q(q)
    grid, cdf = _cdf_table(float(q))
    return np.interp(np.asarray(x, dtype=float) / sigma2, grid, cdf, left=0.0, right=1.0)


def mp_quantile(prob, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Inverse of :func:`mp_cdf`."""
    _check_q(q)
    grid, cdf = _cdf_table(float(q))
    return sigma2 * np.interp(np.asarray(prob, dtype=float), cdf, grid)


def mp_median(q: float) -> float:
    """Median of the unit-variance Marchenko-Pastur law."""
    return float(mp_quantile(0.5, q))


def _check_q(q: float) -> None:
    if not 0.0 < q <= 1.0:
        raise ValueError(f"aspect ratio q must lie in (0, 1], got {q!r}")
