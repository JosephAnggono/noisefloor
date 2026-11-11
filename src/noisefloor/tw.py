"""Tracy-Widom fluctuations of the largest eigenvalue of a pure-noise matrix.

The Marchenko-Pastur edge is only exact for infinitely large matrices. At finite
size the largest noise eigenvalue wobbles around the edge on a scale of
``n**(-2/3)``, and after centring and scaling it follows the Tracy-Widom law of
order 1 (Johnstone, 2001). Using that law turns "is this eigenvalue above the
edge?" into a proper hypothesis test with a controlled false-alarm rate.

The Tracy-Widom distribution has no closed form. We use the shifted-gamma
approximation of Chiani (2014), accurate to roughly 1e-3 in the CDF, which is far
below the sampling noise in any single weight matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import gamma

# Chiani (2014), Table I: TW1 ~ Gamma(shape, scale) - shift
_SHAPE, _SCALE, _SHIFT = 46.446, 0.186054, 9.84801


def tw1_cdf(x) -> np.ndarray:
    """CDF of the Tracy-Widom distribution of order 1."""
    return gamma.cdf(np.asarray(x, dtype=float) + _SHIFT, _SHAPE, scale=_SCALE)


def tw1_pdf(x) -> np.ndarray:
    """Density of the Tracy-Widom distribution of order 1."""
    return gamma.pdf(np.asarray(x, dtype=float) + _SHIFT, _SHAPE, scale=_SCALE)


def tw1_ppf(prob) -> np.ndarray:
    """Quantile function of the Tracy-Widom distribution of order 1."""
    return gamma.ppf(np.asarray(prob, dtype=float), _SHAPE, scale=_SCALE) - _SHIFT


def johnstone_centre_scale(n: int, p: int) -> tuple[float, float]:
    """Centring and scaling for the top eigenvalue of ``X^T X``, X of shape (n, p).

    For standard Gaussian entries, ``(l_max - centre) / scale`` converges to
    Tracy-Widom of order 1 (Johnstone 2001, Theorem 1.1).
    """
    a, b = np.sqrt(n - 1.0), np.sqrt(float(p))
    centre = (a + b) ** 2
    scale = (a + b) * (1.0 / a + 1.0 / b) ** (1.0 / 3.0)
    return float(centre), float(scale)


def edge_threshold(n: int, p: int, sigma2: float = 1.0, alpha: float = 0.01) -> float:
    """Smallest eigenvalue of ``(1/n) W W^T`` that pure noise reaches with prob. ``alpha``.

    An eigenvalue above this value is declared signal at significance level ``alpha``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    centre, scale = johnstone_centre_scale(n, p)
    return float(sigma2 * (centre + scale * tw1_ppf(1.0 - alpha)) / n)
