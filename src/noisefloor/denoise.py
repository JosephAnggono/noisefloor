"""Low-rank approximation of a weight matrix with the rank chosen by the noise floor."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .spectrum import SpectrumReport, analyze, as_matrix, to_numpy


def truncate(weight, rank: int) -> np.ndarray:
    """Best rank-``rank`` approximation in Frobenius norm (Eckart-Young)."""
    w = to_numpy(weight)
    u, s, vt = np.linalg.svd(as_matrix(w), full_matrices=False)
    rank = int(np.clip(rank, 0, s.size))
    return ((u[:, :rank] * s[:rank]) @ vt[:rank]).reshape(w.shape)


def shrink_singular_values(s: np.ndarray, n: int, q: float, sigma2: float) -> np.ndarray:
    """Optimal singular-value shrinkage for Frobenius loss (Gavish & Donoho, 2017).

    A singular value observed above the noise edge is biased upward and its singular
    vectors are only partly aligned with the true ones. The shrinker corrects for
    both, and maps everything inside the noise bulk to zero.
    """
    scale = np.sqrt(sigma2 * n)
    y = np.asarray(s, dtype=float) / scale
    inside = (y**2 - q - 1.0) ** 2 - 4.0 * q
    out = np.zeros_like(y)
    keep = (y > 1.0 + np.sqrt(q)) & (inside > 0.0)
    out[keep] = np.sqrt(inside[keep]) / y[keep]
    return out * scale


def denoise(
    weight,
    method: Literal["hard", "shrink"] = "hard",
    alpha: float = 0.01,
) -> tuple[np.ndarray, SpectrumReport]:
    """Remove the noise bulk from a weight matrix.

    ``hard`` keeps the detected spikes unchanged and zeroes the rest.
    ``shrink`` additionally shrinks the kept singular values (Gavish & Donoho, 2017).
    Returns the cleaned weight (same shape as the input) and the spectrum report.
    """
    if method not in ("hard", "shrink"):
        raise ValueError(f"unknown method {method!r}")
    w = to_numpy(weight)
    report = analyze(w, alpha=alpha)
    u, s, vt = np.linalg.svd(as_matrix(w), full_matrices=False)
    k = report.n_spikes
    kept = s[:k]
    if method == "shrink":
        kept = shrink_singular_values(kept, max(report.shape), report.q, report.sigma2)
    return ((u[:, :k] * kept) @ vt[:k]).reshape(w.shape), report


def low_rank_params(shape: tuple[int, int], rank: int) -> int:
    """Parameters needed to store a rank-``rank`` factorisation of a matrix."""
    return rank * (shape[0] + shape[1])
