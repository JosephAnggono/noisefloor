"""Split the spectrum of a weight matrix into a noise bulk and signal spikes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .mp import mp_cdf, mp_edges, mp_median
from .tw import edge_threshold


@dataclass(frozen=True)
class SpectrumReport:
    """Random-matrix summary of one weight matrix.

    Eigenvalues are ``s**2 / n`` for singular values ``s`` and larger dimension ``n``,
    which puts them on the scale of the Marchenko-Pastur law (see ``noisefloor.mp``).
    """

    shape: tuple[int, int]
    q: float  # aspect ratio p / n in (0, 1]
    singular_values: np.ndarray  # descending
    eigenvalues: np.ndarray  # descending
    sigma2: float  # noise variance per entry
    bulk_edge: float  # asymptotic Marchenko-Pastur upper edge
    threshold: float  # finite-size (Tracy-Widom) detection threshold
    n_spikes: int  # eigenvalues above the threshold
    ks_distance: float  # how far the bulk is from Marchenko-Pastur (0 = perfect fit)
    signal_energy: float  # share of ||W||_F^2 carried by the spikes
    stable_rank: float  # ||W||_F^2 / ||W||_2^2
    effective_rank: float  # exp(entropy of normalised singular values)

    @property
    def rank_fraction(self) -> float:
        """Detected signal rank as a share of the full rank."""
        return self.n_spikes / len(self.singular_values)


def to_numpy(weight) -> np.ndarray:
    """Accept NumPy arrays, nested lists, or framework tensors (PyTorch, JAX, ...)."""
    if hasattr(weight, "detach"):  # torch.Tensor, without importing torch
        weight = weight.detach().cpu().numpy()
    return np.asarray(weight, dtype=float)


def as_matrix(weight) -> np.ndarray:
    """View a dense or convolutional weight as a 2-D matrix (out, everything else)."""
    w = to_numpy(weight)
    if w.ndim < 2:
        raise ValueError(f"expected a weight with at least 2 dimensions, got shape {w.shape}")
    return w.reshape(w.shape[0], -1)


def estimate_noise(
    eigenvalues: np.ndarray, n: int, p: int, alpha: float = 0.01, max_iter: int = 100
) -> tuple[float, int]:
    """Jointly estimate the noise variance and the number of spikes.

    The two are coupled: spikes inflate a naive variance estimate, and the variance
    sets the detection threshold. We solve for both in two stages.

    1. Robust start. Match the median of the non-spike eigenvalues to the
       Marchenko-Pastur median (Gavish & Donoho, 2014), re-detect, and repeat. The
       median barely moves when a few large spikes are present, and the spike count
       can only grow, so this converges.
    2. Efficient refinement. Replace the median by the trace of the non-spike
       eigenvalues, which uses all of them and has about half the sampling error.
       Removing ``k`` signal directions leaves a ``(p - k) x (n - k)`` noise block,
       hence the ``(p - k)(n - k)`` normalisation.

    ``eigenvalues`` must be sorted in descending order. Returns ``(sigma2, n_spikes)``.
    """
    if eigenvalues[0] <= 0.0:
        return 0.0, 0
    if np.median(eigenvalues) <= 1e-12 * eigenvalues[0]:
        # Exactly low-rank input (for example an already-truncated matrix): no noise.
        return 0.0, int(np.sum(eigenvalues > 1e-12 * eigenvalues[0]))

    def count(sigma2: float) -> int:
        return int(np.sum(eigenvalues > edge_threshold(n, p, sigma2, alpha)))

    k, sigma2 = 0, 0.0
    for _ in range(max_iter):
        sigma2 = float(np.median(eigenvalues[k:]) / mp_median((p - k) / (n - k)))
        k_new = max(k, count(sigma2))
        if k_new == k or k_new >= p - 2:
            break
        k = k_new
    for _ in range(max_iter):
        if k >= p - 2:
            break
        sigma2 = float(eigenvalues[k:].sum() * n / ((p - k) * (n - k)))
        k_new = count(sigma2)
        if k_new == k:
            break
        k = k_new
    return sigma2, count(sigma2)


def analyze(weight, alpha: float = 0.01, sigma2: float | None = None) -> SpectrumReport:
    """Analyse one weight matrix.

    Parameters
    ----------
    weight : array of shape (out, in) or (out, in, kh, kw)
    alpha : false-alarm rate of the spike test (per matrix)
    sigma2 : known noise variance per entry (for example the initialisation variance);
        estimated from the spectrum if omitted, see :func:`estimate_noise`
    """
    w = as_matrix(weight)
    n, p = max(w.shape), min(w.shape)
    q = p / n
    s = np.linalg.svd(w, compute_uv=False)
    eig = s**2 / n

    if sigma2 is None:
        sigma2, n_spikes = estimate_noise(eig, n, p, alpha)
    else:
        n_spikes = int(np.sum(eig > edge_threshold(n, p, sigma2, alpha)))
    threshold = edge_threshold(n, p, sigma2, alpha) if sigma2 > 0.0 else 0.0
    bulk = eig[n_spikes:]

    total = float(np.sum(s**2))
    probs = s / s.sum()
    probs = probs[probs > 0]
    return SpectrumReport(
        shape=(w.shape[0], w.shape[1]),
        q=q,
        singular_values=s,
        eigenvalues=eig,
        sigma2=float(sigma2),
        bulk_edge=mp_edges(q, sigma2)[1] if sigma2 > 0.0 else 0.0,
        threshold=threshold,
        n_spikes=n_spikes,
        ks_distance=_ks_distance(bulk, q, sigma2),
        signal_energy=float(np.sum(s[:n_spikes] ** 2) / total),
        stable_rank=total / float(s[0] ** 2),
        effective_rank=float(np.exp(-np.sum(probs * np.log(probs)))),
    )


def analyze_model(weights: Mapping[str, object], alpha: float = 0.01) -> dict[str, SpectrumReport]:
    """Analyse every matrix-shaped entry of a ``{name: weight}`` mapping.

    Works directly on a PyTorch ``state_dict()``. Biases, scalars and matrices whose
    smaller side is below 8 are skipped: there is no bulk to speak of.
    """
    reports = {}
    for name, weight in weights.items():
        w = to_numpy(weight)
        if w.ndim >= 2 and min(as_matrix(w).shape) >= 8:
            reports[name] = analyze(w, alpha=alpha)
    return reports


def format_reports(reports: Mapping[str, SpectrumReport]) -> str:
    """Plain-text table, one row per layer."""
    header = f"{'layer':<24}{'shape':>12}{'spikes':>8}{'rank %':>8}{'signal E%':>11}{'KS':>7}{'eff.rank':>10}"
    rows = [header, "-" * len(header)]
    for name, r in reports.items():
        shape = f"{r.shape[0]}x{r.shape[1]}"
        rows.append(
            f"{name:<24}{shape:>12}{r.n_spikes:>8d}{100 * r.rank_fraction:>8.1f}"
            f"{100 * r.signal_energy:>11.1f}{r.ks_distance:>7.3f}{r.effective_rank:>10.1f}"
        )
    return "\n".join(rows)


def _ks_distance(bulk: np.ndarray, q: float, sigma2: float) -> float:
    """Kolmogorov-Smirnov distance between the bulk eigenvalues and the fitted MP law."""
    if bulk.size == 0 or sigma2 <= 0.0:
        return float("nan")
    x = np.sort(bulk)
    model = mp_cdf(x, q, sigma2)
    k = np.arange(1, x.size + 1)
    return float(max(np.max(k / x.size - model), np.max(model - (k - 1) / x.size)))
