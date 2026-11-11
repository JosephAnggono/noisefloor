"""Datasets for the experiments: a planted low-rank task and handwritten digits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def planted_subspace_task(
    n_train: int, n_test: int, dim: int, rank: int, seed: int = 0
) -> tuple[Split, np.ndarray]:
    """Regression task whose target depends on only ``rank`` directions of the input.

    ``y = sum_k a_k * relu(<u_k, x>)`` with orthonormal ``u_k`` and Gaussian inputs
    (a "multi-index model"). Returns the data and the ``(rank, dim)`` matrix of true
    directions, so we can check whether a trained network found them.
    """
    rng = np.random.default_rng(seed)
    directions = np.linalg.qr(rng.normal(size=(dim, rank)))[0].T
    signs = np.where(np.arange(rank) % 2 == 0, 1.0, -1.0)

    def sample(n: int) -> tuple[np.ndarray, np.ndarray]:
        x = rng.normal(size=(n, dim))
        y = np.maximum(x @ directions.T, 0.0) @ signs
        return x, y

    x_train, y_train = sample(n_train)
    x_test, y_test = sample(n_test)
    mean, std = y_train.mean(), y_train.std()
    split = Split(x_train, (y_train - mean) / std, x_test, (y_test - mean) / std)
    return split, directions


def planted_matrix(
    shape: tuple[int, int], rank: int, strength: float, rng: np.random.Generator, sigma: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Random rank-``rank`` signal plus i.i.d. noise of standard deviation ``sigma``.

    ``strength`` is measured in units of the detection limit: every signal singular
    value equals ``strength * sigma * (n * p) ** 0.25``, and 1.0 is the Baik-Ben
    Arous-Peche threshold below which no method can see the signal.
    Returns ``(signal + noise, signal)``.
    """
    n, p = max(shape), min(shape)
    u = np.linalg.qr(rng.normal(size=(shape[0], rank)))[0]
    v = np.linalg.qr(rng.normal(size=(shape[1], rank)))[0]
    signal = strength * sigma * (n * p) ** 0.25 * u @ v.T
    return signal + sigma * rng.normal(size=shape), signal


def digits_task(label_noise: float = 0.0, test_fraction: float = 0.33, seed: int = 0) -> Split:
    """8x8 handwritten digits (scikit-learn), optionally with corrupted training labels.

    A fraction ``label_noise`` of the *training* labels is replaced by a different,
    uniformly random class. Test labels are always clean.
    """
    from sklearn.datasets import load_digits

    rng = np.random.default_rng(seed)
    x, y = load_digits(return_X_y=True)
    x = x / 16.0 - 0.5
    order = rng.permutation(len(y))
    n_test = int(round(test_fraction * len(y)))
    test, train = order[:n_test], order[n_test:]
    y_train = y[train].copy()
    flip = rng.random(len(train)) < label_noise
    y_train[flip] = (y_train[flip] + rng.integers(1, 10, size=int(flip.sum()))) % 10
    return Split(x[train], y_train, x[test], y[test])


def subspace_overlap(found: np.ndarray, truth: np.ndarray) -> float:
    """Share of the true subspace captured by the found one (1 = fully recovered).

    Both arguments hold orthonormal rows. Returns ``||truth @ found.T||_F^2 / rank(truth)``.
    """
    if found.size == 0:
        return 0.0
    return float(np.sum((truth @ found.T) ** 2) / truth.shape[0])
