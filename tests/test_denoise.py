"""Truncation and shrinkage recover a planted low-rank matrix."""

import numpy as np
import pytest

from noisefloor import denoise, low_rank_params, shrink_singular_values, truncate
from noisefloor.cli import main
from noisefloor.data import planted_matrix as planted


def test_truncate_is_the_best_low_rank_approximation():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(60, 40))
    s = np.linalg.svd(matrix, compute_uv=False)
    approx = truncate(matrix, 6)
    assert np.linalg.matrix_rank(approx) == 6
    assert np.linalg.norm(matrix - approx) == pytest.approx(np.sqrt(np.sum(s[6:] ** 2)))
    np.testing.assert_allclose(truncate(matrix, 999), matrix, atol=1e-10)
    assert not truncate(matrix, 0).any()


def test_denoising_moves_the_matrix_closer_to_the_truth():
    rng = np.random.default_rng(1)
    errors = {"noisy": [], "zero": [], "hard": [], "shrink": []}
    for _ in range(10):
        matrix, signal = planted((400, 200), 5, strength=1.6, rng=rng)
        errors["noisy"].append(np.linalg.norm(matrix - signal))
        errors["zero"].append(np.linalg.norm(signal))
        for method in ("hard", "shrink"):
            cleaned, report = denoise(matrix, method)
            assert report.n_spikes == 5
            errors[method].append(np.linalg.norm(cleaned - signal))
    mean = {name: float(np.mean(values)) for name, values in errors.items()}
    assert mean["hard"] < 0.3 * mean["noisy"]
    # Gavish & Donoho (2017): shrinkage beats hard thresholding. For a signal this weak,
    # hard thresholding is even worse than returning zero; shrinkage is not.
    assert mean["shrink"] < mean["hard"]
    assert mean["shrink"] < mean["zero"] < mean["hard"]


def test_shrinker_properties():
    n, q = 400, 0.5
    edge = np.sqrt(n) * (1 + np.sqrt(q))
    s = np.array([0.5, 0.99, 1.2, 2.0, 5.0, 50.0]) * edge
    out = shrink_singular_values(s, n, q, sigma2=1.0)
    assert np.all(out[:2] == 0.0)  # inside the bulk
    assert np.all(out[2:] > 0.0) and np.all(out[2:] < s[2:])  # shrunk, never inflated
    assert np.all(np.diff(out) >= 0.0)
    assert out[-1] == pytest.approx(s[-1], rel=1e-3)  # a huge signal is left almost untouched
    np.testing.assert_allclose(shrink_singular_values(3.0 * s, n, q, sigma2=9.0), 3.0 * out)


def test_denoise_keeps_shape_and_validates_method():
    rng = np.random.default_rng(2)
    kernel = rng.normal(size=(32, 8, 3, 3))
    cleaned, _ = denoise(kernel)
    assert cleaned.shape == kernel.shape
    with pytest.raises(ValueError):
        denoise(kernel, "soft")
    assert low_rank_params((512, 256), 12) == 12 * 768


def test_command_line_round_trip(tmp_path, capsys):
    rng = np.random.default_rng(3)
    matrix, _ = planted((128, 64), 3, strength=3.0, rng=rng)
    source, target = tmp_path / "weights.npz", tmp_path / "clean.npz"
    np.savez(source, layer=matrix, bias=np.zeros(128))
    assert main([str(source), "--denoise", str(target)]) == 0
    assert "layer" in capsys.readouterr().out
    with np.load(target) as cleaned:
        assert np.linalg.matrix_rank(cleaned["layer"]) == 3
        assert cleaned["bias"].shape == (128,)
    np.save(tmp_path / "vector.npy", np.zeros(5))
    assert main([str(tmp_path / "vector.npy")]) == 1
