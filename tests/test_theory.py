"""The closed-form laws agree with known values and with simulation."""

import numpy as np
import pytest

from noisefloor import (edge_threshold, johnstone_centre_scale, mp_cdf, mp_edges, mp_median, mp_pdf,
                        mp_quantile, tw1_cdf, tw1_pdf, tw1_ppf)


@pytest.mark.parametrize("q", [0.1, 0.5, 1.0])
def test_mp_is_a_probability_law_with_mean_sigma2(q):
    sigma2 = 2.5
    lo, hi = mp_edges(q, sigma2)
    assert mp_cdf(lo, q, sigma2) == pytest.approx(0.0, abs=1e-9)
    assert mp_cdf(hi, q, sigma2) == pytest.approx(1.0, abs=1e-9)
    # mean of the law equals the entry variance: E[x] = integral of (1 - CDF)
    xs = np.linspace(0.0, hi, 200_001)
    assert np.trapezoid(1.0 - mp_cdf(xs, q, sigma2), xs) == pytest.approx(sigma2, rel=1e-4)


def test_mp_pdf_is_the_derivative_of_the_cdf():
    q = 0.4
    lo, hi = mp_edges(q)
    xs = np.linspace(lo + 0.05, hi - 0.05, 50)
    step = 1e-4
    slope = (mp_cdf(xs + step, q) - mp_cdf(xs - step, q)) / (2 * step)
    np.testing.assert_allclose(slope, mp_pdf(xs, q), rtol=2e-3)


def test_mp_quantile_inverts_cdf():
    probs = np.linspace(0.01, 0.99, 25)
    np.testing.assert_allclose(mp_cdf(mp_quantile(probs, 0.3, 4.0), 0.3, 4.0), probs, atol=1e-6)
    assert mp_cdf(mp_median(0.3), 0.3) == pytest.approx(0.5, abs=1e-6)


def test_mp_matches_a_simulated_noise_matrix():
    rng = np.random.default_rng(0)
    n, p = 1200, 600
    eig = np.sort(np.linalg.svd(rng.normal(size=(n, p)), compute_uv=False) ** 2 / n)
    ks = np.max(np.abs(mp_cdf(eig, p / n) - np.arange(1, p + 1) / p))
    assert ks < 0.01


@pytest.mark.parametrize("q", [0.0, -0.2, 1.5])
def test_mp_rejects_invalid_aspect_ratio(q):
    with pytest.raises(ValueError):
        mp_edges(q)


def test_tracy_widom_matches_published_values():
    # Tabulated TW1 values: mean -1.2065, variance 1.6078, quantiles 0.9793 (95%), 2.0234 (99%).
    xs = np.linspace(-12, 12, 200_001)
    pdf = tw1_pdf(xs)
    mean = np.trapezoid(xs * pdf, xs)
    assert np.trapezoid(pdf, xs) == pytest.approx(1.0, abs=1e-6)
    assert mean == pytest.approx(-1.2065, abs=2e-3)
    assert np.trapezoid((xs - mean) ** 2 * pdf, xs) == pytest.approx(1.6078, abs=2e-3)
    assert tw1_ppf(0.95) == pytest.approx(0.9793, abs=0.02)
    assert tw1_ppf(0.99) == pytest.approx(2.0234, abs=0.02)
    assert tw1_cdf(tw1_ppf(0.7)) == pytest.approx(0.7)


def test_largest_noise_eigenvalue_follows_tracy_widom():
    rng = np.random.default_rng(1)
    n, p, trials = 120, 60, 800
    centre, scale = johnstone_centre_scale(n, p)
    top = np.array([np.linalg.svd(rng.normal(size=(n, p)), compute_uv=False)[0] ** 2 for _ in range(trials)])
    exceed = np.mean((top - centre) / scale > tw1_ppf(0.9))
    assert exceed == pytest.approx(0.10, abs=0.035)  # 3+ standard errors of slack


def test_edge_threshold_sits_just_above_the_mp_edge_and_scales_with_variance():
    n, p = 800, 400
    edge = mp_edges(p / n)[1]
    threshold = edge_threshold(n, p)
    assert edge < threshold < 1.05 * edge
    assert edge_threshold(n, p, sigma2=3.0) == pytest.approx(3.0 * threshold)
    assert edge_threshold(n, p, alpha=0.001) > threshold
    with pytest.raises(ValueError):
        edge_threshold(n, p, alpha=0.0)
