"""Spike detection: calibrated on noise, exact on planted signals."""

import numpy as np
import pytest

from noisefloor import analyze, analyze_model, format_reports, truncate
from noisefloor.data import planted_matrix as planted
from noisefloor.spectrum import as_matrix


def test_pure_noise_has_no_spikes_and_the_right_variance():
    rng = np.random.default_rng(0)
    report = analyze(0.2 * rng.normal(size=(512, 256)))
    assert report.n_spikes == 0
    assert report.sigma2 == pytest.approx(0.04, rel=0.03)
    assert report.ks_distance < 0.03
    assert report.signal_energy == 0.0
    assert report.bulk_edge < report.threshold


def test_false_alarm_rate_is_close_to_alpha():
    rng = np.random.default_rng(1)
    trials = 400
    false_alarms = sum(analyze(rng.normal(size=(128, 64))).n_spikes > 0 for _ in range(trials))
    assert false_alarms / trials < 0.04  # nominal 0.01; the estimated variance costs a little


@pytest.mark.parametrize("shape,rank", [((400, 200), 1), ((512, 256), 5), ((256, 256), 20), ((100, 300), 3)])
def test_planted_rank_is_recovered(shape, rank):
    rng = np.random.default_rng(2)
    matrix, _ = planted(shape, rank, strength=3.0, rng=rng, sigma=0.5)
    report = analyze(matrix)
    assert report.n_spikes == rank
    assert report.sigma2 == pytest.approx(0.25, rel=0.05)
    assert report.rank_fraction == pytest.approx(rank / min(shape))


def test_signal_below_the_detection_limit_stays_hidden():
    rng = np.random.default_rng(3)
    matrix, _ = planted((512, 256), 1, strength=0.5, rng=rng)
    assert analyze(matrix).n_spikes == 0


def test_invariant_to_scale_and_transpose():
    rng = np.random.default_rng(4)
    matrix, _ = planted((300, 150), 4, strength=2.5, rng=rng)
    base = analyze(matrix)
    scaled = analyze(7.0 * matrix)
    flipped = analyze(matrix.T)
    assert scaled.n_spikes == flipped.n_spikes == base.n_spikes == 4
    assert scaled.sigma2 == pytest.approx(49.0 * base.sigma2)
    assert flipped.sigma2 == pytest.approx(base.sigma2)
    assert scaled.stable_rank == pytest.approx(base.stable_rank)


def test_known_variance_can_be_supplied():
    rng = np.random.default_rng(5)
    matrix, _ = planted((300, 150), 2, strength=3.0, rng=rng)
    assert analyze(matrix, sigma2=1.0).n_spikes == 2
    assert analyze(matrix, sigma2=100.0).n_spikes == 0


def test_exactly_low_rank_input_reports_its_rank():
    rng = np.random.default_rng(6)
    report = analyze(truncate(rng.normal(size=(120, 80)), 7))
    assert report.n_spikes == 7
    assert report.sigma2 == 0.0
    assert report.signal_energy == pytest.approx(1.0)


def test_convolution_kernels_are_flattened():
    rng = np.random.default_rng(7)
    kernel = rng.normal(size=(64, 16, 3, 3))
    assert as_matrix(kernel).shape == (64, 144)
    assert analyze(kernel).shape == (64, 144)
    with pytest.raises(ValueError):
        as_matrix(np.zeros(5))


def test_rank_summaries():
    identity = analyze(np.eye(40), sigma2=1e-6)
    assert identity.stable_rank == pytest.approx(40.0)
    assert identity.effective_rank == pytest.approx(40.0)
    rank_one = analyze(np.outer(np.ones(40), np.ones(30)), sigma2=1e-6)
    assert rank_one.stable_rank == pytest.approx(1.0)
    assert rank_one.effective_rank == pytest.approx(1.0, abs=1e-6)


class FakeTensor:
    """Stands in for torch.Tensor so the test suite does not need PyTorch."""

    def __init__(self, array):
        self._array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


def test_analyze_model_accepts_a_state_dict_and_skips_non_matrices():
    rng = np.random.default_rng(8)
    state = {
        "fc1.weight": FakeTensor(rng.normal(size=(64, 32))),
        "fc1.bias": FakeTensor(np.zeros(64)),
        "head.weight": rng.normal(size=(4, 64)),  # too small to have a bulk
        "conv.weight": rng.normal(size=(32, 8, 3, 3)),
    }
    reports = analyze_model(state)
    assert list(reports) == ["fc1.weight", "conv.weight"]
    table = format_reports(reports)
    assert "fc1.weight" in table and "32x72" in table
