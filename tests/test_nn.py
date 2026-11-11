"""The NumPy network: gradients are exact and training works."""

import numpy as np
import pytest

from noisefloor.data import digits_task, planted_subspace_task, subspace_overlap
from noisefloor.nn import MLP, accuracy, r2_score, train


@pytest.mark.parametrize("loss,out_dim", [("cross_entropy", 4), ("mse", 1)])
def test_gradients_match_finite_differences(loss, out_dim):
    rng = np.random.default_rng(0)
    model = MLP([6, 9, 7, out_dim], seed=1)
    model.b = [rng.normal(scale=0.1, size=b.shape) for b in model.b]
    x = rng.normal(size=(11, 6))
    y = rng.integers(0, out_dim, size=11) if loss == "cross_entropy" else rng.normal(size=11)
    _, grads_w, grads_b = model.loss_and_grads(x, y, loss)

    step = 1e-6
    for params, grads in ((model.W, grads_w), (model.b, grads_b)):
        for param, grad in zip(params, grads):
            for index in [tuple(rng.integers(0, s) for s in param.shape) for _ in range(6)]:
                original = param[index]
                param[index] = original + step
                up = model.loss_and_grads(x, y, loss)[0]
                param[index] = original - step
                down = model.loss_and_grads(x, y, loss)[0]
                param[index] = original
                assert grad[index] == pytest.approx((up - down) / (2 * step), rel=1e-4, abs=1e-8)


def test_training_learns_digits_and_is_reproducible():
    split = digits_task(seed=0)

    def fit():
        model = MLP([64, 64, 10], seed=0)
        return train(model, split.x_train, split.y_train, "cross_entropy", epochs=40, seed=0)

    first, second = fit(), fit()
    assert accuracy(first, split.x_test, split.y_test) > 0.93
    np.testing.assert_array_equal(first.W[0], second.W[0])


def test_weight_decay_shrinks_weights_and_copy_is_independent():
    split, _ = planted_subspace_task(400, 100, dim=20, rank=2, seed=0)
    kwargs = dict(loss="mse", epochs=30, lr=1e-2, seed=0)
    plain = train(MLP([20, 32, 1], seed=0), split.x_train, split.y_train, **kwargs)
    decayed = train(MLP([20, 32, 1], seed=0), split.x_train, split.y_train, weight_decay=0.5, **kwargs)
    assert np.linalg.norm(decayed.W[0]) < np.linalg.norm(plain.W[0])
    assert r2_score(plain, split.x_train, split.y_train) > 0.9
    clone = plain.copy()
    clone.W[0][:] = 0.0
    assert plain.W[0].any()
    assert list(plain.weights()) == ["layer1", "layer2"]


def test_planted_task_and_overlap():
    split, directions = planted_subspace_task(200, 50, dim=30, rank=4, seed=0)
    assert split.x_train.shape == (200, 30) and split.y_test.shape == (50,)
    np.testing.assert_allclose(directions @ directions.T, np.eye(4), atol=1e-10)
    assert split.y_train.std() == pytest.approx(1.0)
    assert subspace_overlap(directions, directions) == pytest.approx(1.0)
    assert subspace_overlap(directions[:2], directions) == pytest.approx(0.5)
    assert subspace_overlap(np.empty((0, 30)), directions) == 0.0


def test_label_noise_only_touches_training_labels():
    clean, noisy = digits_task(0.0, seed=3), digits_task(0.3, seed=3)
    np.testing.assert_array_equal(clean.y_test, noisy.y_test)
    np.testing.assert_array_equal(clean.x_train, noisy.x_train)
    assert np.mean(clean.y_train != noisy.y_train) == pytest.approx(0.3, abs=0.05)
