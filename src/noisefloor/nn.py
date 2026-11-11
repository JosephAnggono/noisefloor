"""A small fully-connected network in NumPy, used to produce weights to analyse.

Kept dependency-free on purpose: the experiments run on a laptop CPU in minutes and
every gradient is checked against finite differences in the test suite.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

Loss = Literal["cross_entropy", "mse"]


class MLP:
    """ReLU multilayer perceptron. Weights have shape (fan_out, fan_in)."""

    def __init__(self, sizes: list[int], init_scale: float = 1.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.W = [
            init_scale * rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_out, fan_in))
            for fan_in, fan_out in zip(sizes[:-1], sizes[1:])
        ]
        self.b = [np.zeros(fan_out) for fan_out in sizes[1:]]

    # ------------------------------------------------------------------ forward
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Return the output and the list of layer inputs (needed for backprop)."""
        acts = [x]
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            x = x @ w.T + b
            if i < len(self.W) - 1:
                x = np.maximum(x, 0.0)
            acts.append(x)
        return x, acts

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0]

    # ----------------------------------------------------------------- backward
    def loss_and_grads(
        self, x: np.ndarray, y: np.ndarray, loss: Loss
    ) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        out, acts = self.forward(x)
        m = x.shape[0]
        if loss == "cross_entropy":
            z = out - out.max(axis=1, keepdims=True)
            log_prob = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            value = -log_prob[np.arange(m), y].mean()
            delta = np.exp(log_prob)
            delta[np.arange(m), y] -= 1.0
            delta /= m
        elif loss == "mse":
            diff = out - y.reshape(out.shape)
            value = 0.5 * np.mean(np.sum(diff**2, axis=1))
            delta = diff / m
        else:
            raise ValueError(f"unknown loss {loss!r}")

        grads_w: list[np.ndarray] = [np.empty(0)] * len(self.W)
        grads_b: list[np.ndarray] = [np.empty(0)] * len(self.W)
        for i in reversed(range(len(self.W))):
            grads_w[i] = delta.T @ acts[i]
            grads_b[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i]) * (acts[i] > 0.0)
        return float(value), grads_w, grads_b

    # ------------------------------------------------------------------ helpers
    def weights(self) -> dict[str, np.ndarray]:
        return {f"layer{i + 1}": w for i, w in enumerate(self.W)}

    def copy(self) -> "MLP":
        clone = MLP(self.sizes)
        clone.W = [w.copy() for w in self.W]
        clone.b = [b.copy() for b in self.b]
        return clone


def train(
    model: MLP,
    x: np.ndarray,
    y: np.ndarray,
    loss: Loss,
    epochs: int,
    lr: float = 1e-3,
    batch_size: int = 128,
    weight_decay: float = 0.0,
    seed: int = 0,
    callback: Callable[[int, MLP], None] | None = None,
) -> MLP:
    """Train in place with Adam (decoupled weight decay) and return the model.

    ``callback(epoch, model)`` is called before training (epoch 0) and after each epoch.
    """
    rng = np.random.default_rng(seed)
    params = model.W + model.b
    m1 = [np.zeros_like(p) for p in params]
    m2 = [np.zeros_like(p) for p in params]
    beta1, beta2, eps, step = 0.9, 0.999, 1e-8, 0
    n_weights = len(model.W)

    if callback is not None:
        callback(0, model)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(x.shape[0])
        for start in range(0, x.shape[0], batch_size):
            idx = order[start : start + batch_size]
            _, gw, gb = model.loss_and_grads(x[idx], y[idx], loss)
            step += 1
            for j, (p, g) in enumerate(zip(params, gw + gb)):
                m1[j] = beta1 * m1[j] + (1.0 - beta1) * g
                m2[j] = beta2 * m2[j] + (1.0 - beta2) * g * g
                m_hat = m1[j] / (1.0 - beta1**step)
                v_hat = m2[j] / (1.0 - beta2**step)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)
                if weight_decay and j < n_weights:
                    p -= lr * weight_decay * p
        if callback is not None:
            callback(epoch, model)
    return model


def accuracy(model: MLP, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(model.predict(x).argmax(axis=1) == y))


def r2_score(model: MLP, x: np.ndarray, y: np.ndarray) -> float:
    pred = model.predict(x).ravel()
    return float(1.0 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2))
