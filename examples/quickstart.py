"""Quick start: train two small networks and read their weights with noisefloor.

Run with ``python examples/quickstart.py`` (about 15 seconds on a laptop CPU).
"""

from __future__ import annotations

import noisefloor as nf
from noisefloor.data import planted_subspace_task
from noisefloor.nn import MLP, r2_score, train

# A regression task whose answer depends on only 5 of the 256 input directions.
split, _ = planted_subspace_task(n_train=6000, n_test=2000, dim=256, rank=5, seed=0)

for name, init_scale in (("standard init", 1.0), ("small init", 0.3)):
    model = MLP([256, 512, 1], init_scale=init_scale, seed=1)
    train(model, split.x_train, split.y_train, "mse", epochs=30, seed=1)

    print(f"\n{name}: test R^2 = {r2_score(model, split.x_test, split.y_test):.3f}")
    print(nf.format_reports(nf.analyze_model(model.weights())))

    # Keep only what rises above the noise floor in the first layer, then re-test.
    model.W[0], report = nf.denoise(model.W[0])
    print(f"first layer cut to rank {report.n_spikes}: test R^2 = "
          f"{r2_score(model, split.x_test, split.y_test):.3f}")
