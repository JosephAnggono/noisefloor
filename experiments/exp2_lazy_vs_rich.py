"""Experiment 2 - where does a network store what it learned?

Two identical networks learn a task whose target depends on only 5 directions of a
256-dimensional input. The only difference is the scale of the initial weights.

  * Standard initialisation ("lazy"): fits the training set, yet its weight matrix
    stays statistically indistinguishable from noise.
  * Small initialisation ("rich"): a few singular values break out of the noise bulk,
    and their singular vectors are the planted directions.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from common import (BLUE, INK, LAZY_INIT, MUTED, ORANGE, RANK, RICH_INIT, apply_style,
                    planted_data, record, save, train_planted)

from noisefloor import analyze
from noisefloor.data import subspace_overlap
from noisefloor.nn import r2_score

EPOCHS = 60


def run() -> dict:
    apply_style()
    split, directions = planted_data(seed=0)
    runs = {"rich": (RICH_INIT, BLUE, f"small init ({RICH_INIT}x)"),
            "lazy": (LAZY_INIT, ORANGE, f"standard init ({LAZY_INIT:g}x)")}
    history: dict[str, dict[str, list]] = {}
    final = {}

    for name, (init_scale, _, _) in runs.items():
        log: dict[str, list] = {"epoch": [], "spikes": [], "test_r2": []}

        def callback(epoch, model, log=log):
            log["epoch"].append(epoch)
            log["spikes"].append(analyze(model.W[0]).n_spikes)
            log["test_r2"].append(r2_score(model, split.x_test, split.y_test))

        model = train_planted(split, init_scale, seed=1, epochs=EPOCHS, callback=callback)
        report = analyze(model.W[0])
        vt = np.linalg.svd(model.W[0], full_matrices=False)[2]
        history[name] = log
        final[name] = {
            "init_scale": init_scale,
            "train_r2": r2_score(model, split.x_train, split.y_train),
            "test_r2": log["test_r2"][-1],
            "spikes_detected": report.n_spikes,
            "spikes_at_epoch_10": log["spikes"][10],
            "ks_distance_to_noise": report.ks_distance,
            "signal_energy": report.signal_energy,
            "overlap_detected_directions": subspace_overlap(vt[: report.n_spikes], directions),
            "overlap_top5_directions": subspace_overlap(vt[:RANK], directions),
            "top_eigenvalue_over_threshold": float(report.eigenvalues[0] / report.threshold),
            "_relative_spectrum": (np.sqrt(report.eigenvalues / report.threshold)).tolist(),
        }

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), constrained_layout=True)

    ax = axes[0]
    for name, (_, colour, label) in runs.items():
        rel = np.array(final[name]["_relative_spectrum"])
        ax.plot(np.arange(1, rel.size + 1), rel, "o", color=colour, markersize=3.2,
                markeredgewidth=0, label=label)
    ax.axhline(1.0, color=INK, linewidth=1.2)
    ax.text(250, 1.08, "noise floor", color=INK, fontsize=8.5, ha="right", va="bottom")
    ax.set(xscale="log", title="(a) Singular values after training",
           xlabel="rank of singular value (largest first)", ylabel="singular value / noise floor")
    ax.set_xticks([1, 10, 100], labels=["1", "10", "100"])
    ax.legend(loc="upper right")

    ax = axes[1]
    for name, (_, colour, label) in runs.items():
        ax.plot(history[name]["epoch"], history[name]["spikes"], color=colour, label=label)
    ax.axhline(RANK, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(EPOCHS, RANK + 0.5, "directions in the task", color=MUTED, fontsize=8.5, ha="right")
    ax.set(title="(b) Signal directions detected", xlabel="epoch", ylabel="singular values above the noise floor")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper left")

    ax = axes[2]
    for name, (_, colour, label) in runs.items():
        ax.plot(history[name]["epoch"][1:], history[name]["test_r2"][1:], color=colour, label=label)
    ax.set(title="(c) Accuracy on unseen data", xlabel="epoch", ylabel="test R²", ylim=(0, 1))
    ax.legend(loc="lower right")
    save(fig, "fig2_lazy_vs_rich.png")

    for values in final.values():
        del values["_relative_spectrum"]
    record("exp2_lazy_vs_rich", final)
    return final


if __name__ == "__main__":
    print(run())
