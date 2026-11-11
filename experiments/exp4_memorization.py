"""Experiment 4 - does memorisation leave a mark on the spectrum? (real data)

Train the same network on handwritten digits while corrupting a growing share of the
training labels. Every run reaches ~100% training accuracy, so the training loss says
nothing about which runs memorised. The weights do: the number of singular values above
the noise floor grows with the label noise and tracks the generalisation gap.

Uses the standard initialisation and no tuning. The statistic needs no data at all.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from common import BLUE, apply_style, record, save
from scipy.stats import spearmanr

from noisefloor import analyze, denoise
from noisefloor.data import digits_task
from noisefloor.nn import MLP, accuracy, train

NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SEEDS = [0, 1, 2, 3]
SIZES = [64, 256, 256, 10]
EPOCHS = 120


def fit(label_noise: float, seed: int, init_scale: float = 1.0):
    split = digits_task(label_noise=label_noise, seed=seed)
    model = MLP(SIZES, init_scale=init_scale, seed=seed)
    train(model, split.x_train, split.y_train, "cross_entropy", epochs=EPOCHS, lr=1e-3, batch_size=64, seed=seed)
    return model, split


def run() -> dict:
    apply_style()
    runs = []
    for noise in NOISE_LEVELS:
        for seed in SEEDS:
            model, split = fit(noise, seed)
            report = analyze(model.W[1])
            train_acc = accuracy(model, split.x_train, split.y_train)
            test_acc = accuracy(model, split.x_test, split.y_test)
            runs.append({"label_noise": noise, "seed": seed, "train_acc": train_acc, "test_acc": test_acc,
                         "gap": train_acc - test_acc, "spikes": report.n_spikes,
                         "signal_energy": report.signal_energy})

    spikes = np.array([r["spikes"] for r in runs]).reshape(len(NOISE_LEVELS), len(SEEDS))
    gaps = np.array([r["gap"] for r in runs]).reshape(len(NOISE_LEVELS), len(SEEDS))
    rho, p_value = spearmanr(spikes.ravel(), gaps.ravel())
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7), constrained_layout=True)

    ax = axes[0]
    ax.errorbar(100 * np.array(NOISE_LEVELS), spikes.mean(axis=1), yerr=spikes.std(axis=1), fmt="-o", color=BLUE,
                markeredgecolor="white", markeredgewidth=0.8, capsize=2.5, elinewidth=1.2)
    ax.set(title="(a) Memorising noise costs rank", xlabel="training labels corrupted (%)",
           ylabel="singular values above the noise floor", ylim=(0, None))

    ax = axes[1]
    ax.plot(spikes.ravel(), 100 * gaps.ravel(), "o", color=BLUE, markeredgecolor="white", markeredgewidth=0.8)
    ax.text(0.04, 0.93, f"Spearman ρ = {rho:.2f}  ({spikes.size} runs)", transform=ax.transAxes, va="top")
    ax.set(title="(b) ...which tracks the generalisation gap", xlabel="singular values above the noise floor",
           ylabel="train accuracy − test accuracy (points)")
    save(fig, "fig4_memorization.png")

    # Side result: on clean labels, how much accuracy survives removing the bulk? ----
    survive = {}
    for name, scale in (("standard_init", 1.0), ("small_init", 0.3)):
        full, cut, kept = [], [], []
        for seed in SEEDS:
            model, split = fit(0.0, seed, init_scale=scale)
            clone = model.copy()
            reports = []
            for layer in (0, 1):
                clone.W[layer], rep = denoise(model.W[layer], "hard")
                reports.append(rep)
            full.append(accuracy(model, split.x_test, split.y_test))
            cut.append(accuracy(clone, split.x_test, split.y_test))
            kept.append([rep.n_spikes for rep in reports])
        survive[name] = {"init_scale": scale, "test_acc_full": float(np.mean(full)),
                         "test_acc_bulk_removed": float(np.mean(cut)),
                         "mean_rank_kept_layer1_layer2": np.mean(kept, axis=0).tolist()}

    out = {
        "runs": len(runs), "epochs": EPOCHS, "min_train_acc": float(min(r["train_acc"] for r in runs)),
        "spearman_spikes_vs_gap": float(rho), "spearman_p_value": float(p_value),
        "by_label_noise": {
            f"{noise:g}": {"spikes_mean": float(spikes[i].mean()), "spikes_std": float(spikes[i].std()),
                           "gap_mean": float(gaps[i].mean()),
                           "test_acc_mean": float(np.mean([r["test_acc"] for r in runs if r["label_noise"] == noise]))}
            for i, noise in enumerate(NOISE_LEVELS)
        },
        "removing_the_bulk_clean_labels": survive,
    }
    record("exp4_memorization", out)
    return out


if __name__ == "__main__":
    print(run())
