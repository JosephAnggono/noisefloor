"""Experiment 3 - can the noise floor choose a compression rank without any data?

(a, b) Sweep the initial weight scale. The share of weight energy above the noise
       floor rises as the network moves from the lazy to the rich regime, and test
       accuracy rises with it.
(c)    Truncate the trained first layer to rank k and measure test accuracy for every
       k. The rank picked automatically by the noise floor (no validation data, no
       hyper-parameter) lands on the plateau of that curve.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from common import (AXIS, BLUE, DIM, INK, LAZY_INIT, MUTED, RANK, RICH_INIT, WIDTH, apply_style,
                    planted_data, record, save, train_planted)

from noisefloor import analyze, denoise, low_rank_params, truncate
from noisefloor.nn import r2_score

EPOCHS = 40
SCALES = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
SEEDS = [1, 2, 3]


def truncated_r2(model, split, rank: int) -> float:
    clone = model.copy()
    clone.W[0] = truncate(model.W[0], rank)
    return r2_score(clone, split.x_test, split.y_test)


def run() -> dict:
    apply_style()
    split, _ = planted_data(seed=0)

    sweep = {"scale": SCALES, "test_r2": [], "signal_energy": [], "spikes": []}
    rich_models = []
    for scale in SCALES:
        r2, energy, spikes = [], [], []
        for seed in SEEDS:
            model = train_planted(split, scale, seed=seed, epochs=EPOCHS)
            report = analyze(model.W[0])
            r2.append(r2_score(model, split.x_test, split.y_test))
            energy.append(report.signal_energy)
            spikes.append(report.n_spikes)
            if scale == RICH_INIT:
                rich_models.append(model)
        sweep["test_r2"].append(r2)
        sweep["signal_energy"].append(energy)
        sweep["spikes"].append(spikes)

    # Truncation curve and automatic rank for the rich networks --------------------
    ranks = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    curve = np.array([[truncated_r2(m, split, k) for k in ranks] for m in rich_models])
    rows = []
    for model in rich_models:
        hard, report = denoise(model.W[0], "hard")
        shrunk, _ = denoise(model.W[0], "shrink")
        clone_h, clone_s = model.copy(), model.copy()
        clone_h.W[0], clone_s.W[0] = hard, shrunk
        rows.append({
            "full_r2": r2_score(model, split.x_test, split.y_test),
            "hard_r2": r2_score(clone_h, split.x_test, split.y_test),
            "shrink_r2": r2_score(clone_s, split.x_test, split.y_test),
            "rank": report.n_spikes,
            "params_full": WIDTH * DIM,
            "params_low_rank": low_rank_params((WIDTH, DIM), report.n_spikes),
        })

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    # Figure ----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), constrained_layout=True)
    r2 = np.array(sweep["test_r2"])
    energy = 100 * np.array(sweep["signal_energy"])

    for ax, values, title, ylabel in (
        (axes[0], r2, "(a) Smaller initial weights generalise better", "test R²"),
        (axes[1], energy, "(b) ...and put more weight above the floor", "weight energy above noise floor (%)"),
    ):
        ax.errorbar(SCALES, values.mean(axis=1), yerr=values.std(axis=1), fmt="-o", color=BLUE,
                    markeredgecolor="white", markeredgewidth=0.8, capsize=2.5, elinewidth=1.2)
        ax.set(xscale="log", title=title, xlabel="initial weight scale (1 = standard)", ylabel=ylabel)
        ax.set_xticks([0.1, 0.3, 1.0], labels=["0.1", "0.3", "1"])
        ax.minorticks_off()
    axes[0].set_ylim(-0.35, 1)
    axes[0].axhline(0.0, color=AXIS, linewidth=0.8)
    axes[1].set_ylim(0, 100)

    ax = axes[2]
    ax.plot(ranks, curve.mean(axis=0), "-o", color=BLUE, markeredgecolor="white", markeredgewidth=0.8,
            label="first layer truncated to rank k")
    ax.axhline(mean("full_r2"), color=INK, linewidth=1.2, label="full network (rank 256)")
    auto_rank = mean("rank")
    ax.axvline(auto_rank, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(auto_rank * 1.12, 0.42, f"rank chosen by the\nnoise floor ({auto_rank:.0f})", color=MUTED,
            fontsize=8.5, va="bottom")
    ax.set(xscale="log", title="(c) The noise floor picks a safe rank", xlabel="rank kept (k)",
           ylabel="test R²", ylim=(0, 1))
    ax.set_xticks([1, 4, 16, 64, 256], labels=["1", "4", "16", "64", "256"])
    ax.minorticks_off()
    ax.legend(loc="lower right")
    save(fig, "fig3_compression.png")

    out = {
        "sweep": {
            f"{scale:g}": {"test_r2_mean": float(r2[i].mean()), "test_r2_std": float(r2[i].std()),
                           "signal_energy_mean": float(energy[i].mean() / 100),
                           "spikes_mean": float(np.mean(sweep["spikes"][i]))}
            for i, scale in enumerate(SCALES)
        },
        "rich_network_compression": {
            "seeds": len(rows), "init_scale": RICH_INIT, "planted_rank": RANK,
            "rank_chosen_mean": auto_rank, "ranks_chosen": [row["rank"] for row in rows],
            "test_r2_full": mean("full_r2"), "test_r2_truncated": mean("hard_r2"),
            "test_r2_full_per_seed": [row["full_r2"] for row in rows],
            "test_r2_truncated_per_seed": [row["hard_r2"] for row in rows],
            "test_r2_shrunk": mean("shrink_r2"),
            "test_r2_at_planted_rank": float(curve[:, ranks.index(RANK)].mean()),
            "first_layer_params_full": WIDTH * DIM, "first_layer_params_low_rank": mean("params_low_rank"),
            "compression_ratio": WIDTH * DIM / mean("params_low_rank"),
        },
        "lazy_reference_scale": LAZY_INIT,
    }
    record("exp3_compression", out)
    return out


if __name__ == "__main__":
    print(run())
