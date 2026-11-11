"""Shared paths, plot style and helpers for the experiment scripts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from noisefloor.data import Split, planted_subspace_task
from noisefloor.nn import MLP, train

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

# Colour follows the entity in every figure:
BLUE = "#2a78d6"  # measurements / the small-init ("rich") network
ORANGE = "#eb6834"  # the standard-init ("lazy") network
INK = "#0b0b0b"  # theory curves and thresholds
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE_FILL = "#b7d3f6"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "text.color": INK,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def save(fig, name: str) -> None:
    fig.savefig(RESULTS / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote results/{name}")


def record(section: str, values: dict) -> None:
    """Merge one experiment's numbers into results/results.json."""
    path = RESULTS / "results.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[section] = values
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------- planted task
DIM, WIDTH, RANK = 256, 512, 5
N_TRAIN, N_TEST = 6000, 2000
RICH_INIT, LAZY_INIT = 0.3, 1.0  # multiples of the standard He initialisation


def planted_data(seed: int = 0) -> tuple[Split, np.ndarray]:
    return planted_subspace_task(N_TRAIN, N_TEST, DIM, RANK, seed=seed)


def train_planted(split: Split, init_scale: float, seed: int, epochs: int, callback=None) -> MLP:
    """One-hidden-layer network on the planted task; only the initial scale varies."""
    model = MLP([DIM, WIDTH, 1], init_scale=init_scale, seed=seed)
    return train(
        model, split.x_train, split.y_train, "mse", epochs=epochs, lr=1e-3, batch_size=128,
        seed=seed, callback=callback,
    )
