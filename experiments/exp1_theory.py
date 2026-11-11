"""Experiment 1 - does the detector agree with random-matrix theory?

Checks, on synthetic matrices where the truth is known:
  (a) pure noise follows the Marchenko-Pastur law,
  (b) its largest eigenvalue follows the Tracy-Widom law,
  (c) a planted rank-one signal becomes visible exactly at the predicted strength
      (the Baik-Ben Arous-Peche transition), and
  (d) the spike test has the advertised false-alarm rate and near-full power above it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from common import BLUE, BLUE_FILL, INK, MUTED, apply_style, record, save

from noisefloor import analyze, johnstone_centre_scale, mp_edges, mp_pdf, tw1_pdf


def false_alarm_rate(shape: tuple[int, int], trials: int, rng) -> float:
    return float(np.mean([analyze(rng.normal(size=shape)).n_spikes > 0 for _ in range(trials)]))


def top_singular_value_theory(theta: np.ndarray, c: float) -> np.ndarray:
    """Benaych-Georges & Nadakuditi (2012): limit of the top singular value of
    ``theta * u v^T + noise`` when the noise bulk ends at ``1 + sqrt(c)``."""
    above = np.sqrt((1 + theta**2) * (c + theta**2)) / theta
    return np.where(theta > c**0.25, above, 1 + np.sqrt(c))


def run() -> dict:
    apply_style()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6), constrained_layout=True)

    # (a) Marchenko-Pastur bulk --------------------------------------------------
    n, p = 2000, 1000
    report = analyze(rng.normal(size=(n, p)))
    lo, hi = mp_edges(report.q)
    xs = np.linspace(lo, hi, 400)
    ax = axes[0, 0]
    ax.hist(report.eigenvalues, bins=60, density=True, color=BLUE_FILL, edgecolor="white", linewidth=0.4,
            label="eigenvalues of a 2000 x 1000 noise matrix")
    ax.plot(xs, mp_pdf(xs, report.q), color=INK, label="Marchenko-Pastur law")
    ax.set(title="(a) Pure noise has a known spectrum", xlabel="eigenvalue", ylabel="density")
    ax.set_ylim(0, 1.25)
    ax.legend(loc="upper right")

    # (b) Tracy-Widom fluctuations of the largest eigenvalue ----------------------
    n, p, trials = 200, 100, 3000
    centre, scale = johnstone_centre_scale(n, p)
    top = np.array([np.linalg.svd(rng.normal(size=(n, p)), compute_uv=False)[0] ** 2 for _ in range(trials)])
    z = (top - centre) / scale
    xs = np.linspace(-5, 4, 400)
    ax = axes[0, 1]
    ax.hist(z, bins=50, density=True, color=BLUE_FILL, edgecolor="white", linewidth=0.4,
            label=f"largest eigenvalue, {trials} noise matrices")
    ax.plot(xs, tw1_pdf(xs), color=INK, label="Tracy-Widom law")
    ax.set(title="(b) ...and a known largest eigenvalue", xlabel="largest eigenvalue (centred and scaled)",
           ylabel="density", xlim=(-5, 4))
    ax.set_ylim(0, 0.42)
    ax.legend(loc="upper right")

    # (c, d) planted rank-one signal ---------------------------------------------
    rows, cols, reps = 300, 600, 60
    c = rows / cols
    critical = c**0.25
    ratios = np.array([0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0])
    top_mean, power = [], []
    for ratio in ratios:
        tops, hits = [], 0
        for _ in range(reps):
            u = rng.normal(size=rows)
            v = rng.normal(size=cols)
            signal = ratio * critical * np.outer(u / np.linalg.norm(u), v / np.linalg.norm(v))
            x = signal + rng.normal(size=(rows, cols)) / np.sqrt(cols)
            tops.append(np.linalg.svd(x, compute_uv=False)[0])
            hits += analyze(x).n_spikes >= 1
        top_mean.append(np.mean(tops))
        power.append(hits / reps)
    grid = np.linspace(0.3, 3.0, 300)

    ax = axes[1, 0]
    ax.plot(grid, top_singular_value_theory(grid * critical, c), color=INK, label="theory")
    ax.plot(ratios, top_mean, "o", color=BLUE, markeredgecolor="white", markeredgewidth=0.8,
            label="measured (mean of 60 matrices)", zorder=3)
    ax.axvline(1.0, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(1.04, 2.9, "critical strength", color=MUTED, fontsize=8.5, va="top")
    ax.set(title="(c) A planted signal separates from the bulk...",
           xlabel="signal strength / critical strength", ylabel="largest singular value")
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    ax.plot(ratios, power, "-o", color=BLUE, markeredgecolor="white", markeredgewidth=0.8)
    ax.axvline(1.0, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(1.16, 0.06, "critical strength", color=MUTED, fontsize=8.5, va="center")
    ax.set(title="(d) ...and the test finds it once it does",
           xlabel="signal strength / critical strength", ylabel="detection rate", ylim=(-0.03, 1.05))
    save(fig, "fig1_theory.png")

    # False-alarm calibration on pure noise --------------------------------------
    calibration = {
        f"{a}x{b}": false_alarm_rate((a, b), trials, rng)
        for (a, b), trials in {(200, 100): 2000, (512, 256): 1000, (256, 256): 1000, (64, 256): 2000}.items()
    }
    theory = top_singular_value_theory(ratios * critical, c)
    out = {
        "mp_ks_distance_2000x1000": report.ks_distance,
        "false_alarm_rate_at_alpha_0.01": calibration,
        "bbp_max_abs_error_top_singular_value": float(np.max(np.abs(np.array(top_mean) - theory))),
        "bbp_detection_rate": {f"{r:g}": pw for r, pw in zip(ratios, power)},
    }
    record("exp1_theory", out)
    return out


if __name__ == "__main__":
    print(run())
