"""Command line: ``noisefloor weights.npz`` prints a per-layer spectrum report."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .denoise import denoise
from .spectrum import analyze_model, format_reports


def load_weights(path: Path) -> dict[str, np.ndarray]:
    """Read a ``.npz`` archive (many arrays) or a single ``.npy`` matrix."""
    if path.suffix == ".npy":
        return {path.stem: np.load(path)}
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="noisefloor", description="Separate signal from noise in neural-network weight matrices."
    )
    parser.add_argument("weights", type=Path, help=".npz archive or .npy matrix of weights")
    parser.add_argument("--alpha", type=float, default=0.01, help="false-alarm rate per matrix (default 0.01)")
    parser.add_argument("--denoise", type=Path, metavar="OUT.npz",
                        help="also write a copy with the noise bulk removed from every analysed matrix")
    args = parser.parse_args(argv)

    weights = load_weights(args.weights)
    reports = analyze_model(weights, alpha=args.alpha)
    if not reports:
        print("no weight matrices found (need at least 2 dimensions and 8 rows/columns)")
        return 1
    print(format_reports(reports))
    if args.denoise is not None:
        cleaned = dict(weights)
        for name in reports:
            cleaned[name] = denoise(weights[name], "hard", alpha=args.alpha)[0]
        np.savez(args.denoise, **cleaned)  # type: ignore[arg-type]
        print(f"\nwrote {args.denoise}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
