"""Reproduce every figure and number in the README: ``python experiments/run_all.py``."""

from __future__ import annotations

import time

import exp1_theory
import exp2_lazy_vs_rich
import exp3_compression
import exp4_memorization

if __name__ == "__main__":
    start = time.time()
    for module in (exp1_theory, exp2_lazy_vs_rich, exp3_compression, exp4_memorization):
        tic = time.time()
        print(f"{module.__name__} ...")
        module.run()
        print(f"  done in {time.time() - tic:.0f}s")
    print(f"all experiments finished in {time.time() - start:.0f}s; numbers are in results/results.json")
