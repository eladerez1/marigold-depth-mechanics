#!/usr/bin/env python3
"""Exp 04: compare internals of B (multi-step), C (single-step), D (1-NFE)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.compare_models import linear_cka


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, default="B,C,D")
    p.add_argument("--layer_comparison", type=str, default="all")
    p.add_argument("--output", type=str, default="results/exp04")
    p.add_argument("--feature_cache", type=str, default="data/features")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "cka_matrix.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_a", "model_b", "layer", "cka"])
    print(f"Exp04 scaffold: compute CKA B vs C per layer. Wrote empty {csv_path}")


if __name__ == "__main__":
    main()
