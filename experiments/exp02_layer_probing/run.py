#!/usr/bin/env python3
"""Exp 02: layer-wise linear probing (Q1) for models A–D."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.extraction.feature_extractor import FeatureExtractor
from src.models.load_models import ModelID, load_model
from src.probing.probe_tasks import TASK_SPECS
from src.probing.probe_trainer import train_linear_probe


def parse_models(s: str) -> list[ModelID]:
    return [ModelID(x.strip()) for x in s.split(",")]


def parse_tasks(s: str) -> list[str]:
    return [x.strip() for x in s.split(",")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, default="A,B,C,D")
    p.add_argument("--dataset", type=str, default="nyuv2")
    p.add_argument("--tasks", type=str, default="ordinal,depth,normals,boundary,planar")
    p.add_argument("--output", type=str, default="results/exp02")
    p.add_argument("--feature_cache", type=str, default="data/features")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.feature_cache)

    if not cache.exists() or not any(cache.glob("*.h5")):
        raise SystemExit(
            f"No HDF5 feature cache under {cache}.\n"
            "Run feature extraction on GPU first (see src/extraction/extract_and_cache.py)."
        )

    # Placeholder: load cached features, train probes, write CSV
    csv_path = out / "probing_matrix.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "layer", "task", "metric_value"])
    print(f"Scaffold wrote empty {csv_path} — implement HDF5 loader + NYUv2 labels.")


if __name__ == "__main__":
    main()
