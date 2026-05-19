#!/usr/bin/env python3
"""Exp 01: relative weight change SD2 vs Marigold."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.weight_delta import aggregate_by_block, compute_weight_delta
from src.models.load_models import load_unet_pair_for_delta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_a", type=str, required=True)
    p.add_argument("--model_b", type=str, required=True)
    p.add_argument("--output", type=str, default="results/exp01")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading U-Nets (CPU, float32)...")
    unet_a, unet_b = load_unet_pair_for_delta(args.model_a, args.model_b)
    deltas = compute_weight_delta(unet_a, unet_b)
    block = aggregate_by_block(deltas)

    csv_path = out / "weight_delta_by_layer.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "relative_l2_delta"])
        for k, v in sorted(block.items()):
            w.writerow([k, f"{v:.6e}"])
    print(f"Wrote {csv_path} ({len(block)} blocks)")

    try:
        from src.visualization.plot_weight_deltas import plot_weight_deltas

        fig_dir = ROOT / "results" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_weight_deltas(csv_path, fig_dir / "fig1_weight_delta.png")
    except Exception as e:
        print(f"Figure skipped: {e}")


if __name__ == "__main__":
    main()
