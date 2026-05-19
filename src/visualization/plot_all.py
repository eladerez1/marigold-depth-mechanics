#!/usr/bin/env python3
"""Generate all paper figures from results CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.visualization.plot_heatmaps import plot_probing_heatmaps
from src.visualization.plot_timestep_curves import plot_timestep_curves
from src.visualization.plot_weight_deltas import plot_weight_deltas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--output", type=str, default="results/figures")
    args = p.parse_args()

    results = Path(args.results_dir)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    exp01 = results / "exp01" / "weight_delta_by_layer.csv"
    if exp01.exists():
        plot_weight_deltas(exp01, out / "fig1_weight_delta.png")
        print("fig1_weight_delta.png")

    exp02 = results / "exp02" / "probing_matrix.csv"
    if exp02.exists():
        plot_probing_heatmaps(exp02, out / "fig2_layer_task_heatmap.png")
        print("fig2_layer_task_heatmap.png")

    exp03 = results / "exp03" / "timestep_curves.csv"
    if exp03.exists():
        plot_timestep_curves(exp03, out / "fig3_timestep_specialization.png")
        print("fig3_timestep_specialization.png")

    exp04 = results / "exp04" / "cka_matrix.csv"
    if exp04.exists():
        from src.visualization.plot_cka import plot_cka_matrix

        plot_cka_matrix(exp04, out / "fig4_cka_matrix.png")
        print("fig4_cka_matrix.png")


if __name__ == "__main__":
    main()
