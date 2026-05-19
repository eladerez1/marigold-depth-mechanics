"""Figure 1: per-block weight delta bar chart from CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def plot_weight_deltas(csv_path: Path, out_path: Path) -> None:
    layers, values = [], []
    with Path(csv_path).open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            layers.append(row["layer"])
            values.append(float(row["relative_l2_delta"]))

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(layers)), values, color="steelblue")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90, fontsize=7)
    ax.set_ylabel("Relative L2 weight change")
    ax.set_title("SD2 → Marigold weight delta by U-Net block")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
