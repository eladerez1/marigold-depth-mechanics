"""Figure 4: layer-wise CKA heatmap between two models."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_cka_matrix(csv_path: Path, out_path: Path) -> None:
    layers: list[str] = []
    values: list[float] = []
    with Path(csv_path).open() as f:
        for row in csv.DictReader(f):
            layers.append(row["layer"])
            values.append(float(row["cka"]))

    mat = np.array(values).reshape(1, -1)
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90, fontsize=7)
    ax.set_yticks([0])
    ax.set_yticklabels(["CKA"])
    ax.set_title("Layer-wise CKA (Model B vs Model C)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
