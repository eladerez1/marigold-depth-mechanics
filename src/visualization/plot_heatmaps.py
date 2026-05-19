"""Figure 2: layer × task probing heatmaps (one panel per model)."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_probing_matrix(csv_path: Path) -> tuple[list[str], list[str], np.ndarray, list[str]]:
    """CSV columns: model, layer, task, metric_value."""
    models, layers, tasks, values = [], [], [], []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            models.append(row["model"])
            layers.append(row["layer"])
            tasks.append(row["task"])
            values.append(float(row["metric_value"]))
    unique_models = sorted(set(models))
    unique_layers = sorted(set(layers), key=lambda x: layers.index(x))
    unique_tasks = sorted(set(tasks))
    return unique_models, unique_layers, unique_tasks, values


def plot_probing_heatmaps(csv_path: Path, out_path: Path) -> None:
    models_set, layers, tasks, _ = load_probing_matrix(csv_path)
    # Re-read into per-model matrices
    data: dict[str, dict[tuple[str, str], float]] = {m: {} for m in models_set}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            key = (row["layer"], row["task"])
            val = float(row["metric_value"])
            prev = data[row["model"]].get(key)
            data[row["model"]][key] = val if prev is None else max(prev, val)

    n = len(models_set)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, max(6, len(layers) * 0.15)))
    if n == 1:
        axes = [axes]
    for ax, model in zip(axes, models_set):
        mat = np.zeros((len(layers), len(tasks)))
        for i, layer in enumerate(layers):
            for j, task in enumerate(tasks):
                mat[i, j] = data[model].get((layer, task), np.nan)
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels(tasks, rotation=45, ha="right")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"Model {model}")
    fig.suptitle("Layer × task probing accuracy")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
