"""Figure 3: probing metric vs denoising timestep (per model)."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def plot_timestep_curves(csv_path: Path, out_path: Path) -> None:
    # columns: model, timestep, task, metric_value (legacy: no model col)
    by_model_task: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    best_at_t: dict[tuple[str, str, int], float] = {}
    with Path(csv_path).open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row.get("model", "B")
            task = row["task"]
            t = int(row["timestep"])
            val = float(row["metric_value"])
            key = (model, task, t)
            best_at_t[key] = max(best_at_t.get(key, 0.0), val)
    for (model, task, t), val in best_at_t.items():
        by_model_task[(model, task)].append((t, val))

    models = sorted({m for m, _ in by_model_task})
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
    for ax, model in zip(axes[0], models):
        for task in sorted({t for m, t in by_model_task if m == model}):
            pairs = sorted(by_model_task[(model, task)], key=lambda x: -x[0])
            ts, vals = zip(*pairs)
            ax.plot(ts, vals, marker="o", label=task)
        ax.set_xlabel("Denoising timestep t")
        ax.set_ylabel("Probe metric")
        ax.set_title(f"Model {model}")
        ax.legend(fontsize=8)
        ax.invert_xaxis()
    fig.suptitle("Timestep specialization")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
