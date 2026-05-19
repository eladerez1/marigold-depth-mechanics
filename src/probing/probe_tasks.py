"""Probing task definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    output_dim: int
    loss: str  # bce | mse | cosine | silog
    metric: str


TASK_SPECS: dict[str, TaskSpec] = {
    "ordinal": TaskSpec("ordinal", 1, "bce", "accuracy"),
    "depth": TaskSpec("depth", 1, "silog", "absrel"),
    "normals": TaskSpec("normals", 3, "cosine", "mean_angle"),
    "boundary": TaskSpec("boundary", 1, "bce", "f1"),
    "planar": TaskSpec("planar", 1, "bce", "iou"),
}
