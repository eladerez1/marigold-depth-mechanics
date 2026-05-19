#!/usr/bin/env python3
"""CPU smoke test: imports, probe trainer on synthetic features, weight-delta on tiny tensors."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from src.analysis.weight_delta import compute_weight_delta
from src.probing.probe_trainer import train_linear_probe
from src.probing.probe_tasks import TASK_SPECS


def test_weight_delta() -> None:
    a = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    b = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    with torch.no_grad():
        for p in b.parameters():
            p.add_(0.01)
    deltas = compute_weight_delta(a, b)
    assert len(deltas) > 0
    print(f"  weight_delta: {len(deltas)} parameters")


def test_probe_trainer() -> None:
    n, d = 200, 32
    x = torch.randn(n, d)
    y = (x[:, 0] > 0).float().unsqueeze(1)
    spec = TASK_SPECS["ordinal"]
    probe, metrics = train_linear_probe(
        x, y, x[:40], y[:40], output_dim=spec.output_dim, loss_type=spec.loss, max_epochs=5
    )
    print(f"  probe_trainer: val_acc={metrics.get('val_acc', metrics):.3f}")


def test_imports() -> None:
    import diffusers  # noqa: F401
    import transformers  # noqa: F401

    print(f"  diffusers {diffusers.__version__}")


def main() -> None:
    print("Smoke test (CPU)...")
    test_imports()
    test_weight_delta()
    test_probe_trainer()
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
