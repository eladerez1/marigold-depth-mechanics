#!/usr/bin/env python3
"""CPU smoke test for spatial labels + probes."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.probing.spatial_labels import make_spatial_probe_labels
from src.probing.spatial_probe import train_spatial_probe


def main() -> None:
    torch.manual_seed(0)
    h, w, c = 64, 64, 32
    n = 24
    depth = torch.rand(h, w) * 3.0 + 0.5
    labels = make_spatial_probe_labels(depth)
    assert labels["depth"].shape == (h, w)
    assert labels["ordinal"].shape == (h, w - 1)

    feats = []
    labs = []
    for i in range(n):
        f = torch.randn(c, h, w) * 0.1
        f[:, :, : w - 1] += labels["ordinal"].unsqueeze(0) * 4.0
        feats.append(f)
        labs.append(labels["ordinal"].clone())

    metrics = train_spatial_probe(
        feats,
        labs,
        "ordinal",
        train_idx=list(range(16)),
        val_idx=list(range(16, 24)),
        max_pixels_per_image=2048,
        max_epochs=30,
        device=torch.device("cpu"),
    )
    acc = metrics["val_metric"]
    print(f"ordinal val_metric={acc:.3f} (expect >0.55 on synthetic)")
    if acc < 0.55:
        raise SystemExit("Spatial probe did not beat chance on synthetic data")
    print("spatial probe smoke test passed")


if __name__ == "__main__":
    main()
