"""Resolve checkpoint locations (ACR mounts isilon ro; Model C writes under results/)."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def model_c_dir(checkpoint_root: Path | str | None = None) -> Path:
    """Writable training output lives in results/model_C; legacy path is checkpoints/model_C."""
    root = Path(checkpoint_root) if checkpoint_root else project_root() / "checkpoints"
    proj = project_root()
    candidates = [
        proj / "results" / "model_C",
        root / "model_C",
        proj / "checkpoints" / "model_C",
    ]
    isilon = os.environ.get("MARIGOLD_ISILON_ROOT")
    if isilon:
        base = Path(isilon)
        candidates = [
            base / "results" / "model_C",
            base / "checkpoints" / "model_C",
            *candidates,
        ]
    for path in candidates:
        if (path / "unet" / "config.json").exists():
            return path
    # Default write target in ACR / isilon-ro containers
    if isilon:
        return Path(isilon) / "results" / "model_C"
    return proj / "checkpoints" / "model_C"


def model_c_ready(checkpoint_root: Path | str | None = None) -> bool:
    p = model_c_dir(checkpoint_root)
    return (p / "unet" / "config.json").exists()
