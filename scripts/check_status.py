#!/usr/bin/env python3
"""Print which pipeline steps are complete (no GPU required)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ok(path: Path, label: str) -> str:
    return f"[done]    {label}" if path.exists() else f"[pending] {label}"


def main() -> None:
    ckpt = ROOT / "checkpoints"
    data = ROOT / "data"
    results = ROOT / "results"

    lines = [
        "Marigold-internals status",
        "=" * 40,
        ok(ckpt / "model_A_sd2" / "unet", "Model A (SD2 UNet)"),
        ok(ckpt / "model_B_marigold", "Model B (Marigold)"),
        ok(ckpt / "model_C", "Model C (single-step, trained)"),
        ok(data / "nyu_depth_v2", "NYUv2 dir"),
        ok(data / "kitti_eigen", "KITTI Eigen dir"),
        ok(results / "exp01" / "weight_delta_by_layer.csv", "Exp01 CSV"),
        ok(results / "exp02" / "probing_matrix.csv", "Exp02 CSV"),
        ok(results / "exp03" / "timestep_curves.csv", "Exp03 CSV"),
        ok(results / "exp04" / "cka_matrix.csv", "Exp04 CSV"),
        ok(results / "figures" / "fig1_weight_delta.png", "Figure 1"),
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
