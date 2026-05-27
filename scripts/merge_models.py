#!/usr/bin/env python3
"""
Merge two fine-tuned Marigold UNet checkpoints by weighted averaging.

Both models must start from the same base checkpoint (prs-eth/marigold-depth-v1-1).
The merged UNet weights are: merged = alpha * A + (1 - alpha) * B

Usage:
    python scripts/merge_models.py \
        --ckpt_a /results/model_3drealcar/checkpoint-5000/unet \
        --ckpt_b /results/model_uveye/checkpoint-3000/unet \
        --alpha 0.5 \
        --output /results/model_merged/unet

    # Try alpha sweep to find best blend:
    python scripts/merge_models.py --sweep \
        --ckpt_a ... --ckpt_b ... --output /results/model_merged
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from diffusers import UNet2DConditionModel
from safetensors.torch import load_file, save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_weights(ckpt_path: str | Path) -> dict[str, torch.Tensor]:
    p = Path(ckpt_path)
    sf = p / "diffusion_pytorch_model.safetensors"
    if sf.exists():
        return load_file(str(sf))
    # Fall back to pytorch bin
    bin_f = p / "diffusion_pytorch_model.bin"
    if bin_f.exists():
        return torch.load(str(bin_f), map_location="cpu")
    raise FileNotFoundError(f"No weights found in {p}")


def merge_weights(
    weights_a: dict[str, torch.Tensor],
    weights_b: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """merged = alpha * A + (1 - alpha) * B"""
    assert set(weights_a.keys()) == set(weights_b.keys()), \
        "Checkpoints have different parameter sets — both must come from the same base"
    merged = {}
    for k in weights_a:
        wa = weights_a[k].float()
        wb = weights_b[k].float()
        merged[k] = (alpha * wa + (1.0 - alpha) * wb).to(wa.dtype)
    return merged


def save_merged(
    merged: dict[str, torch.Tensor],
    out_dir: str | Path,
    ckpt_a: str | Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy config from ckpt_a
    import shutil
    config_src = Path(ckpt_a) / "config.json"
    if config_src.exists():
        shutil.copy(config_src, out_dir / "config.json")

    save_file(merged, str(out_dir / "diffusion_pytorch_model.safetensors"))
    log.info("Merged UNet saved to %s", out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_a", required=True,
                   help="Path to UNet checkpoint A (e.g. model_3drealcar/.../unet)")
    p.add_argument("--ckpt_b", required=True,
                   help="Path to UNet checkpoint B (e.g. model_uveye/.../unet)")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Weight for model A: merged = alpha*A + (1-alpha)*B")
    p.add_argument("--output", required=True,
                   help="Output directory for merged UNet")
    p.add_argument("--sweep", action="store_true",
                   help="Save alpha=0.3,0.5,0.7 variants for evaluation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("Loading checkpoint A from %s", args.ckpt_a)
    wa = load_weights(args.ckpt_a)
    log.info("Loading checkpoint B from %s", args.ckpt_b)
    wb = load_weights(args.ckpt_b)
    log.info("Parameters: %d tensors", len(wa))

    if args.sweep:
        for alpha in [0.3, 0.5, 0.7]:
            out = Path(args.output) / f"alpha_{alpha:.1f}" / "unet"
            merged = merge_weights(wa, wb, alpha)
            save_merged(merged, out, args.ckpt_a)
            log.info("alpha=%.1f saved to %s", alpha, out)
    else:
        merged = merge_weights(wa, wb, args.alpha)
        save_merged(merged, Path(args.output) / "unet", args.ckpt_a)
        log.info("alpha=%.2f  merged done", args.alpha)


if __name__ == "__main__":
    main()
