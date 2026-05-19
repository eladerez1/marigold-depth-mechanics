#!/usr/bin/env python3
"""
Train Model C: single-step depth regression on SD2 U-Net.

Requires GPU + Hypersim/VKitti data. Estimated 18–24h on 32GB GPU (bf16).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Model C (single-step regression)")
    p.add_argument("--output_dir", type=str, default="checkpoints/model_C")
    p.add_argument("--train_data", type=str, default="hypersim,vkitti")
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", type=str, default="marigold-internals")
    p.add_argument("--wandb_run", type=str, default="model-C-single-step")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except ImportError as e:
        raise SystemExit("PyTorch required") from e

    if not torch.cuda.is_available():
        raise SystemExit(
            "Model C training requires a CUDA GPU.\n"
            "Prepare data and env now; launch on DGX when a GPU is free:\n"
            "  CUDA_VISIBLE_DEVICES=0 python src/models/train_single_step.py ..."
        )

    # Training loop placeholder — wire Hypersim/VKitti loaders before long run.
    raise NotImplementedError(
        "Dataset loaders for Hypersim + Virtual KITTI are not wired yet.\n"
        "Next: implement src/data/hypersim.py and src/data/vkitti.py, then SiLog loss\n"
        "on VAE-encoded depth latents at t=1 (Marigold-style 8-channel UNet input).\n"
        f"Config: steps={args.steps}, data={args.train_data}, out={out}"
    )


if __name__ == "__main__":
    main()
