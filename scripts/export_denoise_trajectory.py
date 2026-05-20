#!/usr/bin/env python3
"""Export per-timestep depth + PCA feature frames for the denoise viewer.

Requires GPU. Example (on dgx04):

  cd /isilon/.../marigold_depth_mechanics
  conda run -n sd_visualizer python scripts/export_denoise_trajectory.py \\
    --nyu_index 0 --n_steps 10 --gpu 0

View: conda run -n sd_visualizer python scripts/denoise_viewer.py --port 8766
Tunnel: ssh -L 8766:localhost:8766 dgx04
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MARIGOLD_ROOT = ROOT / "third_party" / "Marigold"
sys.path.insert(0, str(MARIGOLD_ROOT))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_gpu_pipeline import (  # noqa: E402
    NYU_DEPTH_ROOT,
    NYU_RGB_ROOT,
    collect_nyu_pairs,
    load_depth_tensor,
    load_rgb_tensor,
)
from src.extraction.denoise_trajectory import (  # noqa: E402
    collect_denoise_trajectory,
    export_trajectory_to_disk,
    update_manifest,
)


def _load_pipe(device: str):
    from src.models.marigold_pipe_loader import load_marigold_depth_pipeline

    return load_marigold_depth_pipeline(device, ROOT / "checkpoints" / "model_B_marigold")


def _tensor_to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    x = t.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    x = (x + 1.0) * 0.5
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def _depth_gt_colormap(depth_64: torch.Tensor) -> np.ndarray:
    from src.extraction.denoise_trajectory import _depth_to_uint8

    d = depth_64.numpy()
    return _depth_to_uint8(d)


def main() -> None:
    p = argparse.ArgumentParser(description="Export denoising trajectory PNGs")
    p.add_argument("--out", type=str, default=str(ROOT / "results" / "denoise_viz"))
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--layer", type=str, default=None, help="UNet ResNet layer name (default: mid_block)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image", type=str, default=None, help="RGB image path (overrides NYU)")
    p.add_argument("--depth_gt", type=str, default=None, help="GT depth .npy (optional)")
    p.add_argument("--nyu_index", type=int, default=0, help="NYU pair index if --image not set")
    p.add_argument("--num_samples", type=int, default=1, help="Export this many NYU pairs from nyu_index")
    p.add_argument("--sample_prefix", type=str, default="nyu")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for trajectory export.")

    device = f"cuda:{args.gpu}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    viz_root = Path(args.out)
    viz_root.mkdir(parents=True, exist_ok=True)

    pipe = _load_pipe(device)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    if args.image:
        jobs = [(Path(args.image), Path(args.depth_gt) if args.depth_gt else None)]
    else:
        pairs = collect_nyu_pairs(args.nyu_index + args.num_samples)
        if not pairs:
            raise SystemExit(f"No NYU pairs under {NYU_RGB_ROOT}")
        jobs = [
            (pairs[i][0], pairs[i][1])
            for i in range(args.nyu_index, min(args.nyu_index + args.num_samples, len(pairs)))
        ]

    for j, (rgb_path, depth_path) in enumerate(jobs):
        sample_id = f"{args.sample_prefix}_{args.nyu_index + j:04d}"
        rgb_t = load_rgb_tensor(rgb_path)
        records, hook_name = collect_denoise_trajectory(
            pipe, rgb_t, args.n_steps, args.layer, device, gen
        )
        gt_u8 = None
        if depth_path and depth_path.exists():
            gt_u8 = _depth_gt_colormap(load_depth_tensor(depth_path))
        export_trajectory_to_disk(
            records,
            hook_name,
            _tensor_to_uint8_rgb(rgb_t),
            viz_root,
            sample_id,
            gt_depth_uint8=gt_u8,
            n_steps=args.n_steps,
        )
        update_manifest(viz_root, sample_id, caption=str(rgb_path.name))
        print(f"Wrote {viz_root / sample_id} ({len(records)} steps, layer={hook_name})")


if __name__ == "__main__":
    main()
