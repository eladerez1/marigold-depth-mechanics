#!/usr/bin/env python3
"""
Extract U-Net features and save to HDF5 (expensive — run on GPU).

Usage:
  python src/extraction/extract_and_cache.py --model B --dataset nyuv2 --out data/features/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.extraction.feature_extractor import FeatureExtractor
from src.extraction.timestep_sampler import timesteps_for_model
from src.models.load_models import ModelID, load_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=["A", "B", "C", "D"])
    p.add_argument("--dataset", type=str, default="nyuv2")
    p.add_argument("--out", type=str, default="data/features")
    p.add_argument("--max_images", type=int, default=0, help="0 = all")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Feature extraction requires CUDA.")

    model_id = ModelID(args.model)
    loaded = load_model(model_id)
    extractor = FeatureExtractor(loaded.unet)
    extractor.register_hooks()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.model}_{args.dataset}.h5"

    timesteps = timesteps_for_model(args.model)
    print(f"Model {args.model}, timesteps={timesteps}, output={out_file}")
    raise NotImplementedError(
        "Wire NYUv2/KITTI image loader and Marigold forward pass per timestep, "
        "then write layers to HDF5 groups."
    )


if __name__ == "__main__":
    main()
