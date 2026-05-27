"""
3DRealCar dataset loader for Marigold fine-tuning.

Dataset structure (per car, one folder per car ID):
    <root>/
        <car_id>/
            frame_*.jpg          -- RGB images
            frame_*.json         -- camera params + depth path from ARKit
            depth/
                frame_*.png      -- uint16 depth maps (mm), or
                frame_*.npy      -- float32 depth maps (metres)

The depth is captured with iPhone 14 ARKit (Apple ToF LiDAR) and is
dense and metric (accuracy ~1-3 cm at close range).

Each frame_*.json contains (at minimum):
    {
        "fx": ..., "fy": ..., "cx": ..., "cy": ...,   <- intrinsics
        "depth_path": "depth/frame_XXXX.png",           <- optional
        "transform_matrix": [[...], ...]                <- c2w 4x4
    }

Usage:
    ds = ThreeDRealCarDataset("/data/3DRealCar", image_size=(480, 640))
    sample = ds[0]   # {"rgb": [3,H,W], "depth": [1,H,W], "mask": [1,H,W], ...}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class ThreeDRealCarDataset(Dataset):
    """
    RGB-D dataset from 3DRealCar (ARKit-captured, per-frame depth maps).

    Args:
        root: Path to the 3DRealCar root directory.
        image_size: (H, W) to resize to.
        lighting: Filter by lighting condition ('standard', 'reflective', 'dark', or None = all).
        max_depth_m: Clip depth values above this (metres). Default 10m.
        min_valid_frac: Min fraction of valid depth pixels to include a sample.
        max_samples: Cap on dataset size.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: tuple[int, int] = (480, 640),
        lighting: Optional[str] = "standard",
        max_depth_m: float = 10.0,
        min_valid_frac: float = 0.10,
        max_samples: Optional[int] = None,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.max_depth_m = max_depth_m
        self.min_valid_frac = min_valid_frac

        self.samples = self._discover(lighting)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        log.info("ThreeDRealCarDataset: %d samples from %s", len(self.samples), self.root)

    # ------------------------------------------------------------------
    def _discover(self, lighting: Optional[str]) -> list[dict]:
        samples: list[dict] = []
        car_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        for car_dir in car_dirs:
            # Check lighting condition via info.json if available
            if lighting is not None:
                info_path = car_dir / "info.json"
                if info_path.exists():
                    try:
                        info = json.loads(info_path.read_text())
                        if info.get("lighting", "standard") != lighting:
                            continue
                    except Exception:
                        pass

            # Find all frame JSON files
            for json_path in sorted(car_dir.glob("frame_*.json")):
                frame_idx = json_path.stem.replace("frame_", "")
                img_path = car_dir / f"frame_{frame_idx}.jpg"
                if not img_path.exists():
                    continue

                # Find corresponding depth (PNG uint16 mm, or NPY float32 m)
                depth_path = self._find_depth(car_dir, frame_idx, json_path)
                if depth_path is None:
                    continue

                samples.append({
                    "car_id": car_dir.name,
                    "frame": frame_idx,
                    "img_path": img_path,
                    "depth_path": depth_path,
                    "json_path": json_path,
                })
        return samples

    def _find_depth(
        self, car_dir: Path, frame_idx: str, json_path: Path
    ) -> Optional[Path]:
        """Try multiple possible depth file locations/formats."""
        candidates = [
            car_dir / "depth" / f"frame_{frame_idx}.png",
            car_dir / "depth" / f"frame_{frame_idx}.npy",
            car_dir / f"depth_{frame_idx}.png",
            car_dir / f"frame_{frame_idx}_depth.png",
        ]
        # Also check if the JSON specifies a depth_path
        try:
            meta = json.loads(json_path.read_text())
            if "depth_path" in meta:
                candidates.insert(0, car_dir / meta["depth_path"])
        except Exception:
            pass

        for c in candidates:
            if c.exists():
                return c
        return None

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        H, W = self.image_size

        # --- RGB ---
        img = Image.open(s["img_path"]).convert("RGB")
        img = img.resize((W, H), Image.BILINEAR)
        rgb = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

        # --- Depth ---
        depth_raw = self._load_depth(s["depth_path"])  # float32, metres
        depth_h, depth_w = depth_raw.shape
        if (depth_h, depth_w) != (H, W):
            depth_raw = cv2.resize(depth_raw, (W, H), interpolation=cv2.INTER_NEAREST)

        # Clip and mask
        depth_raw = np.clip(depth_raw, 0.0, self.max_depth_m)
        mask = (depth_raw > 0).astype(np.float32)

        valid_frac = mask.mean()
        if valid_frac < self.min_valid_frac:
            # Fall back to a random valid sample rather than crashing DataLoader
            return self.__getitem__((idx + 1) % len(self))

        depth = torch.from_numpy(depth_raw[None].astype(np.float32))  # [1, H, W]
        mask_t = torch.from_numpy(mask[None].astype(bool))             # [1, H, W]

        return {
            "rgb": rgb,
            "depth": depth,
            "mask": mask_t,
            "valid_frac": float(valid_frac),
            "car_id": s["car_id"],
        }

    def _load_depth(self, depth_path: Path) -> np.ndarray:
        """Load depth map as float32 metres."""
        if depth_path.suffix == ".npy":
            d = np.load(str(depth_path)).astype(np.float32)
            # If values are huge (>100) assume mm → convert to metres
            if d.max() > 100:
                d = d / 1000.0
            return d
        else:
            # PNG: assume uint16 in mm (ARKit convention)
            d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if d is None:
                raise IOError(f"Could not read depth: {depth_path}")
            return d.astype(np.float32) / 1000.0  # mm → m
