"""NYU RGB-D pairs in Marigold training batch format (interim until Hypersim/VKitti)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize

NYU_RGB_ROOT = Path(
    "/isilon/Automotive/RnD/elad.e/Dev/research/sparse_confidence/datasets/nyu_raw/colmap_input"
)
NYU_DEPTH_ROOT = Path(
    "/isilon/Automotive/RnD/elad.e/Dev/research/sparse_confidence/datasets/nyu_raw/gt_depth"
)


def collect_nyu_train_pairs(max_images: int | None = None) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for seq_dir in sorted(NYU_DEPTH_ROOT.iterdir()):
        if not seq_dir.is_dir():
            continue
        rgb_dir = NYU_RGB_ROOT / seq_dir.name / "images"
        if not rgb_dir.exists():
            continue
        for depth_path in sorted(seq_dir.glob("*.npy")):
            rgb_path = rgb_dir / f"{depth_path.stem}.jpg"
            if rgb_path.exists():
                pairs.append((rgb_path, depth_path))
            if max_images is not None and len(pairs) >= max_images:
                return pairs
    return pairs


class NYUMarigoldTrainDataset(Dataset):
    """Returns keys compatible with Marigold depth training."""

    disp_name = "nyu_proxy_train"

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        depth_transform,
        resize_to_hw: tuple[int, int] = (480, 640),
        lr_flip_p: float = 0.5,
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
    ) -> None:
        self.pairs = pairs
        self.depth_transform = depth_transform
        self.resize_to_hw = resize_to_hw
        self.lr_flip_p = lr_flip_p
        self.min_depth = min_depth
        self.max_depth = max_depth
        self._resize = Resize(
            size=resize_to_hw, interpolation=InterpolationMode.BILINEAR
        )
        self._resize_depth = Resize(
            size=resize_to_hw, interpolation=InterpolationMode.NEAREST_EXACT
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rgb_path, depth_path = self.pairs[index]
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        rgb = np.transpose(rgb, (2, 0, 1)).astype(np.int32)
        depth = np.load(depth_path).astype(np.float32)
        if depth.ndim == 3:
            depth = depth.squeeze()
        depth = torch.from_numpy(depth).float().unsqueeze(0)

        rgb_t = torch.from_numpy(rgb).float()
        rgb_norm = rgb_t / 255.0 * 2.0 - 1.0

        rasters = {
            "rgb_int": rgb_t.int(),
            "rgb_norm": rgb_norm,
            "depth_raw_linear": depth,
            "depth_filled_linear": depth.clone(),
        }
        rasters["valid_mask_raw"] = self._valid_mask(rasters["depth_raw_linear"])
        rasters["valid_mask_filled"] = rasters["valid_mask_raw"].clone()

        if random.random() < self.lr_flip_p:
            rasters = {k: v.flip(-1) for k, v in rasters.items()}

        rasters["depth_raw_norm"] = self.depth_transform(
            rasters["depth_raw_linear"], rasters["valid_mask_raw"]
        )
        rasters["depth_filled_norm"] = rasters["depth_raw_norm"].clone()

        if self.depth_transform.far_plane_at_max:
            rasters["depth_filled_norm"][~rasters["valid_mask_filled"]] = (
                self.depth_transform.norm_max
            )
        else:
            rasters["depth_filled_norm"][~rasters["valid_mask_filled"]] = (
                self.depth_transform.norm_min
            )

        rasters["rgb_norm"] = self._resize(rasters["rgb_norm"])
        rasters["rgb_int"] = self._resize_depth(
            rasters["rgb_int"].float()
        ).int()
        for key in (
            "depth_raw_linear",
            "depth_filled_linear",
            "depth_raw_norm",
            "depth_filled_norm",
            "valid_mask_raw",
            "valid_mask_filled",
        ):
            rasters[key] = self._resize_depth(rasters[key])

        return rasters

    def _valid_mask(self, depth: torch.Tensor) -> torch.Tensor:
        return torch.logical_and(
            depth > self.min_depth, depth < self.max_depth
        ).bool()
