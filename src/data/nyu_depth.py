"""NYUv2 depth dataset loader for probing (train/val/test splits)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset


class NYUDepthDataset(Dataset):
  """Expects data/nyu_depth_v2/ with RGB + depth pairs (implement after download)."""

  def __init__(self, root: str | Path, split: str = "train") -> None:
    self.root = Path(root)
    self.split = split
    self.samples: list[tuple[Path, Path]] = []
    self._index_samples()

  def _index_samples(self) -> None:
    raw = self.root / "raw"
    if not raw.exists():
      return
    for rgb in sorted(raw.glob("**/*_rgb.png")):
      depth = rgb.parent / rgb.name.replace("_rgb.png", "_depth.png")
      if depth.exists():
        self.samples.append((rgb, depth))

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
    raise NotImplementedError("Load RGB/depth tensors and probing labels.")


def collate_probe_batch(batch: list) -> dict[str, torch.Tensor]:
  raise NotImplementedError
