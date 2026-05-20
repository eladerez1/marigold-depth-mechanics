"""Load model variants A–D for probing and feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel

from src.models.checkpoint_paths import model_c_dir


class ModelID(str, Enum):
    A = "A"  # SD2 vanilla
    B = "B"  # Marigold multi-step
    C = "C"  # Single-step regression (trained)
    D = "D"  # Marigold 1-NFE at inference


@dataclass
class LoadedModel:
    model_id: ModelID
    unet: UNet2DConditionModel
    vae: AutoencoderKL
    scheduler: DDIMScheduler
    inference_steps: int
    device: torch.device


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    model_id: ModelID,
    checkpoint_root: Path | str = "checkpoints",
    dtype: torch.dtype = torch.float16,
) -> LoadedModel:
    root = Path(checkpoint_root)
    device = _device()

    if model_id == ModelID.A:
        path = root / "model_A_sd2"
        unet = UNet2DConditionModel.from_pretrained(path, subfolder="unet", torch_dtype=dtype)
        vae = AutoencoderKL.from_pretrained(path, subfolder="vae", torch_dtype=dtype)
        scheduler = DDIMScheduler.from_pretrained(path, subfolder="scheduler")
        steps = 1
    elif model_id in (ModelID.B, ModelID.D):
        path = root / "model_B_marigold"
        unet = UNet2DConditionModel.from_pretrained(path, subfolder="unet", torch_dtype=dtype)
        vae = AutoencoderKL.from_pretrained(path, subfolder="vae", torch_dtype=dtype)
        scheduler = DDIMScheduler.from_pretrained(path, subfolder="scheduler")
        steps = 10 if model_id == ModelID.B else 1
    elif model_id == ModelID.C:
        path = model_c_dir(root)
        unet = UNet2DConditionModel.from_pretrained(path, subfolder="unet", torch_dtype=dtype)
        vae = AutoencoderKL.from_pretrained(path, subfolder="vae", torch_dtype=dtype)
        scheduler = DDIMScheduler.from_pretrained(path, subfolder="scheduler")
        steps = 1
    else:
        raise ValueError(model_id)

    unet.to(device).eval()
    vae.to(device).eval()
    return LoadedModel(
        model_id=model_id,
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        inference_steps=steps,
        device=device,
    )


def load_unet_pair_for_delta(
    path_a: str | Path,
    path_b: str | Path,
    dtype: torch.dtype = torch.float32,
) -> tuple[UNet2DConditionModel, UNet2DConditionModel]:
    """Load two U-Nets on CPU for weight-delta (float32 for numerical stability)."""
    unet_a = UNet2DConditionModel.from_pretrained(path_a, subfolder="unet", torch_dtype=dtype)
    unet_b = UNet2DConditionModel.from_pretrained(path_b, subfolder="unet", torch_dtype=dtype)
    return unet_a.cpu(), unet_b.cpu()
