"""Load MarigoldDepthPipeline for ACR / DGX (handles broken isilon→raid symlinks)."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def hf_home() -> str:
    return os.environ.get(
        "HF_HOME",
        "/isilon/Automotive/RnD/elad.e/.cache/huggingface",
    )


def checkpoint_usable(ckpt: Path) -> bool:
    for name in (
        "diffusion_pytorch_model.fp16.safetensors",
        "diffusion_pytorch_model.safetensors",
    ):
        weight = ckpt / "unet" / name
        if not weight.exists():
            continue
        try:
            resolved = weight.resolve()
            if resolved.is_file() and resolved.stat().st_size > 1_000_000:
                return True
        except OSError:
            continue
    return False


def resolve_marigold_hub(checkpoint_dir: Path) -> str:
    if (
        (checkpoint_dir / "unet" / "config.json").exists()
        and (checkpoint_dir / "model_index.json").exists()
        and checkpoint_usable(checkpoint_dir)
    ):
        return str(checkpoint_dir)
    return "prs-eth/marigold-depth-v1-1"


def load_marigold_depth_pipeline(
    device: str,
    checkpoint_dir: Path | str = "checkpoints/model_B_marigold",
):
    from marigold import MarigoldDepthPipeline

    ckpt = Path(checkpoint_dir)
    hub = resolve_marigold_hub(ckpt)
    cache_dir = hf_home()
    print(f"Loading Marigold from: {hub} (HF_HOME={cache_dir})")

    attempts = (
        {"variant": "fp16", "use_safetensors": True},
        {"variant": None, "use_safetensors": True},
        {"variant": None, "use_safetensors": False},
    )
    last_err: Exception | None = None
    for extra in attempts:
        label = f"variant={extra['variant']!r} safetensors={extra['use_safetensors']}"
        try:
            kwargs: dict = dict(
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
                use_safetensors=extra["use_safetensors"],
            )
            if extra["variant"] is not None:
                kwargs["variant"] = extra["variant"]
            pipe = MarigoldDepthPipeline.from_pretrained(hub, **kwargs)
            print(f"  loaded with {label}")
            return pipe.to(device)
        except (OSError, EnvironmentError) as e:
            print(f"  {label} failed: {e}")
            last_err = e
    raise RuntimeError(f"Could not load Marigold from {hub}") from last_err
