"""Per-parameter and per-block relative weight change between two models."""

from __future__ import annotations

import re
from collections import defaultdict

import torch.nn as nn


def _relative_delta(before, after) -> float:
    delta = (after - before).norm(p=2).item()
    baseline = before.norm(p=2).item() + 1e-8
    return delta / baseline


def compute_weight_delta(
    model_before: nn.Module,
    model_after: nn.Module,
) -> dict[str, float]:
    """Relative L2 change per parameter. Handles Marigold conv_in 4→8 channel expansion."""
    deltas: dict[str, float] = {}
    for (name_a, param_a), (name_b, param_b) in zip(
        model_before.named_parameters(),
        model_after.named_parameters(),
    ):
        assert name_a == name_b
        if param_a.shape == param_b.shape:
            deltas[name_a] = _relative_delta(param_a, param_b)
            continue
        # Marigold doubles UNet input: 4 image-latent + 4 depth-latent channels
        if name_a == "conv_in.weight" and param_b.shape[1] == 2 * param_a.shape[1]:
            in_ch = param_a.shape[1]
            deltas[name_a + ".image_latent"] = _relative_delta(
                param_a, param_b[:, :in_ch]
            )
            deltas[name_a + ".depth_latent_new"] = (
                param_b[:, in_ch:].norm(p=2).item() / (in_ch * param_a[0].numel() ** 0.5 + 1e-8)
            )
            continue
        raise ValueError(f"Shape mismatch at {name_a}: {param_a.shape} vs {param_b.shape}")
    return deltas


def aggregate_by_block(deltas: dict[str, float]) -> dict[str, float]:
    """Aggregate relative deltas by U-Net block prefix (e.g. down_blocks.0)."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for name, val in deltas.items():
        # down_blocks.0.resnets.0.conv1.weight -> down_blocks.0
        m = re.match(r"^(down_blocks\.\d+|mid_block|up_blocks\.\d+)", name)
        key = m.group(1) if m else name.split(".")[0]
        buckets[key].append(val)
    return {k: sum(v) / len(v) for k, v in buckets.items()}
