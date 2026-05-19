"""Hook U-Net ResNet and cross-attention blocks; cache activations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.resnet import ResnetBlock2D


CANONICAL_SIZE = (64, 64)


def list_resnet_hook_layers(unet: torch.nn.Module) -> list[str]:
    return [
        f"res::{name}"
        for name, module in unet.named_modules()
        if isinstance(module, ResnetBlock2D)
    ]


def subsample_layers(layer_names: list[str], max_layers: int = 12) -> list[str]:
    if len(layer_names) <= max_layers:
        return layer_names
    step = len(layer_names) / max_layers
    return [layer_names[int(i * step)] for i in range(max_layers)]


class FeatureExtractor:
    def __init__(
        self,
        unet: torch.nn.Module,
        *,
        layers: set[str] | None = None,
    ) -> None:
        self.unet = unet
        self.layers = layers
        self.cache: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._handles: list[Any] = []

    def _hook(self, name: str):
        def fn(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor) and t.dim() == 4:
                t = F.interpolate(t, size=CANONICAL_SIZE, mode="bilinear", align_corners=False)
                self.cache[name].append(t.detach().cpu().half())
        return fn

    def register_hooks(self) -> None:
        self.clear()
        for name, module in self.unet.named_modules():
            if isinstance(module, ResnetBlock2D):
                hook_name = f"res::{name}"
                if self.layers is not None and hook_name not in self.layers:
                    continue
                self._handles.append(module.register_forward_hook(self._hook(hook_name)))
            elif isinstance(module, BasicTransformerBlock):
                if self.layers is not None:
                    continue
                hook_name = f"attn::{name}"
                self._handles.append(module.register_forward_hook(self._hook(hook_name)))

    def clear_cache(self) -> None:
        """Clear activations only — keep hooks registered."""
        self.cache.clear()

    def clear(self) -> None:
        self.remove_hooks()
        self.cache.clear()

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get_latest(self) -> dict[str, torch.Tensor]:
        return {k: v[-1].float() for k, v in self.cache.items() if v}
