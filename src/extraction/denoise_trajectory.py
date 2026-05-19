"""Capture depth + U-Net PCA frames at each denoising step."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers.models.resnet import ResnetBlock2D
from PIL import Image

from src.visualization.feature_pca import feature_map_to_pca_rgb, fit_pca_on_trajectory


def resolve_hook_layer(unet: torch.nn.Module, layer: str | None) -> str:
    if layer:
        for name, mod in unet.named_modules():
            if name == layer or name.endswith(layer):
                if isinstance(mod, ResnetBlock2D):
                    return name
        raise ValueError(f"Layer not found: {layer}")
    for name, mod in unet.named_modules():
        if name.endswith("mid_block.resnets.1") and isinstance(mod, ResnetBlock2D):
            return name
    for name, mod in unet.named_modules():
        if "mid_block.resnets" in name and isinstance(mod, ResnetBlock2D):
            return name
    raise RuntimeError("Could not find a mid_block ResNet layer for PCA hooks")


def _depth_to_uint8(depth_2d: np.ndarray, cmap_name: str = "Spectral") -> np.ndarray:
    import matplotlib.cm as cm

    d = np.asarray(depth_2d, dtype=np.float32)
    d = np.nan_to_num(d, nan=0.0, posinf=1.0, neginf=0.0)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    try:
        cmap = cm.colormaps[cmap_name]
    except AttributeError:
        cmap = cm.get_cmap(cmap_name)
    rgba = cmap(d)
    return (rgba[..., :3] * 255.0).astype(np.uint8)


@torch.no_grad()
def collect_denoise_trajectory(
    pipe: Any,
    rgb: torch.Tensor,
    n_steps: int,
    layer: str | None,
    device: str,
    generator: torch.Generator | None,
) -> tuple[list[dict], str]:
    """
    Run Marigold denoising and record per-step depth decode + hooked features.

    Returns list of dicts with keys: step_idx, timestep, feat [C,H,W] numpy, depth [H,W] numpy.
    """
    pipe.encode_empty_text()
    rgb = rgb.to(device, dtype=pipe.dtype)
    hook_name = resolve_hook_layer(pipe.unet, layer)
    captured: list[torch.Tensor] = []

    def _hook(_mod, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        if isinstance(t, torch.Tensor) and t.dim() == 4:
            captured.append(t[0].detach().float().cpu())

    target = None
    handle = None
    for name, mod in pipe.unet.named_modules():
        if name == hook_name:
            handle = mod.register_forward_hook(_hook)
            break
    if handle is None:
        raise RuntimeError(f"Failed to register hook on {hook_name}")

    try:
        rgb_latent = pipe.encode_rgb(rgb)
        target_latent = torch.randn(
            rgb_latent.shape, device=device, dtype=pipe.dtype, generator=generator
        )
        batch_embed = pipe.empty_text_embed.repeat(rgb_latent.shape[0], 1, 1).to(device)
        pipe.scheduler.set_timesteps(n_steps, device=device)

        records: list[dict] = []
        for step_idx, t in enumerate(pipe.scheduler.timesteps):
            captured.clear()
            unet_in = torch.cat([rgb_latent, target_latent], dim=1)
            noise_pred = pipe.unet(unet_in, t, encoder_hidden_states=batch_embed).sample
            if not captured:
                raise RuntimeError(f"No features captured at step {step_idx} ({hook_name})")
            feat = captured[-1].numpy()

            target_latent = pipe.scheduler.step(
                noise_pred, t, target_latent, generator=generator
            ).prev_sample
            depth = pipe.decode_depth(target_latent)[0, 0].float().cpu().numpy()

            records.append(
                {
                    "step_idx": step_idx,
                    "timestep": int(t.item()),
                    "feat": feat,
                    "depth": depth,
                }
            )
    finally:
        handle.remove()

    return records, hook_name


def export_trajectory_to_disk(
    records: list[dict],
    hook_name: str,
    rgb_uint8: np.ndarray,
    out_dir: Path,
    sample_id: str,
    gt_depth_uint8: np.ndarray | None = None,
    n_steps: int | None = None,
) -> Path:
    """Write PNG sequence + meta.json; update manifest at out_dir.parent or out_dir root."""
    sample_dir = out_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(rgb_uint8).save(sample_dir / "rgb.jpg", quality=92)
    if gt_depth_uint8 is not None:
        Image.fromarray(gt_depth_uint8).save(sample_dir / "gt_depth.png")

    feats = np.stack([r["feat"] for r in records], axis=0)
    pca, lo, hi = fit_pca_on_trajectory(feats)

    steps_meta = []
    for rec in records:
        idx = rec["step_idx"]
        step_dir = sample_dir / f"step_{idx:02d}"
        step_dir.mkdir(exist_ok=True)
        depth_rgb = _depth_to_uint8(rec["depth"])
        pca_rgb = feature_map_to_pca_rgb(rec["feat"], pca, lo, hi)
        Image.fromarray(depth_rgb).save(step_dir / "depth.png")
        Image.fromarray(pca_rgb).save(step_dir / "pca.png")
        steps_meta.append({"step_idx": idx, "timestep": rec["timestep"], "dir": step_dir.name})

    meta = {
        "sample_id": sample_id,
        "hook_layer": hook_name,
        "n_steps": n_steps or len(records),
        "steps": steps_meta,
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return sample_dir


def update_manifest(viz_root: Path, sample_id: str, caption: str = "") -> None:
    manifest_path = viz_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"samples": []}
    ids = {s["id"] for s in manifest["samples"]}
    entry = {"id": sample_id, "caption": caption or sample_id}
    if sample_id not in ids:
        manifest["samples"].append(entry)
    else:
        for s in manifest["samples"]:
            if s["id"] == sample_id:
                s["caption"] = caption or sample_id
    manifest_path.write_text(json.dumps(manifest, indent=2))
